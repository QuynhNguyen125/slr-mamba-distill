"""
Stage 1 — Transfer Matrix Alignment (MOHAWK, layer-by-layer).

Experimental Protocol: Full Convergence Study (no early stopping).

Logs to wandb (X axis = stage1/epoch):
  stage1/train_loss              — mean Frobenius loss per epoch (train)
  stage1/val_loss                — mean Frobenius loss per epoch (val)
  stage1/lr                      — learning rate
  stage1/frob_block_{l:02d}      — per-block Frobenius distance (train)
  stage1/rel_frob_block_{l:02d}  — relative Frobenius = ||M-A||_F / ||A||_F
  stage1/matrices_ep{e:03d}      — heatmap: teacher|student|diff (at viz_epochs)
  stage1/block_bar_ep{e:03d}     — bar plot per-block Frobenius (at checkpoint_epochs)
"""

import os
import torch
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

from models.student import BiMambaSLR
from models.teacher import TeacherModel
from distillation.losses import frobenius_loss


def freeze_non_temporal_mamba(student: BiMambaSLR):
    """
    Đảm bảo chỉ temporal_mamba trainable.
    Thường đã được gọi bởi load_teacher_weights() — hàm này là safety net.
    """
    for name, param in student.named_parameters():
        param.requires_grad_("temporal_mamba" in name)


def _make_matrix_figure(teacher_attn_all, student_trans_all, blocks_show, epoch,
                        head=0, bm=0, v=0):
    """
    Figure 3: heatmap grid — selected blocks × (teacher | student | diff).
    Returns a matplotlib Figure.
    """
    n = len(blocks_show)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n), squeeze=False)
    fig.suptitle(f"Transfer Matrix Alignment — Epoch {epoch}", fontsize=11, y=1.01)
    for row, l in enumerate(blocks_show):
        t = teacher_attn_all[l][bm, v, head].float().cpu().numpy()
        s = student_trans_all[l][bm, v, head].float().cpu().numpy()
        d = s - t
        vmin, vmax = min(t.min(), s.min()), max(t.max(), s.max())
        abs_d = np.abs(d).max() + 1e-8
        for col, (mat, title, cmap, lo, hi) in enumerate([
            (t, f"Teacher  block {l} h{head}", "Blues",  vmin,  vmax),
            (s, f"Student  block {l} h{head}", "Blues",  vmin,  vmax),
            (d, "Difference",                  "RdBu_r", -abs_d, abs_d),
        ]):
            im = axes[row][col].imshow(mat, aspect="auto", cmap=cmap, vmin=lo, vmax=hi)
            axes[row][col].set_title(title, fontsize=9)
            plt.colorbar(im, ax=axes[row][col], fraction=0.046, pad=0.04)
    plt.tight_layout()
    return fig


def _make_block_bar_figure(avg_frob, avg_rel_frob, epoch):
    """
    Figure 4: per-block Frobenius bar plot at checkpoint epochs.
    Left: absolute Frobenius distance.
    Right: relative Frobenius = ||M-A||_F / ||A||_F.
    """
    n = len(avg_frob)
    blocks = list(range(n))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Per-Block Alignment — Epoch {epoch}", fontsize=11)

    bars1 = ax1.bar(blocks, avg_frob, color="steelblue", edgecolor="white")
    ax1.set_xlabel("Block Index")
    ax1.set_ylabel("Frobenius Distance")
    ax1.set_title("Absolute: ||M - A||_F")
    ax1.set_xticks(blocks)
    for bar, val in zip(bars1, avg_frob):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"{val:.3f}", ha="center", va="bottom", fontsize=7)

    bars2 = ax2.bar(blocks, avg_rel_frob, color="coral", edgecolor="white")
    ax2.set_xlabel("Block Index")
    ax2.set_ylabel("Relative Frobenius")
    ax2.set_title("Relative: ||M - A||_F / ||A||_F")
    ax2.set_xticks(blocks)
    for bar, val in zip(bars2, avg_rel_frob):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"{val:.3f}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    return fig


def train_stage1(
    student: BiMambaSLR,
    teacher: TeacherModel,
    dataloader: DataLoader,
    val_dataloader: DataLoader = None,
    device: str = "cuda",
    lr: float = 1e-3,
    num_epochs: int = 100,
    log_freq: int = 50,
    checkpoint_epochs: list = None,   # save ckpt + bar plot tại các epoch này
    viz_epochs: list = None,           # log matrix heatmap tại các epoch này
    viz_blocks: list = None,           # blocks để visualize (default: 4 đều nhau)
    save_path: str = None,             # best val_loss checkpoint path
    save_dir: str = None,              # thư mục lưu milestone checkpoints
    wandb_run=None,
):
    """
    Phase 1 — Full Convergence Study (no early stopping).

    Checkpoints saved:
      - best val_loss → save_path  (e.g. checkpoints/student_stage1_best.pth)
      - milestone epochs → save_dir/student_stage1_ep{e:03d}.pth

    Figures logged to wandb:
      - Figure 1: train/val loss curves (per epoch, automatic)
      - Figure 2: per-block frob curves (per epoch, automatic)
      - Figure 3: matrix heatmap (at viz_epochs)
      - Figure 4: per-block bar plot (at checkpoint_epochs)
    """
    # Defaults
    if checkpoint_epochs is None:
        checkpoint_epochs = [10, 25, 50, 100]
    if viz_epochs is None:
        viz_epochs = [1, 25, 100]

    teacher.eval()
    teacher.to(device)
    student.to(device)

    freeze_non_temporal_mamba(student)
    student.train()

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, student.parameters()),
        lr=lr, weight_decay=0.01,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr * 0.01)

    n_blocks = len(student.blocks)
    if viz_blocks is None:
        step_v = max(1, n_blocks // 4)
        viz_blocks = list(range(0, n_blocks, step_v))[:4]

    trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"[Stage1] Trainable params : {trainable:,}  (temporal_mamba only)")
    print(f"[Stage1] Epochs           : {num_epochs}  (no early stopping)")
    print(f"[Stage1] Checkpoint epochs: {checkpoint_epochs}")
    print(f"[Stage1] Matrix viz epochs: {viz_epochs}")

    best_val_loss = float("inf")
    global_step   = 0

    for epoch in range(num_epochs):
        student.train()
        epoch_loss       = 0.0
        frob_per_block   = [0.0] * n_blocks   # absolute Frobenius (sum over steps)
        rel_frob_per_block = [0.0] * n_blocks  # relative Frobenius (sum over steps)

        for step, batch in enumerate(dataloader):
            x = _get_x(batch, device)
            optimizer.zero_grad()

            with torch.no_grad():
                t_out = teacher(x, return_attn=True, return_hidden_states=True)
            tm_teacher_all      = t_out["temporal_attn_matrices"]
            teacher_hidden_states = t_out["hidden_states"]

            loss = torch.tensor(0.0, device=device)
            n = min(n_blocks, len(tm_teacher_all))

            for l in range(n):
                student_input = teacher_hidden_states[l].to(device)
                s_out = student.blocks[l](
                    hidden_states=student_input,
                    run_mlp_component=False,
                    return_transfer_matrix=True,
                )
                tm_s = s_out["transfer_matrix"]
                tm_t = tm_teacher_all[l].to(device)

                block_loss = frobenius_loss(tm_s, tm_t)
                loss = loss + block_loss

                # ── relative Frobenius: ||M - A||_F / ||A||_F ────────────
                with torch.no_grad():
                    teacher_norm = torch.linalg.matrix_norm(
                        tm_t.float(), ord="fro"
                    ).mean().item()
                rel_frob = block_loss.item() / (teacher_norm + 1e-8)

                frob_per_block[l]     += block_loss.item()
                rel_frob_per_block[l] += rel_frob

            loss = loss / n
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()

            epoch_loss  += loss.item()
            global_step += 1

            if (step + 1) % log_freq == 0:
                print(
                    f"[Stage1] Epoch {epoch+1}/{num_epochs}  "
                    f"Step {step+1}/{len(dataloader)}  "
                    f"Loss: {loss.item():.4f}"
                )

        scheduler.step()

        # ── Epoch-level averages ──────────────────────────────────────────
        n_steps      = len(dataloader)
        avg_loss     = epoch_loss / n_steps
        avg_frob     = [f / n_steps for f in frob_per_block]
        avg_rel_frob = [r / n_steps for r in rel_frob_per_block]

        print(f"[Stage1] Epoch {epoch+1}/{num_epochs} — train_loss: {avg_loss:.4f}  "
              f"lr: {optimizer.param_groups[0]['lr']:.2e}")

        # ── Validation loss ───────────────────────────────────────────────
        val_loss = None
        if val_dataloader is not None:
            val_loss = _compute_val_loss(student, teacher, val_dataloader, device, n_blocks)
            print(f"[Stage1] Epoch {epoch+1}/{num_epochs} — val_loss:   {val_loss:.4f}")

        # ── Wandb: epoch-level (Figures 1 & 2) ───────────────────────────
        if wandb_run is not None:
            log_dict = {
                "stage1/epoch":      epoch + 1,
                "stage1/train_loss": avg_loss,
                "stage1/lr":         optimizer.param_groups[0]["lr"],
            }
            if val_loss is not None:
                log_dict["stage1/val_loss"] = val_loss
            for l in range(n_blocks):
                log_dict[f"stage1/frob_block_{l:02d}"]     = avg_frob[l]
                log_dict[f"stage1/rel_frob_block_{l:02d}"] = avg_rel_frob[l]
            wandb_run.log(log_dict)

        # ── Figure 3: matrix visualization at viz_epochs ─────────────────
        if (epoch + 1) in viz_epochs and wandb_run is not None:
            _log_matrix_viz(
                student, teacher, dataloader, device,
                viz_blocks, epoch, wandb_run,
            )

        # ── Figure 4 + milestone checkpoint at checkpoint_epochs ─────────
        if (epoch + 1) in checkpoint_epochs:
            # Bar plot
            if wandb_run is not None:
                _log_block_bar_plot(avg_frob, avg_rel_frob, epoch + 1, wandb_run)

            # Milestone checkpoint
            if save_dir is not None:
                ep_path = os.path.join(save_dir, f"student_stage1_ep{epoch+1:03d}.pth")
                _save(student, ep_path)
                print(f"[Stage1] Milestone checkpoint → {ep_path}")

        # ── Best val_loss checkpoint ──────────────────────────────────────
        monitor = val_loss if val_loss is not None else avg_loss
        if monitor < best_val_loss:
            best_val_loss = monitor
            if save_path:
                _save(student, save_path)
                print(f"[Stage1] ✓ Best checkpoint (loss={best_val_loss:.4f}) → {save_path}")

    print(f"\n[Stage1] Training done. Best val_loss = {best_val_loss:.4f}")
    return student


@torch.no_grad()
def _compute_val_loss(student, teacher, val_loader, device, n_blocks):
    """
    Tính Frobenius loss trên validation set.

    Hỗ trợ hai kiểu multi-crop từ Sign_Dataset:
      Kiểu A — flat k_copies: (B, C, T*n, V, M) → split theo T
      Kiểu B — stacked:       (B, n, C, T, V, M) → flatten B*n
    """
    student.eval()
    total = 0.0
    count = 0
    seq_len = student.seq_len - 1   # trừ CLS token

    for batch in val_loader:
        x = _get_x(batch, device)

        if x.ndim == 5:
            B, C, T_total, V, M_dim = x.shape
            if T_total > seq_len and T_total % seq_len == 0:
                n_copies = T_total // seq_len
                x = (x.view(B, C, n_copies, seq_len, V, M_dim)
                      .permute(0, 2, 1, 3, 4, 5)
                      .contiguous()
                      .view(B * n_copies, C, seq_len, V, M_dim))
        elif x.ndim == 6:
            B, n_copies, C, T, V, M_dim = x.shape
            x = x.contiguous().view(B * n_copies, C, T, V, M_dim)

        t_out = teacher(x, return_attn=True, return_hidden_states=True)
        tm_teacher_all = t_out["temporal_attn_matrices"]
        teacher_hidden = t_out["hidden_states"]

        n = min(n_blocks, len(tm_teacher_all))
        for l in range(n):
            s_out = student.blocks[l](
                hidden_states=teacher_hidden[l].to(device),
                run_mlp_component=False,
                return_transfer_matrix=True,
            )
            total += frobenius_loss(s_out["transfer_matrix"], tm_teacher_all[l].to(device)).item()
            count += 1

    student.train()
    return total / max(count, 1)


@torch.no_grad()
def _log_matrix_viz(student, teacher, dataloader, device, viz_blocks, epoch, wandb_run):
    """Figure 3: matrix heatmap tại viz_epochs."""
    import wandb
    student.eval()

    x = _get_x(next(iter(dataloader)), device)
    t_out = teacher(x, return_attn=True, return_hidden_states=True)
    tm_teacher_all = t_out["temporal_attn_matrices"]
    teacher_hidden = t_out["hidden_states"]

    student_trans = []
    for l in range(min(len(student.blocks), len(tm_teacher_all))):
        s_out = student.blocks[l](
            hidden_states=teacher_hidden[l].to(device),
            run_mlp_component=False,
            return_transfer_matrix=True,
        )
        student_trans.append(s_out["transfer_matrix"].cpu())

    fig = _make_matrix_figure(
        [m.cpu() for m in tm_teacher_all],
        student_trans,
        blocks_show=viz_blocks,
        epoch=epoch + 1,
    )
    wandb_run.log({
        f"stage1/matrices_ep{epoch+1:03d}": wandb.Image(fig, caption=f"Epoch {epoch+1}"),
    })
    plt.close(fig)
    student.train()


def _log_block_bar_plot(avg_frob, avg_rel_frob, epoch, wandb_run):
    """Figure 4: per-block bar plot tại checkpoint epochs."""
    import wandb
    fig = _make_block_bar_figure(avg_frob, avg_rel_frob, epoch)
    wandb_run.log({
        f"stage1/block_bar_ep{epoch:03d}": wandb.Image(fig, caption=f"Epoch {epoch}"),
    })
    plt.close(fig)


def _get_x(batch, device):
    if isinstance(batch, dict):
        return batch["skeleton_data"].to(device).float()
    return batch[0].to(device).float()


def _save(model, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"model_state_dict": model.state_dict()}, path)
