"""
Stage 1 — Transfer Matrix Alignment (MOHAWK, layer-by-layer).

Logs to wandb:
  stage1/loss_step        — Frobenius loss per optimizer step
  stage1/loss_epoch       — mean loss per epoch
  stage1/lr               — learning rate
  stage1/frob_block_{l}   — per-block Frobenius distance (logged each epoch)
  stage1/matrices         — heatmap grid (teacher vs student, logged every viz_freq epochs)
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


def _make_matrix_figure(teacher_attn_all, student_trans_all, blocks_show, head=0, bm=0, v=0):
    """
    Quick heatmap grid: selected blocks × (teacher | student | diff).
    Returns a matplotlib Figure.
    """
    n = len(blocks_show)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n), squeeze=False)
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


def train_stage1(
    student: BiMambaSLR,
    teacher: TeacherModel,
    dataloader: DataLoader,
    val_dataloader: DataLoader = None,
    device: str = "cuda",
    lr: float = 1e-3,
    num_epochs: int = 10,
    log_freq: int = 50,
    viz_freq: int = 2,
    viz_blocks: list = None,
    wandb_run=None,
    save_path: str = None,
):
    teacher.eval()
    teacher.to(device)
    student.to(device)

    freeze_non_temporal_mamba(student)
    student.train()

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, student.parameters()),
        lr=lr, weight_decay=0.01,
    )

    n_blocks = len(student.blocks)
    if viz_blocks is None:
        step_v = max(1, n_blocks // 4)
        viz_blocks = list(range(0, n_blocks, step_v))[:4]

    global_step = 0

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        frob_per_block = [0.0] * n_blocks

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
                frob_per_block[l] += block_loss.item()

            loss = loss / n
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            global_step += 1

            if (step + 1) % log_freq == 0:
                print(
                    f"[Stage1] Epoch {epoch+1}/{num_epochs}  "
                    f"Step {step+1}/{len(dataloader)}  "
                    f"Loss: {loss.item():.4f}"
                )
                if wandb_run is not None:
                    wandb_run.log({
                        "stage1/loss_step": loss.item(),
                        "stage1/lr":        optimizer.param_groups[0]["lr"],
                    }, step=global_step)

        # ── Epoch-level logging ───────────────────────────────────────
        avg_loss = epoch_loss / len(dataloader)
        avg_frob = [f / len(dataloader) for f in frob_per_block]
        print(f"[Stage1] Epoch {epoch+1}/{num_epochs} — train_loss: {avg_loss:.4f}")

        # ── Validation loss ───────────────────────────────────────────
        val_loss = None
        if val_dataloader is not None:
            val_loss = _compute_val_loss(student, teacher, val_dataloader, device, n_blocks)
            print(f"[Stage1] Epoch {epoch+1}/{num_epochs} — val_loss:   {val_loss:.4f}")

        if wandb_run is not None:
            log_dict = {
                "stage1/train_loss": avg_loss,
                "stage1/epoch":      epoch + 1,
            }
            if val_loss is not None:
                log_dict["stage1/val_loss"] = val_loss
            for l, fval in enumerate(avg_frob):
                log_dict[f"stage1/frob_block_{l:02d}"] = fval
            wandb_run.log(log_dict, step=global_step)

        # ── Matrix visualization ──────────────────────────────────────
        if (epoch + 1) % viz_freq == 0 and wandb_run is not None:
            _log_matrix_viz(
                student, teacher, dataloader, device,
                viz_blocks, epoch, global_step, wandb_run,
            )

    if save_path:
        _save(student, save_path)
        print(f"[Stage1] Checkpoint saved → {save_path}")

    return student


@torch.no_grad()
def _compute_val_loss(student, teacher, val_loader, device, n_blocks):
    """
    Tính Frobenius loss trên validation set.

    Hỗ trợ hai kiểu multi-crop từ Sign_Dataset:

    Kiểu A — flat k_copies (Sign_Dataset mặc định):
        shape: (B, C, T*n_copies, V, M)   — 5D
        Sign_Dataset nối tất cả copies dọc theo trục thời gian.
        → split T*n_copies thành n_copies clip riêng biệt T frames.

    Kiểu B — stacked n_copies (legacy):
        shape: (B, n_copies, C, T, V, M)  — 6D
        → flatten B và n_copies.
    """
    student.eval()
    total = 0.0
    count = 0

    # seq_len gốc (không tính cls token) — dùng để detect flat k_copies
    seq_len = student.seq_len - 1   # student lưu seq_len+1

    for batch in val_loader:
        x = _get_x(batch, device)

        # ── Kiểu A: flat k_copies → (B, C, T*n, V, M) ────────────────
        if x.ndim == 5:
            B, C, T_total, V, M_dim = x.shape
            if T_total > seq_len and T_total % seq_len == 0:
                n_copies = T_total // seq_len
                # (B, C, n_copies, seq_len, V, M) → (B*n_copies, C, seq_len, V, M)
                x = (x.view(B, C, n_copies, seq_len, V, M_dim)
                      .permute(0, 2, 1, 3, 4, 5)
                      .contiguous()
                      .view(B * n_copies, C, seq_len, V, M_dim))

        # ── Kiểu B: stacked n_copies → (B, n_copies, C, T, V, M) ─────
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
def _log_matrix_viz(student, teacher, dataloader, device, viz_blocks, epoch, step, wandb_run):
    """Grab one batch, collect matrices, log heatmap to wandb."""
    import wandb
    student.eval()

    x = _get_x(next(iter(dataloader)), device)
    t_out = teacher(x, return_attn=True, return_hidden_states=True)
    tm_teacher_all    = t_out["temporal_attn_matrices"]
    teacher_hidden    = t_out["hidden_states"]

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
    )
    wandb_run.log({
        "stage1/matrices": wandb.Image(fig, caption=f"Epoch {epoch+1}"),
    }, step=step)
    plt.close(fig)
    student.train()


def _get_x(batch, device):
    # Batch có thể là dict {"skeleton_data": ..., "label": ...} hoặc tuple (x, label)
    if isinstance(batch, dict):
        return batch["skeleton_data"].to(device).float()
    return batch[0].to(device).float()


def _save(model, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"model_state_dict": model.state_dict()}, path)
