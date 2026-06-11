"""
Stage 2 — Hidden State Alignment (MOHAWK).

Theo phi-mamba/assets/mohawk_stage2.py (default: freeze_mlp=True):

    Với mỗi block l:
        student_input  = teacher.hidden_states[l]           ← input đến block l
        student_output = student.blocks[l](student_input,
                             run_mlp_component=False)       ← spatial+temporal only (no FFN)
        teacher_target = teacher.pre_ffn_states[l]          ← phi-mamba: all_attn_outputs[l]
        loss = ||student_output - teacher_target||_2        ← per-token L2 norm

    freeze_mlp=True  (phi-mamba default):
        frozen : feed_forward_network, norm_layer3, embedding, fc
        trained: multihead_self_attention1, temporal_mamba, norm_layer1, norm_layer2
        target : pre_ffn_states[l]  (state trước FFN = sau temporal attention)
        run_mlp_component=False

    freeze_mlp=False (full block alignment):
        frozen : embedding, fc
        trained: tất cả block params
        target : block_outputs[l]   (full block output)
        run_mlp_component=True

Backward per-block (phi-mamba pattern) để tiết kiệm memory:
    → backward() ngay sau mỗi block, chỉ giữ 1 computation graph tại 1 thời điểm

Logs to wandb (X axis = stage2/epoch, khai báo bằng define_metric):
    stage2/train_loss        — train loss trung bình mỗi epoch
    stage2/val_loss          — val loss mỗi epoch
    stage2/lr                — learning rate
    stage2/block_{l:02d}_loss — loss từng block (trung bình epoch)
"""

import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from models.student import BiMambaSLR
from models.teacher import TeacherModel
from distillation.losses import hidden_state_l2_loss


def set_stage2_trainable(student: BiMambaSLR, freeze_mlp: bool = True):
    """
    Theo phi-mamba Stage 2:

    freeze_mlp=True  (phi-mamba default):
        frozen  — "mlp" (feed_forward_network), "norm_layer3" (post-FFN norm),
                  "embedding", "fc"
        trained — "multihead_self_attention1", "norm_layer1",
                  "temporal_mamba", "norm_layer2"
        ↳ Chỉ train temporal SSM + spatial attn; FFN giữ nguyên từ Stage 1.

    freeze_mlp=False (full block, ít dùng):
        frozen  — "embedding", "fc"
        trained — tất cả params trong blocks
    """
    freeze_keys_base = ["embedding", "fc"]
    freeze_keys_ffn  = ["feed_forward_network", "norm_layer3"]

    for name, param in student.named_parameters():
        frozen_base = any(k in name for k in freeze_keys_base)
        frozen_ffn  = freeze_mlp and any(k in name for k in freeze_keys_ffn)
        if frozen_base or frozen_ffn:
            param.requires_grad_(False)
        else:
            param.requires_grad_(True)


def train_stage2(
    student: BiMambaSLR,
    teacher: TeacherModel,
    dataloader: DataLoader,
    val_dataloader: DataLoader = None,
    device: str = "cuda",
    lr: float = 5e-4,
    num_epochs: int = 20,
    freeze_mlp: bool = True,    # True = phi-mamba default: chỉ train temporal SSM + spatial attn
    log_freq: int = 10,
    wandb_run=None,
    save_path: str = None,
):
    teacher.eval()
    teacher.to(device)
    student.to(device)

    set_stage2_trainable(student, freeze_mlp=freeze_mlp)
    student.train()

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, student.parameters()),
        lr=lr, weight_decay=0.01,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    n_blocks = len(student.blocks)
    global_step = 0

    trainable_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
    target_desc = "pre_ffn_states (phi-mamba: all_attn_outputs)" if freeze_mlp else "block_outputs (full block)"
    print(f"[Stage2] freeze_mlp={freeze_mlp}")
    print(f"[Stage2] Target     : {target_desc}")
    print(f"[Stage2] Trainable  : {trainable_params:,} params")

    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        student.train()
        epoch_loss    = 0.0
        mse_per_block = [0.0] * n_blocks

        for step, batch in enumerate(dataloader):
            x = _get_x(batch, device)

            # ── Teacher forward ───────────────────────────────────────
            with torch.no_grad():
                t_out = teacher(x, return_attn=False, return_hidden_states=True)

            block_inputs  = t_out["hidden_states"]    # list[n_blocks]: input → block l
            # Target selection (phi-mamba mohawk_stage2.py):
            #   freeze_mlp=True  → all_attn_outputs[l]    = pre_ffn_states[l]
            #   freeze_mlp=False → all_hidden_states[l+1] = block_outputs[l]
            teacher_targets = (
                t_out["pre_ffn_states"] if freeze_mlp else t_out["block_outputs"]
            )

            # ── Backward per-block (phi-mamba pattern) ────────────────
            # loss.backward() ngay sau mỗi block → chỉ 1 graph trong memory
            optimizer.zero_grad()
            n         = min(n_blocks, len(block_inputs), len(teacher_targets))
            step_loss = 0.0

            for l in range(n):
                student_input  = block_inputs[l].to(device)
                teacher_target = teacher_targets[l].to(device)

                s_out = student.blocks[l](
                    hidden_states=student_input,
                    run_mlp_component=not freeze_mlp,  # False when freeze_mlp=True (phi-mamba)
                    return_transfer_matrix=False,
                )
                student_output = s_out["hidden_states"]

                # L2 norm per-token, chia n để gradient scale nhất quán
                block_loss = hidden_state_l2_loss(student_output, teacher_target) / n

                block_loss.backward()   # giải phóng graph ngay

                block_val = block_loss.item()
                step_loss += block_val
                mse_per_block[l] += block_val

                del s_out, student_output, teacher_target, block_loss
                if device != "cpu":
                    torch.cuda.empty_cache()

            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()

            epoch_loss  += step_loss
            global_step += 1

            if (step + 1) % log_freq == 0:
                print(
                    f"[Stage2] Epoch {epoch+1}/{num_epochs}  "
                    f"Step {step+1}/{len(dataloader)}  "
                    f"train_loss: {step_loss:.4f}"
                )

        scheduler.step()

        avg_train_loss = epoch_loss / len(dataloader)
        avg_mse        = [m / len(dataloader) for m in mse_per_block]

        # ── Validation ────────────────────────────────────────────────
        val_loss = None
        if val_dataloader is not None:
            val_loss = _compute_val_loss(
                student, teacher, val_dataloader, device, n_blocks, freeze_mlp=freeze_mlp
            )

        # ── Console log ───────────────────────────────────────────────
        val_str = f"  val_loss: {val_loss:.4f}" if val_loss is not None else ""
        print(
            f"[Stage2] Epoch {epoch+1}/{num_epochs} — "
            f"train_loss: {avg_train_loss:.4f}{val_str}"
        )

        # ── Wandb log — epoch là X axis (define_metric ở run_stage2.py) ──
        if wandb_run is not None:
            log_dict = {
                "stage2/epoch":      epoch + 1,   # X axis (declared via define_metric)
                "stage2/train_loss": avg_train_loss,
                "stage2/lr":         optimizer.param_groups[0]["lr"],
            }
            if val_loss is not None:
                log_dict["stage2/val_loss"] = val_loss
            for l, v in enumerate(avg_mse):
                log_dict[f"stage2/block_{l:02d}_loss"] = v
            wandb_run.log(log_dict)

        # ── Save best checkpoint (theo val_loss nếu có, else train_loss) ──
        monitor = val_loss if val_loss is not None else avg_train_loss
        if monitor < best_val_loss:
            best_val_loss = monitor
            if save_path:
                _save(student, save_path)
                print(f"[Stage2] ✓ Best checkpoint saved (loss={best_val_loss:.4f}) → {save_path}")

    # Luôn save final checkpoint
    if save_path:
        final_path = save_path.replace(".pth", "_final.pth")
        _save(student, final_path)
        print(f"[Stage2] Final checkpoint saved → {final_path}")

    return student


@torch.no_grad()
def _compute_val_loss(student, teacher, val_loader, device, n_blocks, freeze_mlp: bool = True):
    student.eval()
    total = 0.0
    count = 0
    seq_len = student.seq_len - 1   # seq_len sau khi thêm CLS token

    for batch in val_loader:
        x = _get_x(batch, device)

        # k_copies: (B, C, T*n, V, M) → (B*n, C, T, V, M)
        if x.ndim == 5:
            B, C, T_total, V, M_dim = x.shape
            if T_total > seq_len and T_total % seq_len == 0:
                n_copies = T_total // seq_len
                x = (x.view(B, C, n_copies, seq_len, V, M_dim)
                      .permute(0, 2, 1, 3, 4, 5)
                      .contiguous()
                      .view(B * n_copies, C, seq_len, V, M_dim))

        t_out = teacher(x, return_attn=False, return_hidden_states=True)
        block_inputs   = t_out["hidden_states"]
        teacher_targets = (
            t_out["pre_ffn_states"] if freeze_mlp else t_out["block_outputs"]
        )

        n = min(n_blocks, len(block_inputs), len(teacher_targets))
        for l in range(n):
            s_out = student.blocks[l](
                hidden_states=block_inputs[l].to(device),
                run_mlp_component=not freeze_mlp,
                return_transfer_matrix=False,
            )
            target = teacher_targets[l].to(device)
            total += hidden_state_l2_loss(s_out["hidden_states"], target).item()
            count += 1

    student.train()
    return total / max(count, 1)


def _get_x(batch, device):
    if isinstance(batch, dict):
        return batch["skeleton_data"].to(device).float()
    return batch[0].to(device).float()


def _save(model, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"model_state_dict": model.state_dict()}, path)
