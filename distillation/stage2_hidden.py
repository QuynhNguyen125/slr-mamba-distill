"""
Stage 2 — Hidden State Alignment (MOHAWK, block-wise).

Theo phi-mamba Stage 2 skeleton:

    freeze_mlp=True  (Phase A — mixer-only):
        student_input  = teacher.hidden_states[l]      ← input của block l
        student_output = student.blocks[l](input, run_mlp=False)
                       = output SAU spatial MHA + temporal Mamba, TRƯỚC FFN
        teacher_target = teacher.pre_ffn_states[l]     ← phi-mamba: all_attn_outputs[l]
                       = output SAU temporal attention + norm2 + transpose, TRƯỚC FFN
        loss = MSE(student_output, teacher_target)

    freeze_mlp=False (Phase B — mixer+FFN):
        student_output = student.blocks[l](input, run_mlp=True)
                       = full block output (spatial + temporal + FFN + B2T)
        teacher_target = teacher.block_outputs[l]      ← full block output
        loss = MSE(student_output, teacher_target)

Mapping với phi-mamba:
    teacher.pre_ffn_states  ↔  teacher_outputs.all_attn_outputs
    teacher.block_outputs   ↔  teacher_outputs.all_hidden_states[l+1]
    run_mlp_component       ↔  not freeze_mlp (đồng nhất với phi-mamba)

Logs to wandb:
  stage2/loss_step, stage2/loss_epoch, stage2/lr
  stage2/mse_block_{l:02d}
  stage2/phase  ("mixer-only" | "mixer+FFN")
"""

import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from models.student import BiMambaSLR
from models.teacher import TeacherModel
from distillation.losses import hidden_state_l2_loss


def set_stage2_trainable(student: BiMambaSLR, freeze_mlp: bool):
    """
    Phase A (freeze_mlp=True):
        trainable  = temporal_mamba
        frozen     = embedding, spatial MHA, norm1/2, FFN, norm3, fc

    Phase B (freeze_mlp=False):
        trainable  = temporal_mamba + feed_forward_network + norm_layer3
        frozen     = embedding, spatial MHA, norm1/2, fc

    Mapping với phi-mamba:
        frozen "mlp"            → feed_forward_network   (Phase A only)
        frozen "input_layernorm"→ norm_layer1/2/3        (norm3 unfrozen in Phase B)
        frozen "embedding"      → embedding
        frozen "lm_head"        → fc
    """
    for name, param in student.named_parameters():
        if "temporal_mamba" in name:
            param.requires_grad_(True)
        elif not freeze_mlp and any(
            k in name for k in ["feed_forward_network", "norm_layer3"]
        ):
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)


def train_stage2(
    student: BiMambaSLR,
    teacher: TeacherModel,
    dataloader: DataLoader,
    val_dataloader: DataLoader = None,
    device: str = "cuda",
    lr: float = 5e-4,
    num_epochs: int = 10,
    freeze_mlp: bool = True,
    log_freq: int = 50,
    wandb_run=None,
    save_path: str = None,
):
    teacher.eval()
    teacher.to(device)
    student.to(device)

    set_stage2_trainable(student, freeze_mlp)
    student.train()

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, student.parameters()),
        lr=lr, weight_decay=0.01,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    n_blocks = len(student.blocks)
    mode     = "mixer-only" if freeze_mlp else "mixer+FFN"
    global_step = 0

    trainable_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"[Stage2/{mode}] Trainable params: {trainable_params:,}")

    for epoch in range(num_epochs):
        student.train()
        epoch_loss    = 0.0
        mse_per_block = [0.0] * n_blocks

        for step, batch in enumerate(dataloader):
            x = _get_x(batch, device)

            # ── Teacher forward — thu thập hidden states ──────────────
            with torch.no_grad():
                t_out = teacher(x, return_attn=False, return_hidden_states=True)

            block_inputs    = t_out["hidden_states"]    # list[n_blocks]  input  → block l
            pre_ffn_states  = t_out["pre_ffn_states"]  # list[n_blocks]  TRƯỚC FFN
            block_outputs   = t_out["block_outputs"]   # list[n_blocks]  full block output

            # Guard
            assert len(pre_ffn_states) == n_blocks, (
                f"[Stage2] pre_ffn_states={len(pre_ffn_states)} ≠ n_blocks={n_blocks}. "
                "Kiểm tra hook feed_forward_network trong TeacherModel."
            )

            # ── Backward per-block (phi-mamba pattern) ───────────────
            # Gọi loss.backward() ngay sau mỗi block thay vì accumulate
            # tất cả n blocks rồi backward một lần.
            # → chỉ 1 computation graph trong memory tại mỗi thời điểm
            # → giảm peak memory từ O(n_blocks) xuống O(1)
            optimizer.zero_grad()
            n         = min(n_blocks, len(block_inputs))
            step_loss = 0.0   # chỉ để log, không dùng cho backward

            for l in range(n):
                student_input = block_inputs[l].to(device)

                # ── Target theo phi-mamba ─────────────────────────────
                if freeze_mlp:
                    teacher_target = pre_ffn_states[l].to(device)
                else:
                    teacher_target = block_outputs[l].to(device)

                s_out = student.blocks[l](
                    hidden_states=student_input,
                    run_mlp_component=not freeze_mlp,
                    return_transfer_matrix=False,
                )
                student_output = s_out["hidden_states"]

                # Chia n để gradient scale nhất quán với loss trung bình
                block_loss = hidden_state_l2_loss(student_output, teacher_target) / n

                # Backward ngay — giải phóng graph của block l trước block l+1
                block_loss.backward()

                block_val = block_loss.item()
                step_loss += block_val
                mse_per_block[l] += block_val

                # Giải phóng fragment memory giữa các blocks
                del s_out, student_output, teacher_target, block_loss
                if device != "cpu":
                    torch.cuda.empty_cache()

            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()

            epoch_loss  += step_loss
            global_step += 1

            if (step + 1) % log_freq == 0:
                print(
                    f"[Stage2/{mode}] Epoch {epoch+1}/{num_epochs}  "
                    f"Step {step+1}/{len(dataloader)}  "
                    f"Loss: {step_loss:.4f}"
                )
                if wandb_run is not None:
                    wandb_run.log({
                        "stage2/loss_step": step_loss,
                        "stage2/lr":        optimizer.param_groups[0]["lr"],
                    }, step=global_step)

        scheduler.step()

        avg_loss = epoch_loss / len(dataloader)
        avg_mse  = [m / len(dataloader) for m in mse_per_block]
        print(f"[Stage2/{mode}] Epoch {epoch+1}/{num_epochs} — avg loss: {avg_loss:.4f}")

        # ── Validation ────────────────────────────────────────────────
        val_loss = None
        if val_dataloader is not None:
            val_loss = _compute_val_loss(
                student, teacher, val_dataloader, device, n_blocks, freeze_mlp
            )
            print(f"[Stage2/{mode}] Epoch {epoch+1}/{num_epochs} — val_loss:  {val_loss:.4f}")

        if wandb_run is not None:
            log_dict = {
                "stage2/loss_epoch": avg_loss,
                "stage2/epoch":      epoch + 1,
                "stage2/phase":      mode,
            }
            if val_loss is not None:
                log_dict["stage2/val_loss"] = val_loss
            for l, v in enumerate(avg_mse):
                log_dict[f"stage2/mse_block_{l:02d}"] = v
            wandb_run.log(log_dict, step=global_step)

    if save_path:
        _save(student, save_path)
        print(f"[Stage2] Checkpoint saved → {save_path}")

    return student


@torch.no_grad()
def _compute_val_loss(student, teacher, val_loader, device, n_blocks, freeze_mlp):
    student.eval()
    total = 0.0
    count = 0
    seq_len = student.seq_len - 1

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
        pre_ffn_states = t_out["pre_ffn_states"]
        block_outputs  = t_out["block_outputs"]

        n = min(n_blocks, len(block_inputs))
        for l in range(n):
            s_out = student.blocks[l](
                hidden_states=block_inputs[l].to(device),
                run_mlp_component=not freeze_mlp,
                return_transfer_matrix=False,
            )
            target = (pre_ffn_states[l] if freeze_mlp else block_outputs[l]).to(device)
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
