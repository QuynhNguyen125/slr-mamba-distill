"""
Stage 2 — Hidden State Alignment (MOHAWK, block-wise output alignment).

Triết lý: Cùng Input → Output Student phải giống Output Teacher.

Cho mỗi block l:
  student_input   = teacher.hidden_states[l]    ← input của teacher block l
  student_output  = student.blocks[l](student_input, run_mlp=not freeze_mlp)
  teacher_output  = teacher.block_outputs[l]    ← output của teacher block l
  loss_l          = MSE(student_output, teacher_output)

Đây là alignment chính xác nhất:
  • Cùng một input → buộc student học cách biến đổi giống teacher
  • Không phụ thuộc vào input[l+1] xấp xỉ

freeze_mlp=True  → chỉ train temporal_mamba (mixer)
freeze_mlp=False → train temporal_mamba + FFN/norm3 (toàn block trừ spatial MHA)

Logs to wandb:
  stage2/loss_step, stage2/loss_epoch, stage2/lr
  stage2/mse_block_{l}
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
    freeze_mlp=True  → chỉ temporal_mamba trainable
    freeze_mlp=False → temporal_mamba + FFN + norm3 trainable
                       (spatial MHA, embedding, PE, head vẫn đóng băng)
    """
    for name, param in student.named_parameters():
        if "temporal_mamba" in name:
            param.requires_grad_(True)
        elif not freeze_mlp and any(k in name for k in ["feed_forward_network", "norm_layer3"]):
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)


def train_stage2(
    student: BiMambaSLR,
    teacher: TeacherModel,
    dataloader: DataLoader,
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

    n_blocks  = len(student.blocks)
    mode      = "mixer-only" if freeze_mlp else "mixer+FFN"
    global_step = 0

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        mse_per_block = [0.0] * n_blocks

        for step, batch in enumerate(dataloader):
            x = _get_x(batch, device)

            # ── Teacher: thu thập input VÀ output từng block ──────────
            with torch.no_grad():
                t_out = teacher(x, return_attn=False, return_hidden_states=True)

            block_inputs  = t_out["hidden_states"]   # list[n] input  đến block l
            block_outputs = t_out["block_outputs"]   # list[n] output của block l

            # Guard: hook phải thu thập đủ dữ liệu
            if len(block_outputs) == 0:
                raise RuntimeError(
                    "[Stage2] block_outputs rỗng — post-hook của teacher không "
                    "thu thập được output. Kiểm tra kiểu trả về của teacher block."
                )
            if len(block_inputs) != len(block_outputs):
                raise RuntimeError(
                    f"[Stage2] Số block_inputs ({len(block_inputs)}) "
                    f"≠ block_outputs ({len(block_outputs)}). "
                    "Kiểm tra hook registration trong TeacherModel."
                )

            optimizer.zero_grad()
            n = min(n_blocks, len(block_inputs), len(block_outputs))
            loss = torch.tensor(0.0, device=device)

            for l in range(n):
                # Cùng input → so sánh output
                student_input   = block_inputs[l].to(device)
                teacher_output  = block_outputs[l].to(device)   # (BM, T+1, V, D)

                s_out = student.blocks[l](
                    hidden_states=student_input,
                    run_mlp_component=not freeze_mlp,
                    return_transfer_matrix=False,
                )
                student_output = s_out["hidden_states"]          # (BM, T+1, V, D)

                block_loss = hidden_state_l2_loss(student_output, teacher_output)
                loss = loss + block_loss
                mse_per_block[l] += block_loss.item()

            loss = loss / n
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            global_step += 1

            if (step + 1) % log_freq == 0:
                print(
                    f"[Stage2/{mode}] Epoch {epoch+1}/{num_epochs}  "
                    f"Step {step+1}/{len(dataloader)}  "
                    f"Loss: {loss.item():.4f}"
                )
                if wandb_run is not None:
                    wandb_run.log({
                        "stage2/loss_step": loss.item(),
                        "stage2/lr":        optimizer.param_groups[0]["lr"],
                    }, step=global_step)

        avg_loss = epoch_loss / len(dataloader)
        avg_mse  = [m / len(dataloader) for m in mse_per_block]
        print(f"[Stage2/{mode}] Epoch {epoch+1} — avg loss: {avg_loss:.4f}")

        if wandb_run is not None:
            log_dict = {
                "stage2/loss_epoch": avg_loss,
                "stage2/epoch":      epoch + 1,
            }
            for l, v in enumerate(avg_mse):
                log_dict[f"stage2/mse_block_{l:02d}"] = v
            wandb_run.log(log_dict, step=global_step)

    if save_path:
        _save(student, save_path)
        print(f"[Stage2] Checkpoint saved → {save_path}")

    return student


def _get_x(batch, device):
    if isinstance(batch, dict):
        return batch["skeleton_data"].to(device).float()
    return batch[0].to(device).float()


def _save(model, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"model_state_dict": model.state_dict()}, path)
