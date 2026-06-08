"""
Stage 3 — Full Model Distillation.

Loss = alpha * KL(student || teacher, T) + (1-alpha) * CE(student, hard_labels)

Logs to wandb:
  stage3/train_loss_step, stage3/train_loss_epoch
  stage3/val_loss, stage3/val_top1, stage3/val_top5
  stage3/lr
"""

import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from models.student import BiMambaSLR
from models.teacher import TeacherModel
from distillation.losses import combined_stage3_loss


def train_stage3(
    student: BiMambaSLR,
    teacher: TeacherModel,
    dataloader: DataLoader,
    val_dataloader: DataLoader = None,
    device: str = "cuda",
    lr: float = 1e-4,
    num_epochs: int = 50,
    alpha: float = 0.5,
    temperature: float = 4.0,
    scheduler_patience: int = 5,
    save_path: str = "checkpoints/student_stage3.pth",
    log_freq: int = 50,
    wandb_run=None,
):
    teacher.eval()
    teacher.to(device)
    student.to(device)

    for p in student.parameters():
        p.requires_grad_(True)
    student.train()

    optimizer = optim.AdamW(student.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=scheduler_patience, verbose=False,
    )

    best_val = float("inf")
    global_step = 0

    for epoch in range(num_epochs):
        student.train()
        epoch_loss = 0.0

        for step, batch in enumerate(dataloader):
            x, labels = _get_x_labels(batch, device)

            with torch.no_grad():
                t_logits = teacher(x)["logits"]

            optimizer.zero_grad()
            s_logits = student(x)
            loss = combined_stage3_loss(s_logits, t_logits, labels, alpha, temperature)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            global_step += 1

            if (step + 1) % log_freq == 0:
                print(
                    f"[Stage3] Epoch {epoch+1}/{num_epochs}  "
                    f"Step {step+1}/{len(dataloader)}  "
                    f"Loss: {loss.item():.4f}"
                )
                if wandb_run is not None:
                    wandb_run.log({
                        "stage3/train_loss_step": loss.item(),
                        "stage3/lr": optimizer.param_groups[0]["lr"],
                    }, step=global_step)

        avg_loss = epoch_loss / len(dataloader)
        print(f"[Stage3] Epoch {epoch+1} — avg train loss: {avg_loss:.4f}")

        # ── Validation ────────────────────────────────────────────────
        val_log = {"stage3/train_loss_epoch": avg_loss, "stage3/epoch": epoch + 1}

        if val_dataloader is not None:
            val_loss, val_top1, val_top5 = _eval(student, val_dataloader, device)
            print(
                f"[Stage3] Epoch {epoch+1} — "
                f"val_loss: {val_loss:.4f}  "
                f"top-1: {val_top1*100:.2f}%  "
                f"top-5: {val_top5*100:.2f}%"
            )
            val_log.update({
                "stage3/val_loss":  val_loss,
                "stage3/val_top1":  val_top1,
                "stage3/val_top5":  val_top5,
            })
            scheduler.step(val_loss)
            if val_loss < best_val:
                best_val = val_loss
                _save(student, save_path)
                print(f"  → Best checkpoint saved ({save_path})")
        else:
            scheduler.step(avg_loss)

        if wandb_run is not None:
            wandb_run.log(val_log, step=global_step)

    return student


@torch.no_grad()
def _eval(model, loader, device, topk=(1, 5)):
    model.eval()
    total_loss = 0.0
    correct = {k: 0 for k in topk}
    total = 0
    ce = torch.nn.CrossEntropyLoss()

    for batch in loader:
        x, labels = _get_x_labels(batch, device)
        logits = model(x)
        total_loss += ce(logits, labels).item()

        for k in topk:
            _, pred_k = logits.topk(min(k, logits.size(-1)), dim=-1)
            correct[k] += (pred_k == labels.unsqueeze(-1)).any(-1).sum().item()
        total += labels.size(0)

    top1 = correct[1] / total
    top5 = correct[5] / total if 5 in correct else 0.0
    return total_loss / len(loader), top1, top5


def _save(model, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"model_state_dict": model.state_dict()}, path)


def _get_x_labels(batch, device):
    if isinstance(batch, (list, tuple)) and len(batch) >= 2:
        return batch[0].to(device).float(), batch[1].to(device).long()
    raise ValueError("Batch must be (skeleton_data, labels)")
