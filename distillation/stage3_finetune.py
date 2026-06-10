"""
Stage 3 — Full End-to-End Distillation (MOHAWK).

Theo MOHAWK paper:
    Loss = alpha * KL(student || teacher, T) + (1 - alpha) * CE(student, labels)

    - Tất cả parameters được train (student.requires_grad_(True))
    - Teacher hoàn toàn frozen (eval, no_grad)
    - KL divergence với temperature scaling → học "dark knowledge" từ teacher
    - CE trên hard labels → giữ classification signal thực sự
    - LR nhỏ hơn Stage 2 (fine-tuning toàn mô hình)

Logs to wandb (X axis = stage3/epoch, khai báo bằng define_metric):
    stage3/train_loss   — combined KL+CE loss
    stage3/train_kl     — KL component
    stage3/train_ce     — CE component
    stage3/train_acc    — top-1 accuracy trên train
    stage3/val_loss     — val combined loss
    stage3/val_acc      — top-1 accuracy trên val (k_copies multi-crop)
    stage3/lr
"""

import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from models.student import BiMambaSLR
from models.teacher import TeacherModel
from distillation.losses import combined_stage3_loss, kl_distillation_loss, classification_loss


def set_stage3_trainable(student: BiMambaSLR):
    """
    Stage 3: tất cả parameters được train.
    Theo phi-mamba: student_model.requires_grad_(True)
    """
    for param in student.parameters():
        param.requires_grad_(True)


@torch.no_grad()
def _compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Top-1 accuracy."""
    preds = logits.argmax(dim=-1)
    tgts  = labels.long().squeeze(-1)
    return (preds == tgts).float().mean().item()


def train_stage3(
    student: BiMambaSLR,
    teacher: TeacherModel,
    dataloader: DataLoader,
    val_dataloader: DataLoader = None,
    device: str = "cuda",
    lr: float = 1e-4,
    num_epochs: int = 30,
    alpha: float = 0.5,
    temperature: float = 4.0,
    grad_accum: int = 4,
    log_freq: int = 10,
    wandb_run=None,
    save_path: str = None,
):
    """
    Args:
        alpha       : weight cho KL loss  (1-alpha = weight CE)
        temperature : distillation temperature (Hinton et al. recommend T=4)
        grad_accum  : gradient accumulation steps (effective_batch = batch_size * grad_accum)
    """
    teacher.eval()
    teacher.to(device)
    student.to(device)

    set_stage3_trainable(student)
    student.train()

    optimizer = optim.AdamW(student.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"[Stage3] Trainable params: {trainable:,}  (all)")
    print(f"[Stage3] alpha={alpha}  temperature={temperature}  lr={lr}  grad_accum={grad_accum}")

    best_val_acc = 0.0

    for epoch in range(num_epochs):
        student.train()
        epoch_loss = epoch_kl = epoch_ce = 0.0
        epoch_correct = epoch_total = 0

        optimizer.zero_grad()

        for step, batch in enumerate(dataloader):
            x, labels = _get_x_labels(batch, device)

            # ── Teacher forward (frozen, no graph) ───────────────────
            with torch.no_grad():
                teacher_logits = teacher(x)["logits"]

            # ── Student forward (full, end-to-end) ────────────────────
            student_logits = student(x)

            # ── Loss (scaled by grad_accum) ───────────────────────────
            kl   = kl_distillation_loss(student_logits, teacher_logits, temperature)
            ce   = classification_loss(student_logits, labels)
            loss = (alpha * kl + (1 - alpha) * ce) / grad_accum

            loss.backward()

            # ── Metrics (use unscaled loss for logging) ───────────────
            unscaled = loss.item() * grad_accum
            epoch_loss += unscaled
            epoch_kl   += kl.item()
            epoch_ce   += ce.item()

            with torch.no_grad():
                preds = student_logits.detach().argmax(dim=-1)
                tgts  = labels.long().squeeze(-1)
                epoch_correct += (preds == tgts).sum().item()
                epoch_total   += tgts.size(0)

            # ── Optimizer step every grad_accum steps ─────────────────
            if (step + 1) % grad_accum == 0 or (step + 1) == len(dataloader):
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                torch.cuda.empty_cache()

            if (step + 1) % log_freq == 0:
                print(
                    f"[Stage3] Epoch {epoch+1}/{num_epochs}  "
                    f"Step {step+1}/{len(dataloader)}  "
                    f"loss: {unscaled:.4f}  "
                    f"kl: {kl.item():.4f}  "
                    f"ce: {ce.item():.4f}"
                )

        scheduler.step()

        n_steps        = len(dataloader)
        avg_loss       = epoch_loss / n_steps
        avg_kl         = epoch_kl   / n_steps
        avg_ce         = epoch_ce   / n_steps
        train_acc      = epoch_correct / max(epoch_total, 1)

        # ── Validation ────────────────────────────────────────────────
        val_loss = val_acc = None
        if val_dataloader is not None:
            val_loss, val_acc = _compute_val_metrics(
                student, teacher, val_dataloader, device, alpha, temperature
            )

        # ── Console log ───────────────────────────────────────────────
        val_str = ""
        if val_loss is not None:
            val_str = f"  val_loss: {val_loss:.4f}  val_acc: {val_acc*100:.2f}%"
        print(
            f"[Stage3] Epoch {epoch+1}/{num_epochs} — "
            f"loss: {avg_loss:.4f}  kl: {avg_kl:.4f}  ce: {avg_ce:.4f}  "
            f"train_acc: {train_acc*100:.2f}%{val_str}"
        )

        # ── Wandb log ─────────────────────────────────────────────────
        if wandb_run is not None:
            log_dict = {
                "stage3/epoch":      epoch + 1,
                "stage3/train_loss": avg_loss,
                "stage3/train_kl":   avg_kl,
                "stage3/train_ce":   avg_ce,
                "stage3/train_acc":  train_acc,
                "stage3/lr":         optimizer.param_groups[0]["lr"],
            }
            if val_loss is not None:
                log_dict["stage3/val_loss"] = val_loss
                log_dict["stage3/val_acc"]  = val_acc
            wandb_run.log(log_dict)

        # ── Save best checkpoint (theo val_acc > train_acc nếu có) ────
        monitor = val_acc if val_acc is not None else train_acc
        if monitor >= best_val_acc:
            best_val_acc = monitor
            if save_path:
                _save(student, save_path, epoch, monitor)
                print(f"[Stage3] ✓ Best checkpoint saved (acc={best_val_acc*100:.2f}%) → {save_path}")

    # Save final checkpoint
    if save_path:
        final_path = save_path.replace(".pth", "_final.pth")
        _save(student, final_path, num_epochs - 1, best_val_acc)
        print(f"[Stage3] Final checkpoint saved → {final_path}")

    return student


@torch.no_grad()
def _compute_val_metrics(student, teacher, val_loader, device, alpha, temperature):
    """
    Val loss và accuracy với k_copies multi-crop:
        - Mỗi sample được crop k lần → tensor (B, C, T*k, V, M)
        - Reshape → (B*k, C, T, V, M) → forward từng clip → average logits → vote
    """
    student.eval()

    total_loss = total_correct = total_samples = 0
    seq_len = student.seq_len - 1   # trừ CLS token

    for batch in val_loader:
        x, labels = _get_x_labels(batch, device)

        # k_copies reshape
        x_clips, k_copies = _maybe_reshape_kcopies(x, seq_len)

        # Teacher logits
        teacher_logits_clips = teacher(x_clips)["logits"]  # (B*k, C)

        # Student logits
        student_logits_clips = student(x_clips)             # (B*k, C)

        # Average over copies → (B, C)
        B_total, num_cls = student_logits_clips.shape
        B = B_total // k_copies
        s_logits = student_logits_clips.view(B, k_copies, num_cls).mean(dim=1)
        t_logits = teacher_logits_clips.view(B, k_copies, num_cls).mean(dim=1)

        # Loss & accuracy on averaged logits
        kl   = kl_distillation_loss(s_logits, t_logits, temperature)
        ce   = classification_loss(s_logits, labels)
        loss = alpha * kl + (1 - alpha) * ce

        preds   = s_logits.argmax(dim=-1)
        tgts    = labels.long().squeeze(-1)
        correct = (preds == tgts).sum().item()

        total_loss    += loss.item()
        total_correct += correct
        total_samples += tgts.size(0)

    student.train()
    avg_loss = total_loss / max(len(val_loader), 1)
    accuracy = total_correct / max(total_samples, 1)
    return avg_loss, accuracy


def _maybe_reshape_kcopies(x: torch.Tensor, seq_len: int):
    """
    Nếu T_total = T * k (k_copies), reshape về (B*k, C, T, V, M).
    Trả về (x_reshaped, k_copies).
    """
    if x.ndim == 5:
        B, C, T_total, V, M = x.shape
        if T_total > seq_len and T_total % seq_len == 0:
            k = T_total // seq_len
            x = (x.view(B, C, k, seq_len, V, M)
                  .permute(0, 2, 1, 3, 4, 5)
                  .contiguous()
                  .view(B * k, C, seq_len, V, M))
            return x, k
    return x, 1


def _get_x_labels(batch, device):
    if isinstance(batch, dict):
        x      = batch["skeleton_data"].to(device).float()
        labels = batch["label"].to(device).float()
    else:
        x, labels = batch[0].to(device).float(), batch[1].to(device).float()
    return x, labels


def _save(model, path, epoch, metric):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "epoch":            epoch,
        "best_metric":      metric,
    }, path)
