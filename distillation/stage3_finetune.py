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
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models.student import BiMambaSLR
from models.teacher import TeacherModel
from distillation.losses import combined_stage3_loss, kl_distillation_loss, classification_loss


def _sanitize_bn_buffers(model: nn.Module):
    """
    Sau mỗi optimizer step, reset BatchNorm running stats nếu bị NaN/Inf.
    BatchNorm buffers (running_mean, running_var) là EMA của batch stats —
    nếu một batch tạo ra activation cực lớn, EMA tích lũy → NaN.
    Khi eval mode dùng running stats → crash.
    """
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            # Chỉ reset nếu NaN/Inf, không reset vô điều kiện
            if torch.isnan(module.running_mean).any() or torch.isinf(module.running_mean).any():
                module.running_mean.zero_()
            if torch.isnan(module.running_var).any() or torch.isinf(module.running_var).any():
                module.running_var.fill_(1.0)
            # Clamp: tránh tích lũy giá trị cực lớn nhưng giữ nguyên phân phối
            module.running_mean.clamp_(-1e3, 1e3)
            module.running_var.clamp_(1e-6, 1e3)


def set_stage3_trainable(student: BiMambaSLR):
    """
    Stage 3: tất cả parameters được train.
    Theo phi-mamba: student_model.requires_grad_(True)
    """
    for param in student.parameters():
        param.requires_grad_(True)


@torch.no_grad()
def _compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Top-1 accuracy. labels có thể là class index (B,) hoặc one-hot (B,C)."""
    preds = logits.argmax(dim=-1)
    tgts  = labels.argmax(dim=-1) if (labels.dim() > 1 and labels.shape[-1] > 1) else labels.long().squeeze(-1)
    return (preds == tgts).float().mean().item()


def _set_phase_a(student: BiMambaSLR):
    """
    Phase A: chỉ train fc (classification head).
    fc chưa bao giờ được train → cần khởi động trước.
    Spatial attention features → fc → classification.
    """
    for param in student.parameters():
        param.requires_grad_(False)
    for param in student.fc.parameters():
        param.requires_grad_(True)


def train_stage3(
    student: BiMambaSLR,
    teacher: TeacherModel,
    dataloader: DataLoader,
    val_dataloader: DataLoader = None,
    device: str = "cuda",
    lr: float = 1e-4,
    num_epochs: int = 30,
    phase_a_epochs: int = 5,
    alpha: float = 0.5,
    temperature: float = 4.0,
    grad_accum: int = 4,
    log_freq: int = 10,
    wandb_run=None,
    save_path: str = None,
):
    """
    2-phase Stage 3:
      Phase A (phase_a_epochs): chỉ train fc với CE → khởi động classifier
      Phase B (còn lại):        train toàn bộ với KL + CE → full distillation

    Lý do cần Phase A:
      - fc bị freeze suốt Stage 1 và 2 → random weights
      - BiMamba2 collapse → temporal ≈ 0
      - Nếu train toàn bộ ngay từ đầu, gradient qua 10 blocks near-identity
        không đủ để fc học phân loại → val_acc < 1% (random)
    """
    teacher.eval()
    teacher.to(device)
    student.to(device)

    best_val_acc = 0.0
    nan_batches_total = 0

    # ══════════════════════════════════════════════════════════════════
    # Phase A: train fc only với CE loss
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"Phase A: Train fc only ({phase_a_epochs} epochs, CE loss)")
    print(f"{'='*60}")

    _set_phase_a(student)
    student.train()

    opt_a = optim.AdamW(
        filter(lambda p: p.requires_grad, student.parameters()),
        lr=lr * 5,   # LR cao hơn vì chỉ train fc (nhỏ gọn, ít param)
        weight_decay=0.01,
    )
    sched_a = optim.lr_scheduler.CosineAnnealingLR(opt_a, T_max=phase_a_epochs)

    for epoch in range(phase_a_epochs):
        student.train()
        epoch_ce = 0.0
        epoch_correct = epoch_total = 0
        opt_a.zero_grad()

        for step, batch in enumerate(dataloader):
            x, labels = _get_x_labels(batch, device)

            with torch.no_grad():
                student_logits = student(x)   # fc không frozen nhưng grad blocked ở backbone

            # Chỉ CE loss
            ce   = classification_loss(student_logits, labels)
            loss = ce / grad_accum
            loss.backward()

            epoch_ce += ce.item()
            with torch.no_grad():
                preds = student_logits.detach().argmax(dim=-1)
                tgts  = labels.argmax(dim=-1) if (labels.dim() > 1 and labels.shape[-1] > 1) else labels.long().squeeze(-1)
                epoch_correct += (preds == tgts).sum().item()
                epoch_total   += tgts.size(0)

            if (step + 1) % grad_accum == 0 or (step + 1) == len(dataloader):
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                opt_a.step()
                opt_a.zero_grad()
                _sanitize_bn_buffers(student)
                torch.cuda.empty_cache()

        sched_a.step()
        train_acc = epoch_correct / max(epoch_total, 1)

        val_str = ""
        if val_dataloader is not None:
            val_loss_a, val_acc_a = _compute_val_metrics(
                student, teacher, val_dataloader, device, alpha, temperature
            )
            val_str = f"  val_acc: {val_acc_a*100:.2f}%"
            if val_acc_a > best_val_acc:
                best_val_acc = val_acc_a
                if save_path:
                    _save(student, save_path, epoch, val_acc_a)

        print(
            f"[PhaseA] Epoch {epoch+1}/{phase_a_epochs} — "
            f"ce: {epoch_ce/len(dataloader):.4f}  train_acc: {train_acc*100:.2f}%{val_str}"
        )
        if wandb_run is not None:
            log = {
                "stage3/epoch":      epoch + 1,
                "stage3/train_loss": epoch_ce / len(dataloader),
                "stage3/train_ce":   epoch_ce / len(dataloader),
                "stage3/train_kl":   0.0,
                "stage3/train_acc":  train_acc,
                "stage3/lr":         opt_a.param_groups[0]["lr"],
                "stage3/phase":      0,   # 0 = Phase A
            }
            if val_dataloader is not None:
                log["stage3/val_loss"] = val_loss_a
                log["stage3/val_acc"]  = val_acc_a
            wandb_run.log(log)

    print(f"\n[PhaseA] Done. Best val_acc so far: {best_val_acc*100:.2f}%")

    # ══════════════════════════════════════════════════════════════════
    # Phase B: full distillation KL + CE, all params
    # ══════════════════════════════════════════════════════════════════
    phase_b_epochs = num_epochs - phase_a_epochs
    print(f"\n{'='*60}")
    print(f"Phase B: Full distillation ({phase_b_epochs} epochs, KL + CE)")
    print(f"{'='*60}")

    set_stage3_trainable(student)
    student.train()

    import math
    WARMUP_EPOCHS = 2
    opt_b = optim.AdamW(student.parameters(), lr=lr * 0.1, weight_decay=0.01)

    def lr_lambda(epoch):
        if epoch < WARMUP_EPOCHS:
            return (epoch + 1) / WARMUP_EPOCHS
        progress = (epoch - WARMUP_EPOCHS) / max(phase_b_epochs - WARMUP_EPOCHS, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    sched_b = optim.lr_scheduler.LambdaLR(opt_b, lr_lambda)

    trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"[Stage3-B] Trainable: {trainable:,}  alpha={alpha}  T={temperature}  lr={lr}")
    print(f"[Stage3-B] grad_accum={grad_accum}  warmup={WARMUP_EPOCHS}")

    epoch_offset = phase_a_epochs   # offset cho wandb epoch axis

    for epoch in range(phase_b_epochs):
        student.train()
        epoch_loss = epoch_kl = epoch_ce = 0.0
        epoch_correct = epoch_total = 0
        nan_batches = 0

        opt_b.zero_grad()

        for step, batch in enumerate(dataloader):
            x, labels = _get_x_labels(batch, device)

            with torch.no_grad():
                teacher_logits = teacher(x)["logits"]

            student_logits = student(x)

            if torch.isnan(student_logits).any() or torch.isinf(student_logits).any():
                nan_batches += 1
                opt_b.zero_grad()
                torch.cuda.empty_cache()
                continue

            kl   = kl_distillation_loss(student_logits, teacher_logits, temperature)
            ce   = classification_loss(student_logits, labels)
            loss = (alpha * kl + (1 - alpha) * ce) / grad_accum
            loss.backward()

            unscaled = loss.item() * grad_accum
            epoch_loss += unscaled
            epoch_kl   += kl.item()
            epoch_ce   += ce.item()

            with torch.no_grad():
                preds = student_logits.detach().argmax(dim=-1)
                tgts  = labels.argmax(dim=-1) if (labels.dim() > 1 and labels.shape[-1] > 1) else labels.long().squeeze(-1)
                epoch_correct += (preds == tgts).sum().item()
                epoch_total   += tgts.size(0)

            if (step + 1) % grad_accum == 0 or (step + 1) == len(dataloader):
                torch.nn.utils.clip_grad_norm_(student.parameters(), 0.5)
                opt_b.step()
                opt_b.zero_grad()
                _sanitize_bn_buffers(student)
                torch.cuda.empty_cache()

            if (step + 1) % log_freq == 0:
                print(
                    f"[Stage3-B] Epoch {epoch+1}/{phase_b_epochs}  "
                    f"Step {step+1}/{len(dataloader)}  "
                    f"loss: {unscaled:.4f}  kl: {kl.item():.4f}  ce: {ce.item():.4f}"
                )

        sched_b.step()
        nan_batches_total += nan_batches

        n_valid   = max(len(dataloader) - nan_batches, 1)
        avg_loss  = epoch_loss / n_valid
        avg_kl    = epoch_kl   / n_valid
        avg_ce    = epoch_ce   / n_valid
        train_acc = epoch_correct / max(epoch_total, 1)

        val_loss = val_acc = None
        if val_dataloader is not None:
            val_loss, val_acc = _compute_val_metrics(
                student, teacher, val_dataloader, device, alpha, temperature
            )

        val_str = ""
        if val_loss is not None:
            val_str = f"  val_loss: {val_loss:.4f}  val_acc: {val_acc*100:.2f}%"
        nan_str = f"  nan_skip: {nan_batches}" if nan_batches > 0 else ""
        print(
            f"[Stage3-B] Epoch {epoch+1}/{phase_b_epochs} — "
            f"loss: {avg_loss:.4f}  kl: {avg_kl:.4f}  ce: {avg_ce:.4f}  "
            f"train_acc: {train_acc*100:.2f}%{val_str}{nan_str}"
        )

        if wandb_run is not None:
            log_dict = {
                "stage3/epoch":      epoch_offset + epoch + 1,
                "stage3/train_loss": avg_loss,
                "stage3/train_kl":   avg_kl,
                "stage3/train_ce":   avg_ce,
                "stage3/train_acc":  train_acc,
                "stage3/lr":         opt_b.param_groups[0]["lr"],
                "stage3/phase":      1,   # 1 = Phase B
            }
            if val_loss is not None:
                log_dict["stage3/val_loss"] = val_loss
                log_dict["stage3/val_acc"]  = val_acc
            wandb_run.log(log_dict)

        monitor = val_acc if val_acc is not None else train_acc
        if monitor >= best_val_acc:
            best_val_acc = monitor
            if save_path:
                _save(student, save_path, epoch_offset + epoch, monitor)
                print(f"[Stage3-B] ✓ Best checkpoint (acc={best_val_acc*100:.2f}%) → {save_path}")

    if save_path:
        final_path = save_path.replace(".pth", "_final.pth")
        _save(student, final_path, epoch_offset + phase_b_epochs - 1, best_val_acc)
        print(f"[Stage3] Final checkpoint → {final_path}")

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
        tgts    = labels.argmax(dim=-1) if (labels.dim() > 1 and labels.shape[-1] > 1) else labels.long().squeeze(-1)
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
