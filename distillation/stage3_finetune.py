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

REBUILD NOTES (2026-06-24) — sau khi val_acc stuck ~random-chance, val_loss
nổ lên >800 ngay khi Phase B bắt đầu:
    1. Phase B LR thực tế trước đây = lr*0.1 = 1e-5 (quá nhỏ so với paper
       Appendix A.1: Stage 3 dùng LR ổn định ~2e-4) → model gần như không học
       (train_kl/train_ce flatline suốt Phase B). Fix: dùng `lr` trực tiếp,
       caller (run_stage3.py) truyền giá trị paper-aligned.
    2. weight_decay 0.01 → 0.1, grad_clip 0.5 → 1.0 (khớp paper Appendix A.1).
    3. Early-stopping dùng `>=` khiến mọi tie tính là "cải thiện" → patience
       không tích lũy được trong lúc val_acc dao động noise quanh random
       chance → train hết 90 epoch vô ích. Fix: strict `>` + MIN_DELTA.
    4. Thêm BN recalibration (forward-only, no backward) ngay khi chuyển
       Phase A → Phase B, vì backbone vừa unfreeze làm activation distribution
       lệch khỏi distribution mà BN running stats (momentum=0.01) đã hội tụ.
    5. Thêm spike-revert: nếu val_loss > 3x best_val_loss → load lại best
       checkpoint (in-memory) + giảm nửa LR, mirror đúng cách paper Section 5.2
       mô tả xử lý Stage 3 loss spikes ("checkpointing, weight decay, and
       gradient clipping").
    6. Forward pass dùng bf16 autocast (paper dùng bf16 cho mọi stage) — giảm
       memory, cho phép tăng BATCH_SIZE (xem run_stage3.py).
"""

import os
import copy
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models.student import BiMambaSLR
from models.teacher import TeacherModel
from distillation.losses import combined_stage3_loss, kl_distillation_loss, classification_loss

# Min cải thiện để tính là "tốt hơn" — tránh early-stop bị noise của val_acc
# (oscillation random) làm vô hiệu hoá patience (xem REBUILD NOTES cuối file).
MIN_DELTA = 1e-4

# Hệ số phát hiện loss-spike: nếu val_loss > SPIKE_FACTOR * best_val_loss → revert.
# Theo paper MOHAWK Section 5.2: Stage 3 có loss spikes (instability) đã biết,
# tác giả khắc phục bằng "checkpointing, weight decay, and gradient clipping".
SPIKE_FACTOR = 3.0


def _autocast_ctx(device):
    """bf16 autocast — theo paper (Appendix A.1: 'bf16 mixed precision' mọi stage)."""
    device_type = "cuda" if "cuda" in str(device) else "cpu"
    return torch.autocast(device_type=device_type, dtype=torch.bfloat16)


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
def _recalibrate_bn(student: BiMambaSLR, dataloader: DataLoader, device: str, num_batches: int = 30):
    """
    Forward-only pass (no backward) để "làm nóng" lại BN running stats trước khi
    chuyển sang Phase B.

    Lý do cần bước này:
      - Phase A chỉ train fc → backbone weights cố định, nhưng BN running stats
        vẫn được cập nhật (student.train() bật BN train-mode cho TOÀN MODEL).
      - Khi Phase B bắt đầu, set_stage3_trainable() mở khoá toàn bộ backbone
        → forward-pass activation distribution có thể lệch khỏi distribution mà
        BN running stats (momentum=0.01, rất chậm) đã hội tụ tới trong Phase A.
      - Nếu không recalibrate, vài epoch đầu Phase B dùng eval-mode running stats
        SAI lệch → val_loss/val_acc nhiễu mạnh ngay từ đầu Phase B (đúng như quan
        sát trong wandb: val_loss bắt đầu leo thang ngay tại epoch chuyển phase).
    """
    student.train()
    n = 0
    for batch in dataloader:
        if n >= num_batches:
            break
        x, _ = _get_x_labels(batch, device)
        with _autocast_ctx(device):
            student(x)
        n += 1
    print(f"[Stage3-B] BN recalibration: {n} forward batches (no backward)")


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
    patience: int = 15,
    wandb_run=None,
    save_path: str = None,
):
    """
    2-phase Stage 3:
      Phase A (phase_a_epochs): chỉ train fc với CE → khởi động classifier
      Phase B (còn lại):        train toàn bộ với KL + CE → full distillation

    Lý do cần Phase A:
      - fc KHÔNG random — load_teacher_weights() đã copy + freeze fc từ teacher
        trước Stage 1 (xem models/student.py). Nhưng fc đó được train cho
        feature distribution của teacher (spatial attention + temporal attention);
        sau khi temporal attention bị thay bằng BiMamba2 (Stage 1+2), feature
        distribution đầu vào fc đã lệch khỏi cái fc "biết" → fc cần recalibrate.
      - BiMamba2 collapse → temporal ≈ 0
      - Nếu train toàn bộ ngay từ đầu, gradient qua 10 blocks near-identity
        không đủ để fc recalibrate kịp → val_acc < 1% (random)
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

            # BUG FIX: KHÔNG dùng torch.no_grad() ở đây.
            # - Backbone params đã có requires_grad=False → không cần grad
            # - fc params có requires_grad=True → CẦN grad để update
            # - torch.no_grad() sẽ khiến student_logits không có grad_fn
            #   → loss.backward() raise RuntimeError (không có gradient)
            # - PyTorch đủ thông minh: backward chỉ tính grad cho fc params,
            #   không đi qua backbone (backbone output có requires_grad=False)
            with _autocast_ctx(device):
                student_logits = student(x)

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

    # BN recalibration: backbone vừa được unfreeze, running stats của Phase A
    # (chỉ phản ánh frozen-backbone activations) không còn đúng nữa.
    _recalibrate_bn(student, dataloader, device, num_batches=30)
    student.train()

    WARMUP_EPOCHS = 2
    # REBUILD FIX: paper (Appendix A.1) dùng LR ổn định ~2e-4 cho Stage 3 và
    # weight_decay=0.1. Bản cũ dùng lr*0.1 (→ 1e-5 thực tế) + wd=0.01: LR quá
    # nhỏ khiến model gần như không học (train_kl/ce flatline suốt Phase B).
    # Caller (run_stage3.py) truyền `lr` = LR thực tế muốn dùng cho Phase B.
    opt_b = optim.AdamW(student.parameters(), lr=lr, weight_decay=0.1, betas=(0.9, 0.95))

    def lr_lambda(epoch):
        if epoch < WARMUP_EPOCHS:
            return (epoch + 1) / WARMUP_EPOCHS
        progress = (epoch - WARMUP_EPOCHS) / max(phase_b_epochs - WARMUP_EPOCHS, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    sched_b = optim.lr_scheduler.LambdaLR(opt_b, lr_lambda)

    trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"[Stage3-B] Trainable: {trainable:,}  alpha={alpha}  T={temperature}  lr={lr}")
    print(f"[Stage3-B] grad_accum={grad_accum}  warmup={WARMUP_EPOCHS}  patience={patience}")

    epoch_offset = phase_a_epochs   # offset cho wandb epoch axis
    epochs_no_improve = 0           # early stopping counter

    # Spike-revert state (paper Section 5.2: "addressed using checkpointing,
    # weight decay, and gradient clipping" để xử lý loss spikes ở Stage 3).
    #
    # BUG FIX (đã quan sát qua wandb): khởi tạo best_val_loss=inf / best_state=None
    # khiến epoch B đầu tiên (epoch 11 global) KHÔNG có gì để revert (guard
    # `best_state_dict is not None`) → val_loss của epoch đó, dù đã tệ hơn hẳn
    # so với Phase A (~100-150), vẫn được "chấp nhận" làm best_val_loss đầu tiên
    # của Phase B. Mọi spike-check sau đó (epoch 12, 13, ...) bị so với baseline
    # đã hỏng này → ngưỡng 3x trở nên vô nghĩa, spike-revert không bao giờ trigger
    # đúng lúc (khớp với việc val_loss tiếp tục leo lên ~3000 mà không có epoch
    # nào "biến mất" khỏi wandb — dấu hiệu của continue/revert thực sự xảy ra).
    #
    # FIX: seed best_val_loss/best_state_dict bằng trạng thái NGAY SAU BN
    # recalibration (tức là điểm bắt đầu thật của Phase B), không phải inf.
    # Nhờ vậy nếu epoch 11 đã tệ, nó sẽ bị phát hiện + revert ngay, thay vì
    # được dùng làm baseline.
    best_val_loss = float("inf")
    best_state_dict = None
    if val_dataloader is not None:
        seed_val_loss, seed_val_acc = _compute_val_metrics(
            student, teacher, val_dataloader, device, alpha, temperature
        )
        if seed_val_loss == seed_val_loss and seed_val_loss != float("inf"):
            best_val_loss = seed_val_loss
            best_state_dict = copy.deepcopy(student.state_dict())
            print(
                f"[Stage3-B] Seeded spike-revert baseline (post-BN-recalib): "
                f"val_loss={seed_val_loss:.4f}  val_acc={seed_val_acc*100:.2f}%"
            )

    for epoch in range(phase_b_epochs):
        student.train()
        epoch_loss = epoch_kl = epoch_ce = 0.0
        epoch_correct = epoch_total = 0
        nan_batches = 0

        opt_b.zero_grad()

        for step, batch in enumerate(dataloader):
            x, labels = _get_x_labels(batch, device)

            with torch.no_grad(), _autocast_ctx(device):
                teacher_logits = teacher(x)["logits"]

            with _autocast_ctx(device):
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
                # REBUILD FIX: paper Appendix A.1 dùng grad-clip=1.0 cho mọi stage,
                # không phải 0.5 — clip quá chặt + LR quá nhỏ (bug cũ) cộng hưởng
                # khiến model gần như không di chuyển trong suốt Phase B.
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
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

            # ── Spike-revert (paper Section 5.2 mitigation: "checkpointing") ──
            # Nếu val_loss nổ lên > SPIKE_FACTOR x best_val_loss → coi epoch này
            # là 1 loss spike, revert về best checkpoint trong memory + giảm nửa LR
            # thay vì để model tiếp tục train từ trạng thái đã hỏng.
            if val_loss == val_loss and val_loss != float("inf"):  # not NaN
                if best_state_dict is not None and val_loss > SPIKE_FACTOR * best_val_loss:
                    student.load_state_dict(best_state_dict)
                    for g in opt_b.param_groups:
                        g["lr"] *= 0.5
                    print(
                        f"[Stage3-B] ⚠ Loss spike phát hiện (val_loss={val_loss:.4f} > "
                        f"{SPIKE_FACTOR}x best={best_val_loss:.4f}) → revert checkpoint, "
                        f"giảm nửa LR (now {opt_b.param_groups[0]['lr']:.2e})"
                    )
                    student.train()
                    continue  # bỏ qua phần log/checkpoint của epoch bị revert

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state_dict = copy.deepcopy(student.state_dict())

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

        # REBUILD FIX: `>=` coi mọi tie là "cải thiện" → với val_acc dao động
        # ngẫu nhiên quanh random-chance, patience counter gần như không bao giờ
        # tích lũy → early stopping vô hiệu (chạy hết 90 epoch dù không học được
        # gì, đúng như log thực tế). Dùng strict `>` + MIN_DELTA.
        monitor = val_acc if val_acc is not None else train_acc
        if monitor > best_val_acc + MIN_DELTA:
            best_val_acc = monitor
            epochs_no_improve = 0
            if save_path:
                _save(student, save_path, epoch_offset + epoch, monitor)
                print(f"[Stage3-B] ✓ Best checkpoint (acc={best_val_acc*100:.2f}%) → {save_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"[Stage3-B] Early stopping: val_acc không tăng sau {patience} epoch. "
                      f"Best={best_val_acc*100:.2f}%")
                break

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
    @torch.no_grad(): tiết kiệm memory (không build backward graph trong val)

    NaN guard: skip batches có NaN student logits (BN running stats chưa kịp
    recalibrate ở epoch đầu). Loss chỉ tính trên batches hợp lệ; accuracy tính
    riêng để không bị mask bởi NaN.
    """
    student.eval()

    total_loss = 0.0
    total_correct = total_samples = 0
    valid_loss_batches = 0
    seq_len = student.seq_len - 1   # trừ CLS token

    for batch in val_loader:
        x, labels = _get_x_labels(batch, device)

        # k_copies reshape
        x_clips, k_copies = _maybe_reshape_kcopies(x, seq_len)

        with _autocast_ctx(device):
            # Teacher logits
            teacher_logits_clips = teacher(x_clips)["logits"]  # (B*k, C)

            # Student logits
            student_logits_clips = student(x_clips)             # (B*k, C)

        # NaN guard: nếu student logits có NaN/Inf → BN stats chưa hội tụ
        # Skip batch này cho loss; accuracy vẫn dùng nan_to_num để đếm đúng/sai
        has_nan = (torch.isnan(student_logits_clips).any() or
                   torch.isinf(student_logits_clips).any())

        # Average over copies → (B, C)
        B_total, num_cls = student_logits_clips.shape
        B = B_total // k_copies

        if has_nan:
            # Accuracy với logits đã clamp (NaN → 0): ít nhất biết được progress
            s_safe = torch.nan_to_num(student_logits_clips, nan=0.0, posinf=1e4, neginf=-1e4)
            s_logits_safe = s_safe.view(B, k_copies, num_cls).mean(dim=1)
            preds = s_logits_safe.argmax(dim=-1)
            tgts  = (labels.argmax(dim=-1) if (labels.dim() > 1 and labels.shape[-1] > 1)
                     else labels.long().squeeze(-1))
            total_correct += (preds == tgts).sum().item()
            total_samples += tgts.size(0)
            continue   # skip loss accumulation

        s_logits = student_logits_clips.view(B, k_copies, num_cls).mean(dim=1)
        t_logits = teacher_logits_clips.view(B, k_copies, num_cls).mean(dim=1)

        # Loss & accuracy on averaged logits
        kl   = kl_distillation_loss(s_logits, t_logits, temperature)
        ce   = classification_loss(s_logits, labels)
        loss = alpha * kl + (1 - alpha) * ce
        loss_val = loss.item()

        # Secondary NaN check: KL có thể NaN nếu logits cực lớn (overflow trong exp)
        if not (loss_val != loss_val):   # not NaN
            total_loss        += loss_val
            valid_loss_batches += 1

        preds   = s_logits.argmax(dim=-1)
        tgts    = (labels.argmax(dim=-1) if (labels.dim() > 1 and labels.shape[-1] > 1)
                   else labels.long().squeeze(-1))
        total_correct += (preds == tgts).sum().item()
        total_samples += tgts.size(0)

    student.train()
    # avg_loss = NaN nếu không có batch hợp lệ (toàn NaN) → wandb sẽ hiện NaN
    avg_loss = total_loss / max(valid_loss_batches, 1) if valid_loss_batches > 0 else float("nan")
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
