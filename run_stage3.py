"""
Chạy Stage 3: Full End-to-End Distillation (MOHAWK).

Stage 3 theo paper:
    Loss = alpha * KL(student || teacher, T) + (1 - alpha) * CE(student, hard_labels)

    - Tất cả parameters student được train
    - Teacher hoàn toàn frozen
    - Load từ checkpoints/student_stage2.pth
    - Lưu checkpoints/student_stage3.pth (best val_acc)

Kết quả quan trọng nhất: val_acc → so sánh với teacher accuracy (82.54%)

Cách dùng:
    python run_stage3.py
"""

import os, sys, warnings, logging

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import compat  # inject torchvision stub

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)
os.environ["PYTHONWARNINGS"]             = "ignore"
os.environ["TOKENIZERS_PARALLELISM"]     = "false"
os.environ["TRANSFORMERS_VERBOSITY"]     = "error"
os.environ["LIGHTNING_DISABLE_WARNINGS"] = "1"

import random
import numpy as np
import torch

# ══════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════

TEACHER_CKPT = os.path.expanduser(
    "~/sign-language-recognition/skeleton-slr-transformer-main"
    "/scripts/outputs/2026-06-04/16-23-19/checkpoints"
    "/epoch=1400-valid_loss=1.1588-valid_accuracy_PI@01=0.8254.ckpt"
)
STUDENT_STAGE2_CKPT = "checkpoints/student_stage2.pth"
SPLIT_FILE = os.path.expanduser("~/slr-mamba-distill/data/splits/splits/asl100.json")
POSE_ROOT  = os.path.expanduser("~/slr-mamba-distill/data/pose_per_individual_videos")
SSTAN_SRC  = os.path.expanduser(
    "~/sign-language-recognition/skeleton-slr-transformer-main/src"
)
OUTPUT_DIR = "checkpoints"

# ── Dataset ───────────────────────────────────────────────────────────
SEQ_LEN     = 50
N_JOINTS    = 55
IN_CHANNELS = 2
BATCH_SIZE  = 2    # Stage 3: teacher+student+full graph → memory cao, dùng GRAD_ACCUM
GRAD_ACCUM  = 4    # Effective batch = BATCH_SIZE * GRAD_ACCUM = 8
NUM_WORKERS = 4
VAL_COPIES  = 4

# ── Teacher / Student (phải khớp với Stage 1 & 2) ────────────────────
EMBEDDING_DIM = 128
N_BLOCKS      = 10
HEAD_DIM      = 64
N_HEADS       = 8
NORM_TYPE     = "batchnorm"
FFN_EXPAND    = 4.0
FFN_DROPOUT   = 0.25
MAX_STOCH     = 0.25

D_STATE    = 64
D_CONV     = 3
CHUNK_SIZE = 16

# ── Stage 3 ───────────────────────────────────────────────────────────
S3_EPOCHS      = 100      # tổng = Phase A + Phase B (teacher train 1500 epoch → cần đủ)
S3_PHASE_A     = 10       # Phase A: train fc only (CE) → khởi động classifier
S3_LR          = 1e-4    # LR Phase B; Phase A dùng lr*5
ALPHA          = 0.5     # weight KL loss; 1-ALPHA = weight CE
TEMPERATURE    = 4.0     # distillation temperature (Hinton et al.)
GRAD_ACCUM     = 4       # effective batch = BATCH_SIZE * GRAD_ACCUM = 2 * 4 = 8
PATIENCE       = 15       # early stopping Phase B: dừng nếu val_acc không tăng sau N epoch

LOG_FREQ = 10

# ── Wandb ─────────────────────────────────────────────────────────────
USE_WANDB     = True
WANDB_PROJECT = "slr-mamba-distill"
WANDB_NAME    = "stage3-wlasl100-v3"  # v3: stage2 30 epoch freeze_mlp=True + epochs=100 + early_stop

SEED   = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ══════════════════════════════════════════════════════════════════════


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    seed_everything(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if SSTAN_SRC not in sys.path:
        sys.path.insert(0, SSTAN_SRC)

    from models.teacher import TeacherModel
    from models.student import BiMambaSLR
    from distillation.stage3_finetune import train_stage3

    print(f"Device : {DEVICE}  |  Torch : {torch.__version__}")

    # ── Wandb ─────────────────────────────────────────────────────────
    wandb_run = None
    if USE_WANDB:
        import wandb
        wandb_run = wandb.init(
            project=WANDB_PROJECT,
            name=WANDB_NAME,
            config=dict(
                stage=3,
                seq_len=SEQ_LEN, n_joints=N_JOINTS,
                embedding_dim=EMBEDDING_DIM, n_blocks=N_BLOCKS,
                n_heads=N_HEADS, d_state=D_STATE, d_conv=D_CONV,
                epochs=S3_EPOCHS, phase_a=S3_PHASE_A, lr=S3_LR,
                alpha=ALPHA, temperature=TEMPERATURE,
                batch_size=BATCH_SIZE, grad_accum=GRAD_ACCUM,
                effective_batch=BATCH_SIZE * GRAD_ACCUM,
                patience=PATIENCE,
            ),
            settings=wandb.Settings(console="off"),
        )
        # Khai báo epoch là X axis cho tất cả stage3 metrics
        wandb_run.define_metric("stage3/epoch")
        wandb_run.define_metric("stage3/*", step_metric="stage3/epoch")
        print(f"Wandb : {wandb_run.url}\n")

    # ── Dataset ───────────────────────────────────────────────────────
    print("Loading dataset...")
    try:
        import json
        from functools import partial
        from torch.utils.data import DataLoader
        from sstan.dataset import Sign_Dataset
        from sstan.datamodule import collate_fn

        with open(SPLIT_FILE) as f:
            content = json.load(f)
        glosses     = sorted(set(e["gloss"] for e in content))
        num_classes = len(glosses)
        print(f"Classes : {num_classes}")

        train_dataset = Sign_Dataset(
            index_file_path=SPLIT_FILE,
            pose_root=POSE_ROOT,
            split="train",
            num_samples=SEQ_LEN,
            num_copies=1,
            sample_strategy="rnd_start",
            skeleton_augmentation=True,
        )
        val_dataset = Sign_Dataset(
            index_file_path=SPLIT_FILE,
            pose_root=POSE_ROOT,
            split="val",
            num_samples=SEQ_LEN,
            num_copies=VAL_COPIES,
            sample_strategy="k_copies",
            skeleton_augmentation=False,
        )
        _collate = partial(collate_fn, num_classes=num_classes)
        train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True,
            num_workers=NUM_WORKERS, collate_fn=_collate, drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=NUM_WORKERS, collate_fn=_collate, drop_last=False,
        )
        print(f"Train batches : {len(train_loader)}  |  Val batches : {len(val_loader)}")

    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"[ERROR] Dataset: {e}")
        sys.exit(1)

    # ── Teacher ───────────────────────────────────────────────────────
    print("\nLoading teacher...")
    if not os.path.exists(TEACHER_CKPT):
        print(f"[ERROR] Teacher checkpoint không tồn tại: {TEACHER_CKPT}")
        sys.exit(1)

    teacher = TeacherModel(
        checkpoint_path=TEACHER_CKPT,
        num_classes=num_classes,
        in_channels=IN_CHANNELS,
        seq_len=SEQ_LEN,
        n_joints=N_JOINTS,
        embedding_dim=EMBEDDING_DIM,
        n_blocks=N_BLOCKS,
        head_dim=HEAD_DIM,
        n_heads=N_HEADS,
        norm_type=NORM_TYPE,
        ffn_expand_ratio=FFN_EXPAND,
        ffn_dropout_ratio=FFN_DROPOUT,
        max_stochastic_depth_rate=MAX_STOCH,
        device=DEVICE,
    )
    teacher.to(DEVICE).eval()
    print("Teacher loaded ✓")

    # ── Teacher accuracy baseline ──────────────────────────────────────
    print("\nTeacher val accuracy (baseline): 82.54%  (từ checkpoint name)")

    # ── Student — load từ Stage 2 checkpoint ─────────────────────────
    # Fallback: dùng best checkpoint nếu final không có
    student_ckpt = STUDENT_STAGE2_CKPT
    if not os.path.exists(student_ckpt):
        alt = student_ckpt.replace(".pth", "_final.pth")
        if os.path.exists(alt):
            print(f"[INFO] student_stage2.pth not found, dùng: {alt}")
            student_ckpt = alt
        else:
            print(f"[ERROR] Stage 2 checkpoint không tồn tại: {student_ckpt}")
            print("Hãy chạy run_stage2.py trước.")
            sys.exit(1)

    print(f"\nLoading student từ Stage 2: {student_ckpt}")
    student = BiMambaSLR(
        in_channels=IN_CHANNELS,
        num_classes=num_classes,
        seq_len=SEQ_LEN,
        n_joints=N_JOINTS,
        embedding_dim=EMBEDDING_DIM,
        n_blocks=N_BLOCKS,
        head_dim=HEAD_DIM,
        n_heads=N_HEADS,
        norm_type=NORM_TYPE,
        ffn_expand_ratio=FFN_EXPAND,
        ffn_dropout_ratio=FFN_DROPOUT,
        max_stochastic_depth_rate=MAX_STOCH,
        d_state=D_STATE,
        d_conv=D_CONV,
        chunk_size=CHUNK_SIZE,
    )
    ckpt = torch.load(student_ckpt, map_location=DEVICE, weights_only=False)
    student.load_state_dict(ckpt.get("model_state_dict", ckpt))

    # Sanitize params và buffers từ Stage 2
    import torch.nn as nn
    nan_count = 0
    with torch.no_grad():
        # Params (learnable weights)
        for name, param in student.named_parameters():
            n = torch.isnan(param).sum().item() + torch.isinf(param).sum().item()
            if n > 0:
                nan_count += n
                torch.nan_to_num_(param.data, nan=0.0, posinf=1e-2, neginf=-1e-2)
        # Buffers: BatchNorm running_mean / running_var
        # LUÔN reset unconditionally — Stage 2 calibrate BN stats trên TEACHER hidden states
        # (mỗi block được feed độc lập bằng teacher output), không phải end-to-end student
        # representations. Dùng stats đó trong Stage 3 eval mode → sai distribution → NaN.
        # Reset về zero/one → BN tự recalibrate trong Phase A (train mode cập nhật running stats).
        bn_reset_count = 0
        for module in student.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                module.running_mean.zero_()
                module.running_var.fill_(1.0)
                module.momentum = 0.01   # EMA update chậm hơn → ổn định hơn
                bn_reset_count += 1

    if nan_count > 0:
        print(f"[WARN] Sanitized {nan_count} NaN/Inf values in Stage 2 checkpoint")
    else:
        print("Weights OK (no NaN/Inf in learnable params)")
    print(f"BatchNorm: reset {bn_reset_count} BN layers (running stats từ Stage 2 không valid cho end-to-end)")
    print(f"BatchNorm momentum set to 0.01 (stable EMA)")

    total = sum(p.numel() for p in student.parameters())
    print(f"Student params : {total:,}  ✓")

    # ── NaN diagnostic: kiểm tra nhanh student trước khi train ───────
    # Dùng 1 batch từ train_loader đã tạo ở trên
    print("\n[NaN check] Running quick diagnostic forward pass...")
    student.eval()
    with torch.no_grad():
        try:
            _batch = next(iter(train_loader))
            _x = _batch["skeleton_data"].to(DEVICE).float() if isinstance(_batch, dict) else _batch[0].to(DEVICE).float()
            _logits = student(_x)
            _nan = torch.isnan(_logits).any().item()
            _inf = torch.isinf(_logits).any().item()
            print(f"  logits shape : {_logits.shape}")
            print(f"  logits range : [{_logits.min().item():.3f}, {_logits.max().item():.3f}]")
            if _nan or _inf:
                print(f"  ⚠ NaN={_nan}  Inf={_inf} — checkpoint có vấn đề, kiểm tra Stage 2!")
            else:
                print(f"  ✓ No NaN/Inf — checkpoint clean, sẵn sàng Stage 3")
        except Exception as _e:
            print(f"  [WARN] Diagnostic failed: {_e}")
    student.train()

    # ── Stage 3: Full KL+CE distillation ─────────────────────────────
    print("\n" + "="*60)
    print("=== Stage 3: Full Distillation (KL + CE) ===")
    print("="*60)
    print(f"Loss = {ALPHA} * KL(T={TEMPERATURE}) + {1-ALPHA} * CE")
    print(f"Phase A: {S3_PHASE_A} epochs, fc only, CE loss  (fc chưa được train qua Stage 1+2)")
    print(f"Phase B: {S3_EPOCHS - S3_PHASE_A} epochs, full KL+CE  (early stop patience={PATIENCE})")
    print(f"Epochs : {S3_EPOCHS}  |  LR : {S3_LR}")
    print(f"Target : val_acc ≈ teacher ({82.54}%)")

    student = train_stage3(
        student=student,
        teacher=teacher,
        dataloader=train_loader,
        val_dataloader=val_loader,
        device=DEVICE,
        lr=S3_LR,
        num_epochs=S3_EPOCHS,
        phase_a_epochs=S3_PHASE_A,
        alpha=ALPHA,
        temperature=TEMPERATURE,
        grad_accum=GRAD_ACCUM,
        log_freq=LOG_FREQ,
        patience=PATIENCE,
        wandb_run=wandb_run,
        save_path=os.path.join(OUTPUT_DIR, "student_stage3.pth"),
    )
    print(f"\n✓ Stage 3 xong → {OUTPUT_DIR}/student_stage3.pth")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
