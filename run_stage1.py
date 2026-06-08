"""
Chạy Stage 1: Transfer Matrix Alignment.

Cách dùng:
    python run_stage1.py

Sửa các đường dẫn trong phần CONFIG bên dưới trước khi chạy.
"""

import os
import sys
import random
import numpy as np
import torch

# ══════════════════════════════════════════════════════════════════════
# CONFIG — chỉnh sửa các giá trị này
# ══════════════════════════════════════════════════════════════════════

# Đường dẫn teacher checkpoint (.ckpt từ Lightning)
TEACHER_CKPT = os.path.expanduser(
    "~/sign-language-recognition/skeleton-slr-transformer-main"
    "/scripts/outputs/2026-06-04/16-23-19/checkpoints"
    "/epoch=1400-valid_loss=1.1588-valid_accuracy_PI@01=0.8254.ckpt"
)

# Đường dẫn data WLASL
DATA_ROOT = os.path.expanduser("~/slr-mamba-distill/data")

# Đường dẫn sstan source (để import dataset/model teacher)
SSTAN_SRC = os.path.expanduser(
    "~/sign-language-recognition/skeleton-slr-transformer-main/src"
)

# Thư mục lưu checkpoint student
OUTPUT_DIR = "checkpoints"

# ── Dataset ───────────────────────────────────────────────────────────
SUBSET       = "asl100"
NUM_CLASSES  = 100
SEQ_LEN      = 50
N_JOINTS     = 55
IN_CHANNELS  = 2
BATCH_SIZE   = 8
NUM_WORKERS  = 4

# ── Teacher architecture (phải khớp với checkpoint) ───────────────────
EMBEDDING_DIM = 128
N_BLOCKS      = 10
HEAD_DIM      = 64
N_HEADS       = 8
NORM_TYPE     = "batchnorm"
FFN_EXPAND    = 4.0
FFN_DROPOUT   = 0.25
MAX_STOCH     = 0.25

# ── Student-only (BiMamba) ────────────────────────────────────────────
D_STATE    = 64
D_CONV     = 3
CHUNK_SIZE = 16

# ── Stage 1 training ─────────────────────────────────────────────────
S1_EPOCHS  = 10
S1_LR      = 1e-3
LOG_FREQ   = 10    # print loss mỗi N steps
VIZ_FREQ   = 2     # visualize matrices mỗi N epochs (cần --wandb)

# ── Wandb (đặt True để log) ───────────────────────────────────────────
USE_WANDB     = False
WANDB_PROJECT = "slr-mamba-distill"
WANDB_NAME    = "stage1-run1"

# ── Seed ─────────────────────────────────────────────────────────────
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

    # ── sstan vào sys.path ────────────────────────────────────────────
    if SSTAN_SRC not in sys.path:
        sys.path.insert(0, SSTAN_SRC)

    from models.teacher import TeacherModel
    from models.student import BiMambaSLR
    from distillation.stage1_matrix import train_stage1

    print(f"Device : {DEVICE}")
    print(f"Torch  : {torch.__version__}")

    # ── Wandb ─────────────────────────────────────────────────────────
    wandb_run = None
    if USE_WANDB:
        import wandb
        wandb_run = wandb.init(
            project=WANDB_PROJECT,
            name=WANDB_NAME,
            config=dict(
                stage=1, subset=SUBSET, num_classes=NUM_CLASSES,
                seq_len=SEQ_LEN, n_joints=N_JOINTS,
                embedding_dim=EMBEDDING_DIM, n_blocks=N_BLOCKS,
                n_heads=N_HEADS, d_state=D_STATE,
                s1_epochs=S1_EPOCHS, s1_lr=S1_LR,
            ),
        )

    # ── Dataset ───────────────────────────────────────────────────────
    print("\nLoading dataset...")
    try:
        from sstan.datamodule import WLASLMMPoseLightningDataModule
        dm = WLASLMMPoseLightningDataModule(
            data_dir=DATA_ROOT,
            subset=SUBSET,
            seq_len=SEQ_LEN,
            num_copies=1,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
        )
        dm.setup()
        train_loader = dm.train_dataloader()
        print(f"Train batches: {len(train_loader)}")
    except Exception as e:
        print(f"[ERROR] Không load được dataset: {e}")
        print("Kiểm tra lại DATA_ROOT và SSTAN_SRC")
        sys.exit(1)

    # ── Teacher ───────────────────────────────────────────────────────
    print(f"\nLoading teacher từ:\n  {TEACHER_CKPT}")
    if not os.path.exists(TEACHER_CKPT):
        print(f"[ERROR] Không tìm thấy checkpoint: {TEACHER_CKPT}")
        sys.exit(1)

    teacher = TeacherModel(
        checkpoint_path=TEACHER_CKPT,
        num_classes=NUM_CLASSES,
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
    teacher.to(DEVICE)
    teacher.eval()
    print("Teacher loaded ✓")

    # ── Student ───────────────────────────────────────────────────────
    print("\nKhởi tạo student...")
    student = BiMambaSLR(
        in_channels=IN_CHANNELS,
        num_classes=NUM_CLASSES,
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

    total   = sum(p.numel() for p in student.parameters())
    print(f"Student params: {total:,}")

    # ── Weight transfer Teacher → Student ─────────────────────────────
    print("\n=== Weight Transfer: Teacher → Student ===")
    student.load_teacher_weights(teacher)

    # ── Stage 1 ───────────────────────────────────────────────────────
    print("\n=== Stage 1: Transfer Matrix Alignment ===")
    student = train_stage1(
        student=student,
        teacher=teacher,
        dataloader=train_loader,
        device=DEVICE,
        lr=S1_LR,
        num_epochs=S1_EPOCHS,
        log_freq=LOG_FREQ,
        viz_freq=VIZ_FREQ,
        wandb_run=wandb_run,
        save_path=os.path.join(OUTPUT_DIR, "student_stage1.pth"),
    )

    print(f"\n✓ Stage 1 hoàn thành. Checkpoint: {OUTPUT_DIR}/student_stage1.pth")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
