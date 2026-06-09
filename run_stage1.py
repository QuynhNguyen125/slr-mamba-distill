"""
Chạy Stage 1: Transfer Matrix Alignment.
Cách dùng: python run_stage1.py
"""

import os, sys, warnings, logging

# CUDA_LAUNCH_BLOCKING phải được set TRƯỚC khi import torch
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import compat  # inject torchvision stub

# ── Tắt toàn bộ warnings ──────────────────────────────────────────────
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
SPLIT_FILE = os.path.expanduser("~/slr-mamba-distill/data/splits/splits/asl100.json")
POSE_ROOT  = os.path.expanduser("~/slr-mamba-distill/data/pose_per_individual_videos")
SSTAN_SRC  = os.path.expanduser(
    "~/sign-language-recognition/skeleton-slr-transformer-main/src"
)
OUTPUT_DIR = "checkpoints"

# ── Dataset — WLASL100 ────────────────────────────────────────────────
SEQ_LEN     = 50
N_JOINTS    = 55
IN_CHANNELS = 2
BATCH_SIZE  = 8
NUM_WORKERS = 4

# ── Teacher (khớp checkpoint) ─────────────────────────────────────────
EMBEDDING_DIM = 128
N_BLOCKS      = 10
HEAD_DIM      = 64
N_HEADS       = 8
NORM_TYPE     = "batchnorm"
FFN_EXPAND    = 4.0
FFN_DROPOUT   = 0.25
MAX_STOCH     = 0.25

# ── Student-only ──────────────────────────────────────────────────────
D_STATE    = 64
D_CONV     = 3
CHUNK_SIZE = 16

# ── Stage 1 ───────────────────────────────────────────────────────────
S1_EPOCHS  = 20
S1_LR      = 1e-3
LOG_FREQ   = 10
VIZ_FREQ   = 2
VAL_COPIES = 4   # multi-crop validation — match teacher's kcopies=4

# ── Wandb ─────────────────────────────────────────────────────────────
USE_WANDB     = True
WANDB_PROJECT = "slr-mamba-distill"
WANDB_NAME    = "stage1-wlasl100"

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

    # Add sstan to path FIRST
    if SSTAN_SRC not in sys.path:
        sys.path.insert(0, SSTAN_SRC)

    from models.teacher import TeacherModel
    from models.student import BiMambaSLR
    from distillation.stage1_matrix import train_stage1

    print(f"Device : {DEVICE}  |  Torch : {torch.__version__}")

    # ── Wandb ─────────────────────────────────────────────────────────
    wandb_run = None
    if USE_WANDB:
        import wandb
        wandb_run = wandb.init(
            project=WANDB_PROJECT,
            name=WANDB_NAME,
            config=dict(
                stage=1, seq_len=SEQ_LEN, n_joints=N_JOINTS,
                embedding_dim=EMBEDDING_DIM, n_blocks=N_BLOCKS,
                n_heads=N_HEADS, d_state=D_STATE, d_conv=D_CONV,
                s1_epochs=S1_EPOCHS, s1_lr=S1_LR, batch_size=BATCH_SIZE,
            ),
            settings=wandb.Settings(console="off"),  # tắt wandb console capture
        )
        print(f"Wandb : {wandb_run.url}\n")

    # ── Dataset ───────────────────────────────────────────────────────
    print("Loading dataset...")
    try:
        import json
        from functools import partial
        from torch.utils.data import DataLoader
        from sstan.dataset import Sign_Dataset
        from sstan.datamodule import collate_fn

        with open(SPLIT_FILE, "r") as f:
            content = json.load(f)
        glosses     = sorted(set(e["gloss"] for e in content))
        num_classes = len(glosses)
        print(f"Classes : {num_classes}  |  Split file : {SPLIT_FILE}")

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
            num_copies=VAL_COPIES,     # 4 overlapping clips, match teacher kcopies
            sample_strategy="k_copies",  # fix: "sequential" → "k_copies"
            skeleton_augmentation=False,
        )
        _collate    = partial(collate_fn, num_classes=num_classes)
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
    print(f"\nLoading teacher...")
    if not os.path.exists(TEACHER_CKPT):
        print(f"[ERROR] Checkpoint không tồn tại: {TEACHER_CKPT}")
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

    # ── Student ───────────────────────────────────────────────────────
    print("\nKhởi tạo student...")
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
    total = sum(p.numel() for p in student.parameters())
    print(f"Student params : {total:,}")

    # ── Weight Transfer ───────────────────────────────────────────────
    print("\n=== Weight Transfer: Teacher → Student ===")
    student.load_teacher_weights(teacher)

    # ── Stage 1 ───────────────────────────────────────────────────────
    print("\n=== Stage 1: Transfer Matrix Alignment ===")
    student = train_stage1(
        student=student,
        teacher=teacher,
        dataloader=train_loader,
        val_dataloader=val_loader,
        device=DEVICE,
        lr=S1_LR,
        num_epochs=S1_EPOCHS,
        log_freq=LOG_FREQ,
        viz_freq=VIZ_FREQ,
        wandb_run=wandb_run,
        save_path=os.path.join(OUTPUT_DIR, "student_stage1.pth"),
    )

    print(f"\n✓ Stage 1 xong. Checkpoint: {OUTPUT_DIR}/student_stage1.pth")
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
