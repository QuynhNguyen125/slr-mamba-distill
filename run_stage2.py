"""
Chạy Stage 2: Hidden State Alignment (MOHAWK).

Theo paper MOHAWK và phi-mamba repo:
    - 1 phase duy nhất, freeze_mlp=False (full block alignment)
    - Target = full block output của teacher (phi-mamba: all_hidden_states[l+1])
    - Loss = ||h_student[l] - h_teacher[l]||_2  per-token L2 norm
    - Backward per-block để tiết kiệm memory

Tham khảo: phi-mamba/assets/mohawk_stage2.py
    freeze_mlp = True/False  # "up to training scheme"

Cách dùng:
    python run_stage2.py
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
STUDENT_STAGE1_CKPT = "checkpoints/student_stage1.pth"
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
BATCH_SIZE  = 4
NUM_WORKERS = 4
VAL_COPIES  = 4

# ── Teacher / Student (phải khớp với Stage 1) ────────────────────────
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

# ── Stage 2: full block alignment (freeze_mlp=False) theo paper ──────
S2_EPOCHS = 20    # train đến khi val loss hội tụ
S2_LR     = 5e-4

LOG_FREQ = 10

# ── Wandb ─────────────────────────────────────────────────────────────
USE_WANDB     = True
WANDB_PROJECT = "slr-mamba-distill"
WANDB_NAME    = "stage2-wlasl100"

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
    from distillation.stage2_hidden import train_stage2

    print(f"Device : {DEVICE}  |  Torch : {torch.__version__}")

    # ── Wandb ─────────────────────────────────────────────────────────
    wandb_run = None
    if USE_WANDB:
        import wandb
        wandb_run = wandb.init(
            project=WANDB_PROJECT,
            name=WANDB_NAME,
            config=dict(
                stage=2,
                seq_len=SEQ_LEN, n_joints=N_JOINTS,
                embedding_dim=EMBEDDING_DIM, n_blocks=N_BLOCKS,
                n_heads=N_HEADS, d_state=D_STATE, d_conv=D_CONV,
                epochs=S2_EPOCHS, lr=S2_LR,
                batch_size=BATCH_SIZE,
                freeze_mlp=False,
            ),
            settings=wandb.Settings(console="off"),
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

    # ── Student — load từ Stage 1 checkpoint ──────────────────────────
    print(f"\nLoading student từ Stage 1: {STUDENT_STAGE1_CKPT}")
    if not os.path.exists(STUDENT_STAGE1_CKPT):
        print(f"[ERROR] Stage 1 checkpoint không tồn tại: {STUDENT_STAGE1_CKPT}")
        print("Hãy chạy run_stage1.py trước.")
        sys.exit(1)

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
    ckpt = torch.load(STUDENT_STAGE1_CKPT, map_location=DEVICE, weights_only=False)
    student.load_state_dict(ckpt.get("model_state_dict", ckpt))
    total = sum(p.numel() for p in student.parameters())
    print(f"Student params : {total:,}  ✓")

    # ── Stage 2: Full block hidden state alignment (theo paper) ───────
    print("\n" + "="*60)
    print("=== Stage 2: Hidden State Alignment (freeze_mlp=False) ===")
    print("="*60)
    print("Target = teacher full block output  (phi-mamba: all_hidden_states[l+1])")
    print(f"Epochs : {S2_EPOCHS}  |  LR : {S2_LR}")

    student = train_stage2(
        student=student,
        teacher=teacher,
        dataloader=train_loader,
        val_dataloader=val_loader,
        device=DEVICE,
        lr=S2_LR,
        num_epochs=S2_EPOCHS,
        freeze_mlp=False,
        log_freq=LOG_FREQ,
        wandb_run=wandb_run,
        save_path=os.path.join(OUTPUT_DIR, "student_stage2.pth"),
    )
    print(f"\n✓ Stage 2 xong → {OUTPUT_DIR}/student_stage2.pth")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
