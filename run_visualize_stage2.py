"""
Visualize transfer matrices sau Stage 2 distillation.

So sánh với Stage 1:
    Stage 1 → visualizations/          (student_stage1.pth)
    Stage 2 → visualizations_stage2/   (student_stage2.pth)

Chạy:
    python run_visualize_stage2.py
"""

import os
import sys
import subprocess

# ── Config (phải khớp với run_stage2.py) ──────────────────────────────
STUDENT_CKPT  = "checkpoints/student_stage2.pth"
TEACHER_CKPT  = os.path.expanduser(
    "~/sign-language-recognition/skeleton-slr-transformer-main"
    "/scripts/outputs/2026-06-04/16-23-19/checkpoints"
    "/epoch=1400-valid_loss=1.1588-valid_accuracy_PI@01=0.8254.ckpt"
)
SPLIT_FILE    = os.path.expanduser("~/slr-mamba-distill/data/splits/splits/asl100.json")
POSE_ROOT     = os.path.expanduser("~/slr-mamba-distill/data/pose_per_individual_videos")
OUT_DIR       = "visualizations_stage2"

SEQ_LEN       = 50
N_JOINTS      = 55
EMBEDDING_DIM = 128
N_BLOCKS      = 10
HEAD_DIM      = 64
N_HEADS       = 8
NORM_TYPE     = "batchnorm"
D_STATE       = 64
D_CONV        = 3
CHUNK_SIZE    = 16
BATCH_SIZE    = 4
DEVICE        = "cuda"
BLOCKS        = [0, 3, 6, 9]

USE_WANDB     = True
WANDB_PROJECT = "slr-mamba-distill"

# ── Fallback: dùng best checkpoint nếu final không tồn tại ──────────────
if not os.path.exists(STUDENT_CKPT):
    alt = STUDENT_CKPT.replace(".pth", "_final.pth")
    if os.path.exists(alt):
        print(f"[INFO] student_stage2.pth not found, dùng: {alt}")
        STUDENT_CKPT = alt
    else:
        print(f"[ERROR] Không tìm thấy Stage 2 checkpoint: {STUDENT_CKPT}")
        print("Hãy chạy run_stage2.py trước.")
        sys.exit(1)

# ── Chạy visualize_matrices.py ─────────────────────────────────────────
cmd = [
    sys.executable, "visualize_matrices.py",
    "--student_ckpt",  STUDENT_CKPT,
    "--teacher_ckpt",  TEACHER_CKPT,
    "--split_file",    SPLIT_FILE,
    "--pose_root",     POSE_ROOT,
    "--out_dir",       OUT_DIR,
    "--seq_len",       str(SEQ_LEN),
    "--n_joints",      str(N_JOINTS),
    "--embedding_dim", str(EMBEDDING_DIM),
    "--n_blocks",      str(N_BLOCKS),
    "--head_dim",      str(HEAD_DIM),
    "--n_heads",       str(N_HEADS),
    "--norm_type",     NORM_TYPE,
    "--d_state",       str(D_STATE),
    "--d_conv",        str(D_CONV),
    "--chunk_size",    str(CHUNK_SIZE),
    "--batch_size",    str(BATCH_SIZE),
    "--device",        DEVICE,
    "--blocks",        *[str(b) for b in BLOCKS],
]
if USE_WANDB:
    cmd += ["--wandb", "--wandb_project", WANDB_PROJECT]

print(f"Student ckpt : {STUDENT_CKPT}")
print(f"Output dir   : {OUT_DIR}/")
print()

os.makedirs(OUT_DIR, exist_ok=True)
result = subprocess.run(cmd)
sys.exit(result.returncode)
