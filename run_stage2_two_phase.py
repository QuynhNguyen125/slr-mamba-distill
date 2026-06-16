"""
Chạy Stage 2: Hidden State Alignment — Two-Phase Curriculum (thực nghiệm).

Ý tưởng: curriculum learning trong Stage 2.
    Phase A (freeze_mlp=True):
        - Train chỉ spatial attention + temporal Mamba (FFN frozen)
        - Target = pre_ffn_states[l]  (align mixing matrix ≈ attention matrix)
        - Early stopping patience=10
        → Save: student_stage2_phase_a.pth

    Phase B (freeze_mlp=False):
        - Load từ Phase A checkpoint
        - Train toàn bộ block (spatial attn + temporal Mamba + FFN)
        - Target = block_outputs[l]  (full block output)
        - Early stopping patience=10
        → Save: student_stage2_two_phase_best.pth

So sánh với phi-mamba:
    - phi-mamba dùng freeze_mlp=True (default) trong 1 phase duy nhất
    - Two-phase là mở rộng curriculum — chưa được xác nhận trong paper
    - Thực nghiệm để so sánh với run_stage2_from_best.py (single-phase, freeze_mlp=False)

Cách dùng:
    CUDA_VISIBLE_DEVICES=1 python run_stage2_two_phase.py
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
# Load từ best checkpoint của Stage 1 convergence study (100 epoch)
STUDENT_STAGE1_CKPT = "checkpoints/student_stage1_best.pth"

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

# ── Teacher / Student ─────────────────────────────────────────────────
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

# ── Phase A: freeze_mlp=True (phi-mamba default) ─────────────────────
# Target = pre_ffn_states: align SSM/attention component với teacher
PHASE_A_EPOCHS    = 100    # upper bound, early stopping sẽ dừng sớm
PHASE_A_LR        = 5e-4
PHASE_A_PATIENCE  = 10
PHASE_A_MIN_DELTA = 0.005
PHASE_A_SAVE      = "student_stage2_phase_a.pth"

# ── Phase B: freeze_mlp=False (full block) ───────────────────────────
# Target = block_outputs: align toàn bộ block output với teacher
PHASE_B_EPOCHS    = 100    # upper bound, early stopping sẽ dừng sớm
PHASE_B_LR        = 2e-4   # LR nhỏ hơn: đã có warm start từ Phase A
PHASE_B_PATIENCE  = 10
PHASE_B_MIN_DELTA = 0.005
PHASE_B_SAVE      = "student_stage2_two_phase_best.pth"

LOG_FREQ = 10

# ── Wandb ─────────────────────────────────────────────────────────────
USE_WANDB     = True
WANDB_PROJECT = "slr-mamba-distill"
WANDB_NAME    = "stage2-wlasl100-v7-two-phase"  # v7: curriculum two-phase

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
                stage=2, variant="two_phase_curriculum",
                seq_len=SEQ_LEN, n_joints=N_JOINTS,
                embedding_dim=EMBEDDING_DIM, n_blocks=N_BLOCKS,
                n_heads=N_HEADS, d_state=D_STATE, d_conv=D_CONV,
                batch_size=BATCH_SIZE,
                phase_a_epochs=PHASE_A_EPOCHS, phase_a_lr=PHASE_A_LR,
                phase_a_patience=PHASE_A_PATIENCE,
                phase_b_epochs=PHASE_B_EPOCHS, phase_b_lr=PHASE_B_LR,
                phase_b_patience=PHASE_B_PATIENCE,
                stage1_ckpt="student_stage1_best.pth",
            ),
            settings=wandb.Settings(console="off"),
        )
        wandb_run.define_metric("stage2/epoch")
        wandb_run.define_metric("stage2/*", step_metric="stage2/epoch")
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

    # ══════════════════════════════════════════════════════════════════
    # PHASE A — freeze_mlp=True: align SSM/attention với pre_ffn_states
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("=== Stage 2 Phase A: freeze_mlp=True (phi-mamba default) ===")
    print("="*60)
    print(f"Target  = pre_ffn_states[l]  (align SSM/attn trước FFN)")
    print(f"Frozen  = feed_forward_network, norm_layer3, embedding, fc")
    print(f"Trained = spatial_attn, temporal_mamba, norm_layer1, norm_layer2")
    print(f"Epochs  : {PHASE_A_EPOCHS} (upper bound, early stop patience={PHASE_A_PATIENCE})")
    print(f"LR      : {PHASE_A_LR}")
    print(f"Start   : student_stage1_best.pth")

    # Load student từ Stage 1 best
    print(f"\nLoading student từ Stage 1 best: {STUDENT_STAGE1_CKPT}")
    if not os.path.exists(STUDENT_STAGE1_CKPT):
        print(f"[ERROR] Checkpoint không tồn tại: {STUDENT_STAGE1_CKPT}")
        sys.exit(1)

    student = _build_student(num_classes)
    _load_ckpt(student, STUDENT_STAGE1_CKPT, DEVICE)
    print(f"Student params : {sum(p.numel() for p in student.parameters()):,}  ✓")

    phase_a_save = os.path.join(OUTPUT_DIR, PHASE_A_SAVE)
    student = train_stage2(
        student=student,
        teacher=teacher,
        dataloader=train_loader,
        val_dataloader=val_loader,
        device=DEVICE,
        lr=PHASE_A_LR,
        num_epochs=PHASE_A_EPOCHS,
        freeze_mlp=True,          # Phase A: frozen FFN, target=pre_ffn_states
        log_freq=LOG_FREQ,
        patience=PHASE_A_PATIENCE,
        min_delta=PHASE_A_MIN_DELTA,
        wandb_run=wandb_run,
        save_path=phase_a_save,
    )
    print(f"\n✓ Phase A xong → {phase_a_save}")

    # ══════════════════════════════════════════════════════════════════
    # PHASE B — freeze_mlp=False: full block alignment từ Phase A ckpt
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("=== Stage 2 Phase B: freeze_mlp=False (full block) ===")
    print("="*60)
    print(f"Target  = block_outputs[l]  (full block output sau FFN)")
    print(f"Frozen  = embedding, fc  (chỉ frozen head và embedding)")
    print(f"Trained = tất cả block params (attn + Mamba + FFN)")
    print(f"Epochs  : {PHASE_B_EPOCHS} (upper bound, early stop patience={PHASE_B_PATIENCE})")
    print(f"LR      : {PHASE_B_LR}  (nhỏ hơn Phase A — warm start)")
    print(f"Start   : {PHASE_A_SAVE}  (Phase A best checkpoint)")

    # Load lại student từ Phase A best checkpoint
    # (train_stage2 trả về model in-memory, nhưng load từ file để đảm bảo đúng best ckpt)
    phase_a_ckpt = phase_a_save
    if not os.path.exists(phase_a_ckpt):
        # Fallback: dùng model in-memory nếu file chưa được lưu (val_loader=None)
        print(f"[WARN] {phase_a_ckpt} không tồn tại, dùng model in-memory từ Phase A")
    else:
        student = _build_student(num_classes)
        _load_ckpt(student, phase_a_ckpt, DEVICE)
        print(f"Loaded Phase A best ckpt: {phase_a_ckpt}  ✓")

    phase_b_save = os.path.join(OUTPUT_DIR, PHASE_B_SAVE)
    student = train_stage2(
        student=student,
        teacher=teacher,
        dataloader=train_loader,
        val_dataloader=val_loader,
        device=DEVICE,
        lr=PHASE_B_LR,
        num_epochs=PHASE_B_EPOCHS,
        freeze_mlp=False,         # Phase B: full block, target=block_outputs
        log_freq=LOG_FREQ,
        patience=PHASE_B_PATIENCE,
        min_delta=PHASE_B_MIN_DELTA,
        wandb_run=wandb_run,
        save_path=phase_b_save,
    )
    print(f"\n✓ Phase B xong → {phase_b_save}")
    print(f"\n✓ Stage 2 Two-Phase hoàn tất.")
    print(f"  Phase A best : {phase_a_save}")
    print(f"  Phase B best : {phase_b_save}  ← dùng cho Stage 3")

    if wandb_run is not None:
        wandb_run.finish()


# ── Helpers ───────────────────────────────────────────────────────────

def _build_student(num_classes: int):
    from models.student import BiMambaSLR
    return BiMambaSLR(
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


def _load_ckpt(model, path: str, device: str):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))


if __name__ == "__main__":
    main()
