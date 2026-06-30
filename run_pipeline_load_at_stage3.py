"""
Pipeline đủ 3 stage (MOHAWK) — KHÔNG load teacher weights ở Stage 1, mà load
ngay TRƯỚC khi vào Stage 3.

So với pipeline chuẩn (run_stage1.py → run_stage2.py → run_stage3.py), nơi
student.load_teacher_weights(teacher) được gọi NGAY TỪ ĐẦU (trước Stage 1) để
spatial MHA/FFN/embedding/fc có giá trị teacher làm "neo" cho temporal_mamba
học theo, pipeline này đảo lại:

    Stage 1 (Transfer Matrix Alignment):
        student khởi tạo HOÀN TOÀN RANDOM (không load_teacher_weights()).
        train_stage1() gọi freeze_non_temporal_mamba() nội bộ → chỉ
        temporal_mamba trainable; spatial MHA/FFN/embedding/fc vẫn RANDOM
        nhưng bị đóng băng (không train) trong lúc temporal_mamba học theo
        transfer-matrix loss. Input mỗi block vẫn là teacher_hidden_states[l]
        (lấy từ forward thật của teacher) — nhưng đi qua spatial-attn RANDOM
        của student trước khi tới temporal_mamba, nên "không gian" mà
        temporal_mamba thấy nhiễu hơn nhiều so với pipeline chuẩn.

    Stage 2 (Hidden State Alignment):
        Tiếp tục trên student của Stage 1 (vẫn random ở phần spatial/FFN).
        FREEZE_MLP=True (phi-mamba default) — train multihead_self_attention1
        + norm1 + norm2 + temporal_mamba theo target pre_ffn_states của
        teacher; FFN vẫn đóng băng (random) vì dù gì cũng sẽ bị NẠP LẠI bằng
        teacher's FFN ngay sau Stage 2 — train nó ở đây chỉ tốn compute vô ích.
        Mục tiêu thực sự của Stage 1+2 trong pipeline này: dùng gradient
        signal để định hình temporal_mamba (+ một phần spatial attn làm
        "đường dẫn" gradient) mà KHÔNG cần neo vào đặc trưng teacher.

    >>> Load teacher weights NGAY TRƯỚC Stage 3 <<<
        student.load_teacher_weights(teacher) được gọi ở đây — ghi đè
        embedding/cls_token/spatial-MHA/FFN/norm1-3/fc bằng giá trị teacher
        (huỷ phần spatial-attn vừa train ở Stage 2, vì dù sao cũng chỉ là
        "đường dẫn gradient" tạm), NHƯNG GIỮ NGUYÊN temporal_mamba — phần đã
        được Stage 1+2 huấn luyện. Đây chính là điểm khác biệt cốt lõi: kết
        hợp "temporal_mamba học từ scratch qua Stage 1+2" với "spatial
        features chất lượng cao từ teacher", thay vì có cả hai từ đầu.

    Stage 3 (Full Distillation):
        Như run_stage3.py: Phase A (fc warmup) → Phase B (KL+CE full).

Mục đích ablation: so sánh với pipeline chuẩn (load teacher weights từ đầu) —
liệu temporal_mamba có học SSM dynamics "tự nhiên" hơn (không bị ràng buộc
vào đặc trưng cụ thể của teacher ngay từ Stage 1) khi được ghép lại với
spatial features thật của teacher ở Stage 3?

Cách dùng:
    python run_pipeline_load_at_stage3.py
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
SPLIT_FILE = os.path.expanduser("~/slr-mamba-distill/data/splits/splits/asl100.json")  # WLASL100
POSE_ROOT  = os.path.expanduser("~/slr-mamba-distill/data/pose_per_individual_videos")
SSTAN_SRC  = os.path.expanduser(
    "~/sign-language-recognition/skeleton-slr-transformer-main/src"
)
OUTPUT_DIR = "checkpoints"

# ── Dataset (WLASL100 skeleton) ─────────────────────────────────────────
SEQ_LEN     = 50
N_JOINTS    = 55
IN_CHANNELS = 2
BATCH_SIZE1 = 8    # Stage 1 — giống run_stage1.py
BATCH_SIZE2 = 4    # Stage 2 — backward per-block, tốn VRAM hơn (giống run_stage2.py)
BATCH_SIZE3 = 16   # Stage 3 — giống run_stage3.py
NUM_WORKERS = 4
VAL_COPIES  = 4

# ── Kiến trúc (phải khớp teacher để load_teacher_weights() copy được ở
#    Stage 3) ─────────────────────────────────────────────────────────
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

# ── Stage 1: Transfer Matrix Alignment — Full Convergence Study ───────
S1_EPOCHS         = 100              # upper bound (no early stopping, giống run_stage1.py)
S1_LR             = 1e-3
S1_LOG_FREQ       = 10
S1_CHECKPOINT_EPOCHS = [10, 25, 50, 100]
S1_VIZ_EPOCHS        = [1, 25, 100]

# ── Stage 2: Hidden State Alignment ───────────────────────────────────
# freeze_mlp=True (phi-mamba default): FFN giữ random + đóng băng, vì dù sao
# cũng sẽ bị nạp lại bằng teacher's FFN ngay trước Stage 3 — train nó ở đây
# chỉ tốn compute vô ích. Chỉ train multihead_self_attention1/norm1/norm2 +
# temporal_mamba (đường dẫn gradient cho temporal_mamba học).
S2_FREEZE_MLP = True
S2_EPOCHS     = 100   # upper bound — early stopping dừng sớm khi converge
S2_LR         = 5e-4
S2_PATIENCE   = 10
S2_MIN_DELTA  = 0.005

# ── Stage 3: Full Distillation ────────────────────────────────────────
S3_EPOCHS    = 100
S3_PHASE_A   = 20
S3_LR        = 1e-4
S3_ALPHA     = 0.5
S3_TEMPERATURE = 4.0
S3_GRAD_ACCUM  = 4
S3_PATIENCE    = 15
S3_LOG_FREQ    = 10

# ── Wandb (1 run duy nhất, namespace stage1/*, stage2/*, stage3/* tách biệt) ──
USE_WANDB     = True
WANDB_PROJECT = "slr-mamba-distill"
WANDB_NAME    = "pipeline-loadat3-wlasl100"

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
    from distillation.stage1_matrix import train_stage1
    from distillation.stage2_hidden import train_stage2
    from distillation.stage3_finetune import train_stage3

    print(f"Device : {DEVICE}  |  Torch : {torch.__version__}")

    # ── Wandb — 1 run duy nhất cho cả 3 stage ───────────────────────────
    wandb_run = None
    if USE_WANDB:
        import wandb
        wandb_run = wandb.init(
            project=WANDB_PROJECT,
            name=WANDB_NAME,
            config=dict(
                pipeline="load_teacher_weights_at_stage3",
                seq_len=SEQ_LEN, n_joints=N_JOINTS,
                embedding_dim=EMBEDDING_DIM, n_blocks=N_BLOCKS,
                n_heads=N_HEADS, d_state=D_STATE, d_conv=D_CONV,
                s1_epochs=S1_EPOCHS, s1_lr=S1_LR, batch_size1=BATCH_SIZE1,
                s2_epochs=S2_EPOCHS, s2_lr=S2_LR, s2_freeze_mlp=S2_FREEZE_MLP,
                batch_size2=BATCH_SIZE2,
                s3_epochs=S3_EPOCHS, s3_lr=S3_LR, s3_alpha=S3_ALPHA,
                s3_temperature=S3_TEMPERATURE, batch_size3=BATCH_SIZE3,
            ),
            settings=wandb.Settings(console="off"),
        )
        wandb_run.define_metric("stage1/epoch")
        wandb_run.define_metric("stage1/*", step_metric="stage1/epoch")
        wandb_run.define_metric("stage2/epoch")
        wandb_run.define_metric("stage2/*", step_metric="stage2/epoch")
        wandb_run.define_metric("stage3/epoch")
        wandb_run.define_metric("stage3/*", step_metric="stage3/epoch")
        print(f"Wandb : {wandb_run.url}\n")

    # ── Dataset — 3 cặp DataLoader (batch khác nhau theo stage) ─────────
    print("Loading dataset (WLASL100 skeleton)...")
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

        def _loaders(bs):
            tl = DataLoader(train_dataset, batch_size=bs, shuffle=True,
                             num_workers=NUM_WORKERS, collate_fn=_collate, drop_last=True)
            vl = DataLoader(val_dataset, batch_size=bs, shuffle=False,
                             num_workers=NUM_WORKERS, collate_fn=_collate, drop_last=False)
            return tl, vl

        train_loader1, val_loader1 = _loaders(BATCH_SIZE1)
        train_loader2, val_loader2 = _loaders(BATCH_SIZE2)
        train_loader3, val_loader3 = _loaders(BATCH_SIZE3)
        print(f"Stage1 batches: train={len(train_loader1)} val={len(val_loader1)} (batch={BATCH_SIZE1})")
        print(f"Stage2 batches: train={len(train_loader2)} val={len(val_loader2)} (batch={BATCH_SIZE2})")
        print(f"Stage3 batches: train={len(train_loader3)} val={len(val_loader3)} (batch={BATCH_SIZE3})")

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
    print("Teacher loaded ✓  (val accuracy baseline 82.54%, từ tên checkpoint)")

    # ── Student — khởi tạo RANDOM, KHÔNG load_teacher_weights() ─────────
    print("\nKhởi tạo student (random init — KHÔNG load teacher weights)...")
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
    print(f"Student params : {total:,}  (spatial MHA/FFN/embedding/fc RANDOM, sẽ load teacher trước Stage 3)")

    # ══════════════════════════════════════════════════════════════════
    # Stage 1 — Transfer Matrix Alignment (student spatial/FFN vẫn random)
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("=== Stage 1: Transfer Matrix Alignment (KHÔNG load teacher weights) ===")
    print("=" * 60)
    print(f"Epochs : {S1_EPOCHS} (no early stopping)  |  LR : {S1_LR}")
    print("Lưu ý  : chỉ temporal_mamba trainable; spatial/FFN/embedding/fc RANDOM + frozen.")

    student = train_stage1(
        student=student,
        teacher=teacher,
        dataloader=train_loader1,
        val_dataloader=val_loader1,
        device=DEVICE,
        lr=S1_LR,
        num_epochs=S1_EPOCHS,
        log_freq=S1_LOG_FREQ,
        checkpoint_epochs=S1_CHECKPOINT_EPOCHS,
        viz_epochs=S1_VIZ_EPOCHS,
        wandb_run=wandb_run,
        save_path=os.path.join(OUTPUT_DIR, "student_pipeline_loadat3_stage1_best.pth"),
        save_dir=OUTPUT_DIR,
    )
    print(f"✓ Stage 1 xong → {OUTPUT_DIR}/student_pipeline_loadat3_stage1_best.pth")

    # ══════════════════════════════════════════════════════════════════
    # Stage 2 — Hidden State Alignment (vẫn trên student random spatial/FFN)
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print(f"=== Stage 2: Hidden State Alignment (freeze_mlp={S2_FREEZE_MLP}) ===")
    print("=" * 60)
    print(f"Epochs : {S2_EPOCHS} (trần, patience={S2_PATIENCE})  |  LR : {S2_LR}")
    print("Lưu ý  : FFN giữ random + frozen (sẽ bị nạp lại bằng teacher's FFN ngay sau Stage 2).")

    student = train_stage2(
        student=student,
        teacher=teacher,
        dataloader=train_loader2,
        val_dataloader=val_loader2,
        device=DEVICE,
        lr=S2_LR,
        num_epochs=S2_EPOCHS,
        freeze_mlp=S2_FREEZE_MLP,
        log_freq=S1_LOG_FREQ,
        patience=S2_PATIENCE,
        min_delta=S2_MIN_DELTA,
        wandb_run=wandb_run,
        save_path=os.path.join(OUTPUT_DIR, "student_pipeline_loadat3_stage2_best.pth"),
    )
    print(f"✓ Stage 2 xong → {OUTPUT_DIR}/student_pipeline_loadat3_stage2_best.pth")

    # ══════════════════════════════════════════════════════════════════
    # >>> LOAD TEACHER WEIGHTS NGAY TRƯỚC STAGE 3 <<<
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("=== Weight Transfer: Teacher → Student (ngay trước Stage 3) ===")
    print("=" * 60)
    print(
        "Ghi đè embedding/cls_token/spatial-MHA/FFN/norm1-3/fc bằng teacher.\n"
        "GIỮ NGUYÊN temporal_mamba (đã train qua Stage 1+2) — đây là điểm khác\n"
        "biệt cốt lõi của pipeline này so với load-từ-đầu."
    )
    student.load_teacher_weights(teacher)

    import torch.nn as nn
    for module in student.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.momentum = 0.01  # ổn định BN sau khi backbone vừa bị ghi đè

    # NaN diagnostic nhanh sau weight transfer (mirror run_stage3.py)
    print("\n[NaN check] Running quick diagnostic forward pass...")
    student.to(DEVICE).eval()
    with torch.no_grad():
        try:
            batch = next(iter(train_loader3))
            x = batch["skeleton_data"].to(DEVICE).float() if isinstance(batch, dict) else batch[0].to(DEVICE).float()
            logits = student(x)
            print(f"  logits shape : {logits.shape}")
            print(f"  logits range : [{logits.min().item():.3f}, {logits.max().item():.3f}]")
            has_bad = torch.isnan(logits).any().item() or torch.isinf(logits).any().item()
            print("  ⚠ NaN/Inf phát hiện!" if has_bad else "  ✓ No NaN/Inf — sẵn sàng train")
        except Exception as e:
            print(f"  [WARN] Diagnostic failed: {e}")
    student.train()

    # ══════════════════════════════════════════════════════════════════
    # Stage 3 — Full Distillation (Phase A fc-warmup + Phase B KL+CE)
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("=== Stage 3: Full Distillation ===")
    print("=" * 60)
    print(f"Epochs : {S3_EPOCHS}  |  Phase A : {S3_PHASE_A}  |  LR : {S3_LR}")
    print(f"Loss = {S3_ALPHA} * KL(T={S3_TEMPERATURE}) + {1-S3_ALPHA} * CE")
    print(f"Target : val_acc ≈ teacher (82.54%)")

    student = train_stage3(
        student=student,
        teacher=teacher,
        dataloader=train_loader3,
        val_dataloader=val_loader3,
        device=DEVICE,
        lr=S3_LR,
        num_epochs=S3_EPOCHS,
        phase_a_epochs=S3_PHASE_A,
        alpha=S3_ALPHA,
        temperature=S3_TEMPERATURE,
        grad_accum=S3_GRAD_ACCUM,
        log_freq=S3_LOG_FREQ,
        patience=S3_PATIENCE,
        wandb_run=wandb_run,
        save_path=os.path.join(OUTPUT_DIR, "student_pipeline_loadat3_best.pth"),
    )
    print(f"\n✓ Pipeline (load weights tại Stage 3) xong → {OUTPUT_DIR}/student_pipeline_loadat3_best.pth")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
