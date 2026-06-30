"""
Stage 3 ONLY (bỏ qua MOHAWK Stage 1 + Stage 2) + Optuna hyperparameter search.

Ý tưởng:
    Thay vì pretrain temporal_mamba qua Stage 1 (matrix orientation) và
    Stage 2 (hidden-state alignment) trước khi vào Stage 3, file này:
        1. Load teacher weights trực tiếp vào student qua
           student.load_teacher_weights(teacher) — copy spatial MHA, FFN,
           embedding, cls_token, norm1/2/3, fc; đóng băng tất cả, chỉ để
           temporal_mamba trainable (giống hệt điểm khởi đầu của Stage 1,
           NHƯNG không train Stage 1/2 — đi thẳng vào Stage 3).
        2. Train Stage 3 (distillation/stage3_finetune.py: Phase A fc-warmup
           + Phase B full KL+CE end-to-end) ngay trên temporal_mamba random-init.
    Đây là baseline để so sánh: liệu MOHAWK curriculum (Stage 1+2) có thực sự
    cần thiết, hay Stage 3 một mình (với weight-transfer + đủ epoch) đã đủ?

Optuna search space (tham khảo MOHAWK paper Appendix A.1 — paper grid-search
lr={1,2,5}x10^{-3,-4}, batch=2^{15..18}, AdamW betas=(0.9,0.95), wd=0.1,
grad_clip=1.0, WSD scheduler 10% warmup/10% decay — nhưng paper dùng effective
batch ~0.5M cho Stage 3, không thể áp trực tiếp cho effective batch=64 của ta
trên 1 GPU TITAN RTX → search lr/wd/warmup trong khoảng đã scale xuống thực tế,
KHÔNG search batch size/n_blocks/d_state/... vì các tham số kiến trúc phải
khớp 1-1 với teacher để load_teacher_weights() copy được (shape-matched)):
    - lr                    : log-uniform [1e-5, 1e-3]
    - alpha (KL vs CE)      : uniform [0.3, 0.7]
    - temperature           : categorical [2, 3, 4, 6, 8]
    - weight_decay (Phase B): log-uniform [0.02, 0.3]   (paper: 0.1)
    - grad_accum            : categorical [2, 4, 8]      (effective batch = 16 * grad_accum)
    - phase_a_epochs (trần) : int [3, 8]
    - phase_b_warmup_epochs : int [1, 3]

BATCH_SIZE giữ cố định = 16 (an toàn VRAM trên TITAN RTX cc7.5 sau fix gradient
checkpointing ở bi_mamba2.py — xem run_stage3.py) để tránh OOM giữa lúc search
tự động không có người theo dõi.

Mỗi trial chỉ train với budget epoch NHỎ (TRIAL_EPOCHS) để search nhanh, dùng
Optuna MedianPruner cắt sớm các trial rõ ràng tệ (dựa trên val_acc Phase B).
Sau khi tìm được best params, train lại với FULL budget (FINAL_EPOCHS) +
wandb logging thật, lưu checkpoint cuối cùng.

Cách dùng:
    pip install optuna
    python run_stage3_optuna.py
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

import json
import random
import numpy as np
import torch
import torch.nn as nn

try:
    import optuna
    from optuna.exceptions import TrialPruned
except ImportError:
    print("[ERROR] Thiếu optuna. Cài bằng: pip install optuna")
    sys.exit(1)

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

# ── Dataset (WLASL100 skeleton, cố định — không search) ────────────────
SEQ_LEN     = 50
N_JOINTS    = 55
IN_CHANNELS = 2
BATCH_SIZE  = 16     # cố định, an toàn VRAM (xem ghi chú run_stage3.py) — không
                      # đưa vào Optuna search vì OOM giữa lúc search tự động sẽ
                      # làm chết cả study; grad_accum đã đủ để search "effective batch".
NUM_WORKERS = 4
VAL_COPIES  = 4

# ── Kiến trúc (PHẢI khớp teacher 1-1 để load_teacher_weights() copy được —
#    không đưa vào Optuna search) ──────────────────────────────────────
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

# ── Optuna search ────────────────────────────────────────────────────────
N_TRIALS       = 30
TRIAL_EPOCHS   = 18    # budget nhỏ / trial — đủ để thấy xu hướng Phase B
TRIAL_PATIENCE = 6     # early-stop ngắn trong lúc search (tiết kiệm thời gian)
PRUNE_N_STARTUP_TRIALS = 5   # số trial đầu KHÔNG bị prune (cần dữ liệu nền để so sánh)
PRUNE_N_WARMUP_EPOCHS  = 2   # số epoch Phase B đầu của MỖI trial không bị prune

# ── Final retrain (dùng best params từ Optuna) ──────────────────────────
FINAL_EPOCHS   = 100
FINAL_PATIENCE = 15
FINAL_SAVE_PATH = os.path.join(OUTPUT_DIR, "student_stage3_only_optuna_best.pth")

LOG_FREQ = 9999   # tắt print mỗi step trong lúc search (đỡ spam console); vẫn
                   # in mỗi epoch (do train_stage3 tự in epoch summary)

# ── Wandb (CHỈ dùng cho final retrain — KHÔNG dùng cho từng Optuna trial,
#    nếu không sẽ tạo hàng chục/trăm run rác trên wandb) ─────────────────
USE_WANDB     = True
WANDB_PROJECT = "slr-mamba-distill"
WANDB_NAME    = "stage3-only-optuna-wlasl100-final"

SEED   = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ══════════════════════════════════════════════════════════════════════


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class _TrialMetricsCollector:
    """
    Drop-in thay thế cho wandb_run, truyền vào train_stage3(wandb_run=...).

    train_stage3() đã sẵn gọi wandb_run.log(dict) mỗi epoch (cả Phase A và
    Phase B) — class này "bắt" các lệnh log đó để:
      1. Theo dõi best_val_acc của trial (giá trị Optuna sẽ maximize).
      2. Báo cáo val_acc của MỖI epoch Phase B cho Optuna (trial.report) và
         raise TrialPruned() nếu Optuna quyết định trial này không còn triển
         vọng — cho phép cắt sớm trial tệ mà KHÔNG cần sửa train_stage3().
    Chỉ report ở Phase B (stage3/phase == 1) vì Phase A chỉ warmup fc, val_acc
    lúc đó chưa phản ánh chất lượng full-distillation thực sự.
    """

    def __init__(self, trial=None):
        self.trial = trial
        self.best_val_acc = 0.0
        self._phase_b_steps = 0

    def define_metric(self, *args, **kwargs):
        pass  # no-op — wandb-only API, không cần làm gì khi chạy trial

    def log(self, d: dict):
        val_acc = d.get("stage3/val_acc")
        if val_acc is not None and val_acc == val_acc:  # not NaN
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc

        if self.trial is not None and d.get("stage3/phase") == 1 and val_acc is not None:
            self._phase_b_steps += 1
            self.trial.report(val_acc, step=self._phase_b_steps)
            if self.trial.should_prune():
                raise TrialPruned()

    def finish(self):
        pass  # no-op


def _make_student():
    """Student kiến trúc cố định, KHỞI TẠO MỚI mỗi lần gọi (random init
    temporal_mamba, sau đó copy/freeze phần còn lại từ teacher)."""
    from models.student import BiMambaSLR

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
    return student


def _load_teacher_and_freeze(student, teacher):
    """
    Bỏ qua Stage 1 + Stage 2: load_teacher_weights() copy spatial MHA/FFN/
    embedding/cls_token/norm1-3/fc từ teacher rồi đóng băng tất cả, CHỈ để
    temporal_mamba (random init, _init_weights() trong student.py) trainable
    — đây chính là điểm khởi đầu mà Stage 1 lẽ ra sẽ pretrain tiếp; ở đây ta
    đi thẳng vào Stage 3 (Phase A sẽ mở fc, Phase B mở toàn bộ).
    """
    student.load_teacher_weights(teacher)

    # BatchNorm: random-init temporal_mamba sẽ làm activation distribution
    # trôi nhiều trong vài epoch đầu Phase A/B → momentum thấp (EMA chậm hơn,
    # ổn định hơn) nhất quán với cách run_stage3.py xử lý BN sau Stage 2.
    bn_count = 0
    for module in student.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.momentum = 0.01
            bn_count += 1
    return student


def objective(trial: "optuna.Trial", teacher, train_loader, val_loader):
    seed_everything(SEED)  # cố định random init/augmentation — chỉ hyperparam thay đổi giữa trial

    lr                    = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    alpha                 = trial.suggest_float("alpha", 0.3, 0.7)
    temperature           = trial.suggest_categorical("temperature", [2.0, 3.0, 4.0, 6.0, 8.0])
    weight_decay          = trial.suggest_float("weight_decay", 0.02, 0.3, log=True)
    grad_accum            = trial.suggest_categorical("grad_accum", [2, 4, 8])
    phase_a_epochs        = trial.suggest_int("phase_a_epochs", 3, 8)
    phase_b_warmup_epochs = trial.suggest_int("phase_b_warmup_epochs", 1, 3)

    from distillation.stage3_finetune import train_stage3

    student = _make_student()
    student = _load_teacher_and_freeze(student, teacher)

    collector = _TrialMetricsCollector(trial)

    try:
        train_stage3(
            student=student,
            teacher=teacher,
            dataloader=train_loader,
            val_dataloader=val_loader,
            device=DEVICE,
            lr=lr,
            num_epochs=TRIAL_EPOCHS,
            phase_a_epochs=phase_a_epochs,
            alpha=alpha,
            temperature=temperature,
            grad_accum=grad_accum,
            log_freq=LOG_FREQ,
            patience=TRIAL_PATIENCE,
            wandb_run=collector,
            save_path=None,   # không lưu checkpoint cho từng trial — tốn disk vô ích
            weight_decay=weight_decay,
            phase_b_warmup_epochs=phase_b_warmup_epochs,
        )
    except TrialPruned:
        raise
    except Exception as e:
        print(f"[Trial {trial.number}] LỖI: {e}")
        return 0.0
    finally:
        del student
        torch.cuda.empty_cache()

    return collector.best_val_acc


def main():
    seed_everything(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if SSTAN_SRC not in sys.path:
        sys.path.insert(0, SSTAN_SRC)

    global NUM_CLASSES
    from models.teacher import TeacherModel

    print(f"Device : {DEVICE}  |  Torch : {torch.__version__}")

    # ── Dataset (xây 1 lần, dùng lại cho mọi trial — BATCH_SIZE cố định) ──
    print("Loading dataset (WLASL100 skeleton)...")
    try:
        import json as _json
        from functools import partial
        from torch.utils.data import DataLoader
        from sstan.dataset import Sign_Dataset
        from sstan.datamodule import collate_fn

        with open(SPLIT_FILE) as f:
            content = _json.load(f)
        glosses     = sorted(set(e["gloss"] for e in content))
        NUM_CLASSES = len(glosses)
        print(f"Classes : {NUM_CLASSES}")

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
        _collate = partial(collate_fn, num_classes=NUM_CLASSES)
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

    # ── Teacher (xây 1 lần — frozen, dùng lại cho mọi trial) ──────────────
    print("\nLoading teacher...")
    if not os.path.exists(TEACHER_CKPT):
        print(f"[ERROR] Teacher checkpoint không tồn tại: {TEACHER_CKPT}")
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
    teacher.to(DEVICE).eval()
    print("Teacher loaded ✓  (val accuracy baseline 82.54%, từ tên checkpoint)")

    # ══════════════════════════════════════════════════════════════════
    # Optuna search
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print(f"=== Optuna search: {N_TRIALS} trials × {TRIAL_EPOCHS} epoch (Stage 3 only) ===")
    print("=" * 60)

    storage_path = f"sqlite:///{os.path.join(OUTPUT_DIR, 'optuna_stage3_only.db')}"
    study = optuna.create_study(
        study_name="stage3_only_wlasl100",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=PRUNE_N_STARTUP_TRIALS,
            n_warmup_steps=PRUNE_N_WARMUP_EPOCHS,
        ),
        storage=storage_path,
        load_if_exists=True,   # resume nếu chạy lại (vd. sau khi bị ngắt giữa search)
    )

    study.optimize(
        lambda trial: objective(trial, teacher, train_loader, val_loader),
        n_trials=N_TRIALS,
        gc_after_trial=True,   # dọn GPU memory giữa các trial — quan trọng vì train_stage3
                                # tạo optimizer/model mới mỗi trial, không dọn thì rò VRAM dần.
    )

    print("\n" + "=" * 60)
    print("=== Optuna search xong ===")
    print(f"Best val_acc : {study.best_value*100:.2f}%")
    print(f"Best params  : {study.best_params}")
    print("=" * 60)

    best_params_path = os.path.join(OUTPUT_DIR, "stage3_only_optuna_best_params.json")
    with open(best_params_path, "w") as f:
        json.dump({"best_value": study.best_value, "best_params": study.best_params}, f, indent=2)
    print(f"Đã lưu best params → {best_params_path}")

    # ══════════════════════════════════════════════════════════════════
    # Final retrain: full epoch budget + best params + wandb thật
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("=== Final retrain (Stage 3 only) với best params ===")
    print("=" * 60)

    best = study.best_params

    wandb_run = None
    if USE_WANDB:
        import wandb
        wandb_run = wandb.init(
            project=WANDB_PROJECT,
            name=WANDB_NAME,
            config=dict(
                stage="3-only", seq_len=SEQ_LEN, n_joints=N_JOINTS,
                embedding_dim=EMBEDDING_DIM, n_blocks=N_BLOCKS,
                n_heads=N_HEADS, d_state=D_STATE, d_conv=D_CONV,
                epochs=FINAL_EPOCHS, batch_size=BATCH_SIZE,
                effective_batch=BATCH_SIZE * best["grad_accum"],
                patience=FINAL_PATIENCE,
                optuna_best_value=study.best_value,
                **best,
            ),
            settings=wandb.Settings(console="off"),
        )
        wandb_run.define_metric("stage3/epoch")
        wandb_run.define_metric("stage3/*", step_metric="stage3/epoch")
        print(f"Wandb : {wandb_run.url}\n")

    from distillation.stage3_finetune import train_stage3

    student = _make_student()
    student = _load_teacher_and_freeze(student, teacher)

    # NaN diagnostic nhanh trước khi train full budget (mirror run_stage3.py)
    print("\n[NaN check] Running quick diagnostic forward pass...")
    student.to(DEVICE).eval()
    with torch.no_grad():
        try:
            _batch = next(iter(train_loader))
            _x = _batch["skeleton_data"].to(DEVICE).float() if isinstance(_batch, dict) else _batch[0].to(DEVICE).float()
            _logits = student(_x)
            _nan = torch.isnan(_logits).any().item()
            _inf = torch.isinf(_logits).any().item()
            print(f"  logits shape : {_logits.shape}")
            print(f"  logits range : [{_logits.min().item():.3f}, {_logits.max().item():.3f}]")
            print("  ⚠ NaN/Inf phát hiện!" if (_nan or _inf) else "  ✓ No NaN/Inf — sẵn sàng train")
        except Exception as _e:
            print(f"  [WARN] Diagnostic failed: {_e}")
    student.train()

    print(f"Loss = {best['alpha']:.3f} * KL(T={best['temperature']}) + {1-best['alpha']:.3f} * CE")
    print(f"lr={best['lr']:.2e}  weight_decay={best['weight_decay']:.4f}  "
          f"grad_accum={best['grad_accum']}  phase_a_epochs(trần)={best['phase_a_epochs']}  "
          f"phase_b_warmup={best['phase_b_warmup_epochs']}")
    print(f"Target : val_acc ≈ teacher (82.54%)")

    student = train_stage3(
        student=student,
        teacher=teacher,
        dataloader=train_loader,
        val_dataloader=val_loader,
        device=DEVICE,
        lr=best["lr"],
        num_epochs=FINAL_EPOCHS,
        phase_a_epochs=best["phase_a_epochs"],
        alpha=best["alpha"],
        temperature=best["temperature"],
        grad_accum=best["grad_accum"],
        log_freq=10,
        patience=FINAL_PATIENCE,
        wandb_run=wandb_run,
        save_path=FINAL_SAVE_PATH,
        weight_decay=best["weight_decay"],
        phase_b_warmup_epochs=best["phase_b_warmup_epochs"],
    )
    print(f"\n✓ Stage 3 only (Optuna best params) xong → {FINAL_SAVE_PATH}")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
