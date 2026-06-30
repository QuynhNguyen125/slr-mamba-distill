"""
Stage 2 + Stage 3 (bỏ qua MOHAWK Stage 1) + Optuna hyperparameter search.

So với run_stage3_optuna.py (chỉ Stage 3, bỏ cả Stage 1+2), file này thêm lại
Stage 2 (hidden-state alignment) vào pipeline — vẫn bỏ Stage 1 (matrix
orientation), vì:
    1. load_teacher_weights() đã copy spatial MHA/FFN/embedding/fc từ teacher,
       temporal_mamba random-init (_init_weights() trong student.py).
    2. Stage 2: align temporal_mamba + multihead_self_attention1 theo
       pre_ffn_states hoặc block_outputs của teacher (distillation/stage2_hidden.py)
       — đây là bước MOHAWK paper cho rằng quan trọng nhất để temporal mixer
       "học" hành vi gần với teacher TRƯỚC khi vào full end-to-end distillation.
    3. Stage 3: full KL+CE distillation (Phase A fc-warmup + Phase B full),
       y như run_stage3_optuna.py.

Mục đích ablation: so sánh "Stage 3 only" (run_stage3_optuna.py) vs "Stage 2 +
Stage 3" (file này) — liệu việc thêm Stage 2 có cải thiện val_acc cuối cùng so
với việc bỏ qua nó hoàn toàn, trên cùng điều kiện WLASL100 skeleton + TITAN RTX?

Optuna search space (tham khảo MOHAWK paper Appendix A.1 — paper search riêng
từng stage: Stage2 bs=2^15/lr=2e-3, Stage3 bs=2^19/lr=5e-4 — ở đây scale xuống
cho effective batch nhỏ hơn nhiều, và search ĐỒNG THỜI cả Stage2 + Stage3 trong
1 trial, đánh giá bằng val_acc CUỐI CÙNG sau Stage 3):
    Stage 2:
        - stage2_lr          : log-uniform [1e-5, 1e-3]   (paper: 2e-3)
        - stage2_freeze_mlp   : categorical [True, False]  (phi-mamba default=True;
                                 nhưng repo hiện dùng False — search lại vì bắt đầu
                                 từ teacher-weight-transfer thay vì Stage 1 checkpoint
                                 có thể thay đổi lựa chọn tối ưu)
        - stage2_epochs (trần): int [4, 10]   (budget nhỏ cho trial)
    Stage 3 (giống run_stage3_optuna.py):
        - lr, alpha, temperature, weight_decay, grad_accum,
          phase_a_epochs, phase_b_warmup_epochs

Kiến trúc (n_blocks, d_state, ...) và batch size mỗi stage giữ CỐ ĐỊNH (không
search) — lý do giống run_stage3_optuna.py: phải khớp shape với teacher, và
tránh OOM giữa lúc search tự động không người theo dõi. Stage 2 dùng
BATCH_SIZE2=4 (nhỏ hơn Stage 3) vì backward per-block (xem stage2_hidden.py)
tốn nhiều VRAM hơn — đúng theo run_stage2.py hiện tại của repo.

Wandb: MỖI trial có 1 wandb run riêng (group="stage23-optuna-search") để theo
dõi trực tiếp trên dashboard trial nào tốt/xấu trong lúc search (không chỉ
final retrain) — khác với run_stage3_optuna.py (chỉ log final retrain).

Cách dùng:
    pip install optuna
    python run_stage23_optuna.py
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

try:
    import wandb
except ImportError:
    wandb = None

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
BATCH_SIZE2 = 4    # Stage 2 — backward per-block, tốn VRAM hơn (xem run_stage2.py)
BATCH_SIZE3 = 16   # Stage 3 — an toàn VRAM (xem run_stage3.py)
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
N_TRIALS              = 30
TRIAL_STAGE2_EPOCHS   = 10    # trần — budget nhỏ / trial
TRIAL_STAGE2_PATIENCE = 3
TRIAL_STAGE2_MIN_DELTA = 0.01
TRIAL_STAGE3_EPOCHS   = 18
TRIAL_STAGE3_PATIENCE = 6
PRUNE_N_STARTUP_TRIALS = 5    # số trial đầu KHÔNG bị prune (cần dữ liệu nền)
PRUNE_N_WARMUP_EPOCHS  = 2    # số epoch Phase B đầu/trial không bị prune

# ── Final retrain (dùng best params từ Optuna) ──────────────────────────
FINAL_STAGE2_EPOCHS    = 100
FINAL_STAGE2_PATIENCE  = 10
FINAL_STAGE2_MIN_DELTA = 0.005
FINAL_STAGE3_EPOCHS    = 100
FINAL_STAGE3_PATIENCE  = 15
FINAL_SAVE_PATH = os.path.join(OUTPUT_DIR, "student_stage23_optuna_best.pth")

LOG_FREQ = 9999   # tắt print mỗi step trong lúc search (đỡ spam console)

# ── Wandb ────────────────────────────────────────────────────────────────
# Khác run_stage3_optuna.py: ở ĐÂY mỗi trial có 1 wandb run riêng (group lại
# theo WANDB_GROUP) để xem được tiến trình search trực tiếp trên dashboard,
# không chỉ final retrain.
USE_WANDB       = True
WANDB_PROJECT   = "slr-mamba-distill"
WANDB_GROUP     = "stage23-optuna-search"
WANDB_NAME_FINAL = "stage23-optuna-wlasl100-final"

SEED   = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ══════════════════════════════════════════════════════════════════════


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class _Stage3PruningBridge:
    """
    Drop-in cho train_stage3(wandb_run=...). Forward MỌI log() đến wandb_run
    thật (nếu có) để theo dõi trial trên dashboard, ĐỒNG THỜI bắt
    "stage3/val_acc" ở Phase B (stage3/phase == 1) để báo cho Optuna
    (trial.report / should_prune) — cắt sớm trial tệ mà không cần sửa
    train_stage3() nội bộ. Không prune dựa trên Stage 2 (xem docstring đầu
    file) — Stage 2 chỉ forward log thẳng tới wandb, không can thiệp.
    """

    def __init__(self, trial, wandb_run=None):
        self.trial = trial
        self.wandb_run = wandb_run
        self.best_val_acc = 0.0
        self._phase_b_steps = 0

    def define_metric(self, *args, **kwargs):
        if self.wandb_run is not None:
            self.wandb_run.define_metric(*args, **kwargs)

    def log(self, d: dict):
        if self.wandb_run is not None:
            self.wandb_run.log(d)

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
        pass  # wandb_run.finish() được gọi riêng ở objective(), không ở đây


def _make_student():
    from models.student import BiMambaSLR

    return BiMambaSLR(
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


def _load_teacher_and_freeze(student, teacher):
    """
    Bỏ Stage 1: load_teacher_weights() copy spatial MHA/FFN/embedding/fc từ
    teacher, đóng băng tất cả, CHỈ temporal_mamba (random init) trainable.
    Đây là điểm khởi đầu của Stage 2 (set_stage2_trainable sẽ mở lại
    multihead_self_attention1/norm1/norm2 phía trên nền này).
    """
    student.load_teacher_weights(teacher)
    bn_count = 0
    for module in student.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.momentum = 0.01
            bn_count += 1
    return student


def objective(trial: "optuna.Trial", teacher, loaders2, loaders3):
    train_loader2, val_loader2 = loaders2
    train_loader3, val_loader3 = loaders3

    seed_everything(SEED)  # cố định random init/augmentation — chỉ hyperparam thay đổi giữa trial

    # ── Stage 2 search space ────────────────────────────────────────────
    stage2_lr          = trial.suggest_float("stage2_lr", 1e-5, 1e-3, log=True)
    stage2_freeze_mlp  = trial.suggest_categorical("stage2_freeze_mlp", [True, False])
    stage2_epochs      = trial.suggest_int("stage2_epochs", 4, 10)

    # ── Stage 3 search space (giống run_stage3_optuna.py) ───────────────
    lr                    = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    alpha                 = trial.suggest_float("alpha", 0.3, 0.7)
    temperature           = trial.suggest_categorical("temperature", [2.0, 3.0, 4.0, 6.0, 8.0])
    weight_decay          = trial.suggest_float("weight_decay", 0.02, 0.3, log=True)
    grad_accum            = trial.suggest_categorical("grad_accum", [2, 4, 8])
    phase_a_epochs        = trial.suggest_int("phase_a_epochs", 3, 8)
    phase_b_warmup_epochs = trial.suggest_int("phase_b_warmup_epochs", 1, 3)

    from distillation.stage2_hidden import train_stage2
    from distillation.stage3_finetune import train_stage3

    student = _make_student()
    student = _load_teacher_and_freeze(student, teacher)

    wandb_run = None
    if USE_WANDB and wandb is not None:
        wandb_run = wandb.init(
            project=WANDB_PROJECT,
            group=WANDB_GROUP,
            name=f"trial-{trial.number:03d}",
            job_type="trial",
            reinit=True,
            config=dict(
                stage2_lr=stage2_lr, stage2_freeze_mlp=stage2_freeze_mlp,
                stage2_epochs=stage2_epochs,
                lr=lr, alpha=alpha, temperature=temperature,
                weight_decay=weight_decay, grad_accum=grad_accum,
                phase_a_epochs=phase_a_epochs,
                phase_b_warmup_epochs=phase_b_warmup_epochs,
            ),
            settings=wandb.Settings(console="off"),
        )
        wandb_run.define_metric("stage2/epoch")
        wandb_run.define_metric("stage2/*", step_metric="stage2/epoch")
        wandb_run.define_metric("stage3/epoch")
        wandb_run.define_metric("stage3/*", step_metric="stage3/epoch")

    bridge = _Stage3PruningBridge(trial, wandb_run=wandb_run)

    try:
        # ── Stage 2: hidden-state alignment ─────────────────────────────
        student = train_stage2(
            student=student,
            teacher=teacher,
            dataloader=train_loader2,
            val_dataloader=val_loader2,
            device=DEVICE,
            lr=stage2_lr,
            num_epochs=stage2_epochs,
            freeze_mlp=stage2_freeze_mlp,
            log_freq=LOG_FREQ,
            patience=TRIAL_STAGE2_PATIENCE,
            min_delta=TRIAL_STAGE2_MIN_DELTA,
            wandb_run=wandb_run,   # log thẳng, KHÔNG qua bridge (không prune ở Stage 2)
            save_path=None,
        )

        # ── Stage 3: full distillation ───────────────────────────────────
        train_stage3(
            student=student,
            teacher=teacher,
            dataloader=train_loader3,
            val_dataloader=val_loader3,
            device=DEVICE,
            lr=lr,
            num_epochs=TRIAL_STAGE3_EPOCHS,
            phase_a_epochs=phase_a_epochs,
            alpha=alpha,
            temperature=temperature,
            grad_accum=grad_accum,
            log_freq=LOG_FREQ,
            patience=TRIAL_STAGE3_PATIENCE,
            wandb_run=bridge,
            save_path=None,
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
        if wandb_run is not None:
            wandb_run.finish()

    return bridge.best_val_acc


def main():
    seed_everything(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if SSTAN_SRC not in sys.path:
        sys.path.insert(0, SSTAN_SRC)

    global NUM_CLASSES
    from models.teacher import TeacherModel

    print(f"Device : {DEVICE}  |  Torch : {torch.__version__}")

    # ── Dataset — 2 DataLoader (batch khác nhau cho Stage2/Stage3), dùng
    #    lại cho mọi trial ────────────────────────────────────────────────
    print("Loading dataset (WLASL100 skeleton)...")
    try:
        from functools import partial
        from torch.utils.data import DataLoader
        from sstan.dataset import Sign_Dataset
        from sstan.datamodule import collate_fn

        with open(SPLIT_FILE) as f:
            content = json.load(f)
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

        train_loader2 = DataLoader(
            train_dataset, batch_size=BATCH_SIZE2, shuffle=True,
            num_workers=NUM_WORKERS, collate_fn=_collate, drop_last=True,
        )
        val_loader2 = DataLoader(
            val_dataset, batch_size=BATCH_SIZE2, shuffle=False,
            num_workers=NUM_WORKERS, collate_fn=_collate, drop_last=False,
        )
        train_loader3 = DataLoader(
            train_dataset, batch_size=BATCH_SIZE3, shuffle=True,
            num_workers=NUM_WORKERS, collate_fn=_collate, drop_last=True,
        )
        val_loader3 = DataLoader(
            val_dataset, batch_size=BATCH_SIZE3, shuffle=False,
            num_workers=NUM_WORKERS, collate_fn=_collate, drop_last=False,
        )
        print(f"Stage2 batches: train={len(train_loader2)} val={len(val_loader2)}  "
              f"(batch={BATCH_SIZE2})")
        print(f"Stage3 batches: train={len(train_loader3)} val={len(val_loader3)}  "
              f"(batch={BATCH_SIZE3})")

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

    if USE_WANDB and wandb is None:
        print("[WARN] USE_WANDB=True nhưng chưa cài wandb (pip install wandb). Tắt wandb.")

    # ══════════════════════════════════════════════════════════════════
    # Optuna search
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print(f"=== Optuna search: {N_TRIALS} trials (Stage 2 + Stage 3) ===")
    print("=" * 60)

    storage_path = f"sqlite:///{os.path.join(OUTPUT_DIR, 'optuna_stage23.db')}"
    study = optuna.create_study(
        study_name="stage23_wlasl100",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=PRUNE_N_STARTUP_TRIALS,
            n_warmup_steps=PRUNE_N_WARMUP_EPOCHS,
        ),
        storage=storage_path,
        load_if_exists=True,   # resume nếu chạy lại
    )

    study.optimize(
        lambda trial: objective(
            trial, teacher,
            loaders2=(train_loader2, val_loader2),
            loaders3=(train_loader3, val_loader3),
        ),
        n_trials=N_TRIALS,
        gc_after_trial=True,
    )

    print("\n" + "=" * 60)
    print("=== Optuna search xong ===")
    print(f"Best val_acc : {study.best_value*100:.2f}%")
    print(f"Best params  : {study.best_params}")
    print("=" * 60)

    best_params_path = os.path.join(OUTPUT_DIR, "stage23_optuna_best_params.json")
    with open(best_params_path, "w") as f:
        json.dump({"best_value": study.best_value, "best_params": study.best_params}, f, indent=2)
    print(f"Đã lưu best params → {best_params_path}")

    # ══════════════════════════════════════════════════════════════════
    # Final retrain: full epoch budget + best params + wandb thật (1 run
    # duy nhất, gộp cả Stage2 + Stage3 — namespace stage2/*, stage3/* tách
    # biệt sẵn nên không đụng nhau trên dashboard)
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("=== Final retrain (Stage 2 + Stage 3) với best params ===")
    print("=" * 60)

    best = study.best_params

    wandb_run = None
    if USE_WANDB and wandb is not None:
        wandb_run = wandb.init(
            project=WANDB_PROJECT,
            name=WANDB_NAME_FINAL,
            reinit=True,
            config=dict(
                stage="2+3", seq_len=SEQ_LEN, n_joints=N_JOINTS,
                embedding_dim=EMBEDDING_DIM, n_blocks=N_BLOCKS,
                n_heads=N_HEADS, d_state=D_STATE, d_conv=D_CONV,
                stage2_epochs=FINAL_STAGE2_EPOCHS, stage3_epochs=FINAL_STAGE3_EPOCHS,
                batch_size2=BATCH_SIZE2, batch_size3=BATCH_SIZE3,
                effective_batch3=BATCH_SIZE3 * best["grad_accum"],
                optuna_best_value=study.best_value,
                **best,
            ),
            settings=wandb.Settings(console="off"),
        )
        wandb_run.define_metric("stage2/epoch")
        wandb_run.define_metric("stage2/*", step_metric="stage2/epoch")
        wandb_run.define_metric("stage3/epoch")
        wandb_run.define_metric("stage3/*", step_metric="stage3/epoch")
        print(f"Wandb : {wandb_run.url}\n")

    from distillation.stage2_hidden import train_stage2
    from distillation.stage3_finetune import train_stage3

    student = _make_student()
    student = _load_teacher_and_freeze(student, teacher)

    print(f"\n[Stage 2] lr={best['stage2_lr']:.2e}  freeze_mlp={best['stage2_freeze_mlp']}  "
          f"epochs(trần)={FINAL_STAGE2_EPOCHS}  patience={FINAL_STAGE2_PATIENCE}")

    stage2_ckpt = os.path.join(OUTPUT_DIR, "student_stage23_optuna_stage2.pth")
    student = train_stage2(
        student=student,
        teacher=teacher,
        dataloader=train_loader2,
        val_dataloader=val_loader2,
        device=DEVICE,
        lr=best["stage2_lr"],
        num_epochs=FINAL_STAGE2_EPOCHS,
        freeze_mlp=best["stage2_freeze_mlp"],
        log_freq=10,
        patience=FINAL_STAGE2_PATIENCE,
        min_delta=FINAL_STAGE2_MIN_DELTA,
        wandb_run=wandb_run,
        save_path=stage2_ckpt,
    )
    print(f"✓ Stage 2 xong → {stage2_ckpt}")

    print(f"\n[Stage 3] lr={best['lr']:.2e}  weight_decay={best['weight_decay']:.4f}  "
          f"alpha={best['alpha']:.3f}  T={best['temperature']}  "
          f"grad_accum={best['grad_accum']}  phase_a_epochs(trần)={best['phase_a_epochs']}  "
          f"phase_b_warmup={best['phase_b_warmup_epochs']}")
    print(f"Target : val_acc ≈ teacher (82.54%)")

    student = train_stage3(
        student=student,
        teacher=teacher,
        dataloader=train_loader3,
        val_dataloader=val_loader3,
        device=DEVICE,
        lr=best["lr"],
        num_epochs=FINAL_STAGE3_EPOCHS,
        phase_a_epochs=best["phase_a_epochs"],
        alpha=best["alpha"],
        temperature=best["temperature"],
        grad_accum=best["grad_accum"],
        log_freq=10,
        patience=FINAL_STAGE3_PATIENCE,
        wandb_run=wandb_run,
        save_path=FINAL_SAVE_PATH,
        weight_decay=best["weight_decay"],
        phase_b_warmup_epochs=best["phase_b_warmup_epochs"],
    )
    print(f"\n✓ Stage 2+3 (Optuna best params) xong → {FINAL_SAVE_PATH}")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
