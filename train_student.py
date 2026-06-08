"""
Main distillation training entry point.

Runs three MOHAWK stages sequentially:
    Stage 1 — Transfer matrix alignment (Bi-Mamba mixer vs teacher attention)
    Stage 2 — Hidden state alignment
    Stage 3 — Full distillation (KL + CE)

The data pipeline uses the sstan WLASL dataset which produces
tensors of shape (B, 2, T, V, M).

Usage
-----
python train_student.py \\
    --teacher_ckpt   checkpoints/teacher_best.pth \\
    --data_root      /data/wlasl \\
    --subset         asl100 \\
    --num_classes    100 \\
    --device         cuda
"""

import argparse
import os
import sys
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

# Add sstan to path for dataset access
_TEACHER_SRC = os.path.join(
    os.path.dirname(__file__), "..",
    "skeleton-slr-transformer-main (1)",
    "skeleton-slr-transformer-main", "src",
)
if _TEACHER_SRC not in sys.path:
    sys.path.insert(0, _TEACHER_SRC)

from models.student import BiMambaSLR
from models.teacher import TeacherModel
from distillation.stage1_matrix import train_stage1
from distillation.stage2_hidden import train_stage2
from distillation.stage3_full   import train_stage3


# ── Default hyper-parameters from teacher config ──────────────────────
TEACHER_DEFAULTS = dict(
    embedding_dim=128, n_blocks=10, head_dim=64, n_heads=8,
    ffn_expand_ratio=4.0, ffn_dropout_ratio=0.25,
    max_stochastic_depth_rate=0.25, use_bias=False,
)


def get_args():
    p = argparse.ArgumentParser()

    # Paths
    p.add_argument(
        "--teacher_ckpt",
        default=os.path.join(
            os.path.dirname(__file__), "..",
            "skeleton-slr-transformer-main (1)",
            "skeleton-slr-transformer-main", "scripts", "outputs",
            "2026-06-04", "16-23-19", "checkpoints",
            "epoch=1400-valid_loss=1.1588-valid_accuracy_PI@01=0.8254.ckpt",
        ),
        help="PyTorch Lightning .ckpt from the teacher training run",
    )
    p.add_argument("--data_root",    required=True, help="Root folder of WLASL skeleton data")
    p.add_argument("--subset",       default="asl100", help="WLASL subset (asl100 / asl300 etc.)")
    p.add_argument("--output_dir",   default="checkpoints")

    # Dataset
    p.add_argument("--num_classes",  type=int,   default=100)
    p.add_argument("--seq_len",      type=int,   default=50)
    p.add_argument("--n_joints",     type=int,   default=55)
    p.add_argument("--in_channels",  type=int,   default=2)
    p.add_argument("--batch_size",   type=int,   default=16)
    p.add_argument("--num_workers",  type=int,   default=4)

    # Student architecture (must match teacher dims for distillation)
    p.add_argument("--embedding_dim",  type=int,   default=128)
    p.add_argument("--n_blocks",       type=int,   default=10)
    p.add_argument("--head_dim",       type=int,   default=64)
    p.add_argument("--n_heads",        type=int,   default=8)
    p.add_argument("--norm_type",      default="batchnorm",
                   choices=["batchnorm", "layernorm"])
    p.add_argument("--d_state",        type=int,   default=16,
                   help="SSM state dim (not in teacher; student-only hyper-param)")
    p.add_argument("--d_conv",         type=int,   default=3)
    p.add_argument("--ffn_expand_ratio", type=float, default=4.0)
    p.add_argument("--ffn_dropout_ratio", type=float, default=0.25)
    p.add_argument("--max_stochastic_depth_rate", type=float, default=0.25)

    # Training
    p.add_argument("--device",  default="cuda")
    p.add_argument("--seed",    type=int, default=42)

    # Stage 1
    p.add_argument("--s1_epochs", type=int,   default=10)
    p.add_argument("--s1_lr",     type=float, default=1e-3)

    # Stage 2
    p.add_argument("--s2_epochs",     type=int,   default=10)
    p.add_argument("--s2_lr",         type=float, default=5e-4)
    p.add_argument("--s2_freeze_mlp", action="store_true", default=True)

    # Stage 3
    p.add_argument("--s3_epochs",      type=int,   default=50)
    p.add_argument("--s3_lr",          type=float, default=1e-4)
    p.add_argument("--s3_alpha",       type=float, default=0.5)
    p.add_argument("--s3_temperature", type=float, default=4.0)

    # Skip
    p.add_argument("--skip_stage1", action="store_true")
    p.add_argument("--skip_stage2", action="store_true")
    p.add_argument("--resume",      default="")

    # Wandb
    p.add_argument("--wandb",          action="store_true", help="Enable wandb logging")
    p.add_argument("--wandb_project",  default="slr-mamba-distill")
    p.add_argument("--wandb_name",     default=None, help="Run name (auto if None)")
    p.add_argument("--viz_freq",       type=int, default=2,
                   help="Log matrix heatmaps to wandb every N epochs (Stage 1)")
    p.add_argument("--log_freq",       type=int, default=50,
                   help="Print/log loss every N steps")

    return p.parse_args()


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dataloaders(args):
    """Build train/val DataLoaders using the sstan WLASL dataset."""
    try:
        from sstan.datamodule import WLASLMMPoseLightningDataModule
        dm = WLASLMMPoseLightningDataModule(
            data_dir=args.data_root,
            subset=args.subset,
            seq_len=args.seq_len,
            num_copies=4,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        dm.setup()
        return dm.train_dataloader(), dm.val_dataloader()
    except Exception as e:
        print(f"[WARNING] Could not load sstan DataModule ({e}).")
        print("Falling back to custom dataset — ensure your dataset returns (skeleton, label) "
              "where skeleton has shape (B, 2, T, V, M).")
        return None, None


def main():
    args = get_args()
    seed_everything(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"

    # ── Wandb init ────────────────────────────────────────────────────
    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config=vars(args),
            resume="allow",
        )
        print(f"Wandb run: {wandb_run.url}")

    # ── Data ─────────────────────────────────────────────────────────
    print("Loading data...")
    train_loader, val_loader = build_dataloaders(args)
    if train_loader is None:
        raise RuntimeError("Could not build DataLoader. Provide a valid --data_root.")

    # ── Teacher ──────────────────────────────────────────────────────
    print(f"Loading teacher from {args.teacher_ckpt}...")
    teacher = TeacherModel(
        checkpoint_path=args.teacher_ckpt,
        num_classes=args.num_classes,
        in_channels=args.in_channels,
        seq_len=args.seq_len,
        n_joints=args.n_joints,
        embedding_dim=args.embedding_dim,
        n_blocks=args.n_blocks,
        head_dim=args.head_dim,
        n_heads=args.n_heads,
        norm_type=args.norm_type,
        device=device,
    )

    # ── Student ──────────────────────────────────────────────────────
    student = BiMambaSLR(
        in_channels=args.in_channels,
        num_classes=args.num_classes,
        seq_len=args.seq_len,
        n_joints=args.n_joints,
        embedding_dim=args.embedding_dim,
        n_blocks=args.n_blocks,
        head_dim=args.head_dim,
        n_heads=args.n_heads,
        norm_type=args.norm_type,
        d_state=args.d_state,
        d_conv=args.d_conv,
        chunk_size=getattr(args, "chunk_size", 16),
        ffn_expand_ratio=args.ffn_expand_ratio,
        ffn_dropout_ratio=args.ffn_dropout_ratio,
        max_stochastic_depth_rate=args.max_stochastic_depth_rate,
    )

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        student.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
        print(f"Resumed from {args.resume}")

    total_params = sum(p.numel() for p in student.parameters())
    print(f"Student total parameters: {total_params:,}")
    if wandb_run is not None:
        wandb_run.summary["student_params"] = total_params

    # ── Weight transfer: Teacher → Student (trước Stage 1) ───────────
    # Copy embedding, spatial MHA, FFN, norms, classifier từ teacher
    # Đóng băng tất cả trừ temporal_mamba → sẵn sàng cho Stage 1
    print("\n=== Weight Transfer: Teacher → Student ===")
    student.load_teacher_weights(teacher)

    # ── Stage 1 ──────────────────────────────────────────────────────
    if not args.skip_stage1:
        print("\n=== Stage 1: Transfer Matrix Alignment ===")
        student = train_stage1(
            student, teacher, train_loader,
            device=device,
            lr=args.s1_lr,
            num_epochs=args.s1_epochs,
            log_freq=getattr(args, "log_freq", 50),
            viz_freq=getattr(args, "viz_freq", 2),
            wandb_run=wandb_run,
            save_path=os.path.join(args.output_dir, "student_stage1.pth"),
        )

    # ── Stage 2 ──────────────────────────────────────────────────────
    if not args.skip_stage2:
        print("\n=== Stage 2: Hidden State Alignment ===")
        student = train_stage2(
            student, teacher, train_loader,
            device=device,
            lr=args.s2_lr,
            num_epochs=args.s2_epochs,
            freeze_mlp=args.s2_freeze_mlp,
            log_freq=getattr(args, "log_freq", 50),
            wandb_run=wandb_run,
            save_path=os.path.join(args.output_dir, "student_stage2.pth"),
        )

    # ── Stage 3 ──────────────────────────────────────────────────────
    print("\n=== Stage 3: Full Distillation ===")
    student = train_stage3(
        student, teacher, train_loader,
        val_dataloader=val_loader,
        device=device,
        lr=args.s3_lr,
        num_epochs=args.s3_epochs,
        alpha=args.s3_alpha,
        temperature=args.s3_temperature,
        scheduler_patience=5,
        save_path=os.path.join(args.output_dir, "student_best.pth"),
        log_freq=getattr(args, "log_freq", 50),
        wandb_run=wandb_run,
    )

    _save(student, os.path.join(args.output_dir, "student_final.pth"))
    print(f"\nDone. Checkpoints saved to {args.output_dir}/")

    if wandb_run is not None:
        wandb_run.finish()


def _save(model, path):
    torch.save({"model_state_dict": model.state_dict()}, path)
    print(f"  Saved: {path}")


if __name__ == "__main__":
    main()
