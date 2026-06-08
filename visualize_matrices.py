"""
Visualize transfer matrices (student) vs attention matrices (teacher)
after Stage 1 distillation.

Usage
-----
python visualize_matrices.py \
    --student_ckpt  checkpoints/student_stage1.pth \
    --teacher_ckpt  "../skeleton-slr-transformer-main (1)/..." \
    --data_root     /data/wlasl \
    --subset        asl100 \
    --num_classes   100 \
    --blocks        0 3 6 9 \
    --n_heads_show  0 1 \
    --device        cuda

Outputs:
  - Per-block heatmap grid  (teacher | student | difference)
  - Frobenius distance bar chart across all blocks
  - Logged to wandb if --wandb is set
  - Saved as PNG to --out_dir
"""

import argparse
import os
import sys
import math

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

_TEACHER_SRC = os.path.join(
    os.path.dirname(__file__), "..",
    "skeleton-slr-transformer-main (1)",
    "skeleton-slr-transformer-main", "src",
)
if _TEACHER_SRC not in sys.path:
    sys.path.insert(0, _TEACHER_SRC)

from models.student import BiMambaSLR
from models.teacher import TeacherModel


# ──────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--student_ckpt", required=True)
    p.add_argument("--teacher_ckpt", default=os.path.join(
        os.path.dirname(__file__), "..",
        "skeleton-slr-transformer-main (1)",
        "skeleton-slr-transformer-main", "scripts", "outputs",
        "2026-06-04", "16-23-19", "checkpoints",
        "epoch=1400-valid_loss=1.1588-valid_accuracy_PI@01=0.8254.ckpt",
    ))
    p.add_argument("--data_root",    required=True)
    p.add_argument("--subset",       default="asl100")
    p.add_argument("--num_classes",  type=int,   default=100)
    p.add_argument("--seq_len",      type=int,   default=50)
    p.add_argument("--n_joints",     type=int,   default=55)
    p.add_argument("--embedding_dim",type=int,   default=128)
    p.add_argument("--n_blocks",     type=int,   default=10)
    p.add_argument("--head_dim",     type=int,   default=64)
    p.add_argument("--n_heads",      type=int,   default=8)
    p.add_argument("--norm_type",    default="batchnorm")
    p.add_argument("--d_state",      type=int,   default=64)
    p.add_argument("--d_conv",       type=int,   default=3)
    p.add_argument("--chunk_size",   type=int,   default=16)
    p.add_argument("--batch_size",   type=int,   default=4)
    p.add_argument("--num_workers",  type=int,   default=2)
    p.add_argument("--device",       default="cuda")

    # Visualization options
    p.add_argument("--blocks",       type=int, nargs="+", default=[0, 3, 6, 9],
                   help="Block indices to visualize")
    p.add_argument("--n_heads_show", type=int, nargs="+", default=[0, 1],
                   help="Head indices to show per block")
    p.add_argument("--sample_idx",   type=int, default=0,
                   help="Which sample in the batch to visualize")
    p.add_argument("--joint_idx",    type=int, default=0,
                   help="Which V-joint index to show for temporal matrix")
    p.add_argument("--out_dir",      default="visualizations")
    p.add_argument("--wandb",        action="store_true")
    p.add_argument("--wandb_project",default="slr-mamba-distill")
    p.add_argument("--wandb_run",    default="matrix-visualization")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────
# Matrix collection
# ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def collect_matrices(student, teacher, x, device):
    """
    Run one forward pass and return:
        teacher_attn  : list[n_blocks] of (BM, V, H, T+1, T+1)
        student_trans : list[n_blocks] of (BM, V, H, T+1, T+1)
        frob_per_block: list[float]
    """
    x = x.to(device)

    # Teacher
    t_out = teacher(x, return_attn=True, return_hidden_states=True)
    teacher_attn   = t_out["temporal_attn_matrices"]  # list[n_blocks]
    teacher_hidden = t_out["hidden_states"]            # list[n_blocks]

    # Student — layer by layer (same as Stage 1)
    student.eval()
    student_trans = []
    frob_per_block = []

    n = min(len(student.blocks), len(teacher_attn))
    for l in range(n):
        student_input = teacher_hidden[l].to(device)
        s_out = student.blocks[l](
            hidden_states=student_input,
            run_mlp_component=False,
            return_transfer_matrix=True,
        )
        tm_s = s_out["transfer_matrix"]                        # (BM, V, H, T+1, T+1)
        tm_t = teacher_attn[l].to(device)                     # (BM, V, H, T+1, T+1)

        frob = torch.linalg.matrix_norm(tm_s - tm_t, ord="fro").mean().item()
        student_trans.append(tm_s.cpu())
        frob_per_block.append(frob)

    teacher_attn_cpu = [m.cpu() for m in teacher_attn]
    return teacher_attn_cpu, student_trans, frob_per_block


# ──────────────────────────────────────────────────────────────────────
# Plotting helpers
# ──────────────────────────────────────────────────────────────────────

def _matrix_to_numpy(tensor, b, v, h):
    """Extract (T+1, T+1) numpy array for sample b, joint v, head h."""
    return tensor[b, v, h].float().numpy()


def plot_block_comparison(
    block_idx, teacher_tm, student_tm,
    sample_idx, joint_idx, heads_show,
    frob_val,
):
    """
    One figure per block with columns = (teacher | student | difference)
    and rows = selected heads.
    """
    n_rows = len(heads_show)
    fig, axes = plt.subplots(
        n_rows, 3,
        figsize=(12, 4 * n_rows),
        squeeze=False,
    )
    fig.suptitle(
        f"Block {block_idx}  —  Frobenius distance: {frob_val:.4f}",
        fontsize=14, fontweight="bold",
    )

    for row, h in enumerate(heads_show):
        t_mat = _matrix_to_numpy(teacher_tm, sample_idx, joint_idx, h)
        s_mat = _matrix_to_numpy(student_tm, sample_idx, joint_idx, h)
        d_mat = s_mat - t_mat

        vmin_ts = min(t_mat.min(), s_mat.min())
        vmax_ts = max(t_mat.max(), s_mat.max())
        abs_d   = np.abs(d_mat).max()

        for col, (mat, title, cmap, vmin, vmax) in enumerate([
            (t_mat, f"Teacher attn  (head {h})", "Blues",  vmin_ts, vmax_ts),
            (s_mat, f"Student Mamba (head {h})", "Blues",  vmin_ts, vmax_ts),
            (d_mat, f"Difference",               "RdBu_r", -abs_d,  abs_d),
        ]):
            ax = axes[row][col]
            im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("Source position")
            ax.set_ylabel("Query position")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    return fig


def plot_frobenius_bar(frob_per_block):
    """Bar chart of Frobenius distance per block."""
    fig, ax = plt.subplots(figsize=(max(8, len(frob_per_block)), 4))
    x = list(range(len(frob_per_block)))
    bars = ax.bar(x, frob_per_block, color="steelblue", edgecolor="white")
    ax.set_xlabel("Block index")
    ax.set_ylabel("Frobenius distance")
    ax.set_title("Per-block Frobenius distance: Student transfer matrix vs Teacher attention")
    ax.set_xticks(x)
    for bar, val in zip(bars, frob_per_block):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"

    # ── wandb ────────────────────────────────────────────────────────
    run = None
    if args.wandb:
        import wandb
        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run,
            config=vars(args),
        )

    # ── Models ───────────────────────────────────────────────────────
    teacher = TeacherModel(
        checkpoint_path=args.teacher_ckpt,
        num_classes=args.num_classes,
        seq_len=args.seq_len,
        n_joints=args.n_joints,
        embedding_dim=args.embedding_dim,
        n_blocks=args.n_blocks,
        head_dim=args.head_dim,
        n_heads=args.n_heads,
        norm_type=args.norm_type,
        device=device,
    )
    teacher.eval()

    student = BiMambaSLR(
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
        chunk_size=args.chunk_size,
    )
    ckpt = torch.load(args.student_ckpt, map_location=device)
    student.load_state_dict(ckpt.get("model_state_dict", ckpt))
    student.to(device)
    student.eval()

    # ── One sample batch ─────────────────────────────────────────────
    try:
        from sstan.datamodule import WLASLMMPoseLightningDataModule
        dm = WLASLMMPoseLightningDataModule(
            data_dir=args.data_root, subset=args.subset,
            seq_len=args.seq_len, num_copies=1,
            batch_size=args.batch_size, num_workers=args.num_workers,
        )
        dm.setup()
        batch = next(iter(dm.val_dataloader()))
    except Exception as e:
        print(f"[WARNING] Could not load dataset ({e}). Using random input.")
        batch = (torch.randn(args.batch_size, 2, args.seq_len, args.n_joints, 1),)

    x = batch[0].float()
    print(f"Input shape: {x.shape}")

    # ── Collect matrices ──────────────────────────────────────────────
    print("Collecting matrices...")
    teacher_attn, student_trans, frob_per_block = collect_matrices(
        student, teacher, x, device
    )

    print("\nFrobenius distances per block:")
    for i, d in enumerate(frob_per_block):
        print(f"  Block {i:2d}: {d:.4f}")

    # ── Frobenius summary bar chart ───────────────────────────────────
    fig_bar = plot_frobenius_bar(frob_per_block)
    bar_path = os.path.join(args.out_dir, "frobenius_per_block.png")
    fig_bar.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close(fig_bar)
    print(f"Saved: {bar_path}")

    if run is not None:
        import wandb
        run.log({"frobenius/bar_chart": wandb.Image(bar_path)})
        for i, d in enumerate(frob_per_block):
            run.log({f"frobenius/block_{i:02d}": d})

    # ── Per-block heatmaps ────────────────────────────────────────────
    blocks_to_show = [b for b in args.blocks if b < len(frob_per_block)]
    b_idx  = min(args.sample_idx, x.shape[0] - 1)
    # joint_idx: clamp to V dimension after BM reshape
    BM = x.shape[0] * x.shape[-1]
    v_idx  = min(args.joint_idx, args.n_joints - 1)
    # re-index to (BM, V, H, T+1, T+1) shape
    bm_idx = b_idx  # M=1 so BM=B

    for l in blocks_to_show:
        fig = plot_block_comparison(
            block_idx   = l,
            teacher_tm  = teacher_attn[l],
            student_tm  = student_trans[l],
            sample_idx  = bm_idx,
            joint_idx   = v_idx,
            heads_show  = args.n_heads_show,
            frob_val    = frob_per_block[l],
        )
        out_path = os.path.join(args.out_dir, f"block_{l:02d}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")

        if run is not None:
            import wandb
            run.log({f"matrices/block_{l:02d}": wandb.Image(out_path)})

    # ── Summary stats ─────────────────────────────────────────────────
    mean_frob = sum(frob_per_block) / len(frob_per_block)
    print(f"\nMean Frobenius distance across all blocks: {mean_frob:.4f}")
    if run is not None:
        run.log({"frobenius/mean": mean_frob})
        run.finish()


if __name__ == "__main__":
    main()
