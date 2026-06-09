"""
Visualize transfer matrices (student) vs attention matrices (teacher)
after Stage 1 distillation.

4-panel approach — tổng quát cho Sign Language Recognition:

  Panel A — CLS attention row
      T[b, v, h, 0, 1:]  averaged over V joints  → (H, T) bar/heatmap
      Cho biết: frame nào quan trọng nhất để phân loại, theo từng head

  Panel B — Joint group attention
      V=55 = body (0-12) + left_hand (13-33) + right_hand (34-54)
      Average T over joints trong mỗi nhóm → 3 × (H_avg, T+1, T+1) matrices
      Cho biết: mỗi vùng cơ thể tập trung vào khoảng thời gian nào

  Panel C — Head diversity
      Average T over V joints → (H, T+1, T+1) per head
      Show all H=8 heads trong 1 grid → thấy mỗi head học gì khác nhau

  Panel D — Teacher vs Student comparison
      Head-averaged + joint-averaged → single (T+1, T+1) per block
      Cộng thêm Frobenius distance bar chart
      Quick check chất lượng distillation

Usage
-----
python visualize_matrices.py \\
    --student_ckpt  checkpoints/student_stage1.pth \\
    --split_file    data/splits/splits/asl100.json \\
    --pose_root     data/pose_per_individual_videos \\
    --num_classes   100 \\
    --blocks        0 3 6 9 \\
    --device        cuda \\
    --out_dir       visualizations
"""

import argparse
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Path setup ────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_TEACHER_SRC = os.path.join(
    _HERE, "..",
    "skeleton-slr-transformer-main (1)",
    "skeleton-slr-transformer-main", "src",
)
if _TEACHER_SRC not in sys.path:
    sys.path.insert(0, _TEACHER_SRC)

from models.student import BiMambaSLR
from models.teacher import TeacherModel

# ── Joint groups (V=55: 13 body + 21 left hand + 21 right hand) ──────
JOINT_GROUPS = {
    "body":       list(range(0,  13)),
    "left_hand":  list(range(13, 34)),
    "right_hand": list(range(34, 55)),
}
GROUP_COLORS = {"body": "Blues", "left_hand": "Greens", "right_hand": "Oranges"}


# ══════════════════════════════════════════════════════════════════════
# Argument parsing
# ══════════════════════════════════════════════════════════════════════

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--student_ckpt",  required=True)
    p.add_argument("--teacher_ckpt",  default=os.path.expanduser(
        "~/sign-language-recognition/skeleton-slr-transformer-main"
        "/scripts/outputs/2026-06-04/16-23-19/checkpoints"
        "/epoch=1400-valid_loss=1.1588-valid_accuracy_PI@01=0.8254.ckpt"
    ))
    p.add_argument("--split_file",    required=True)
    p.add_argument("--pose_root",     required=True)
    p.add_argument("--num_classes",   type=int,   default=100)
    p.add_argument("--seq_len",       type=int,   default=50)
    p.add_argument("--n_joints",      type=int,   default=55)
    p.add_argument("--embedding_dim", type=int,   default=128)
    p.add_argument("--n_blocks",      type=int,   default=10)
    p.add_argument("--head_dim",      type=int,   default=64)
    p.add_argument("--n_heads",       type=int,   default=8)
    p.add_argument("--norm_type",     default="batchnorm")
    p.add_argument("--d_state",       type=int,   default=64)
    p.add_argument("--d_conv",        type=int,   default=3)
    p.add_argument("--chunk_size",    type=int,   default=16)
    p.add_argument("--batch_size",    type=int,   default=4)
    p.add_argument("--num_workers",   type=int,   default=2)
    p.add_argument("--device",        default="cuda")
    p.add_argument("--blocks",        type=int, nargs="+", default=[0, 3, 6, 9])
    p.add_argument("--sample_idx",    type=int,   default=0)
    p.add_argument("--out_dir",       default="visualizations")
    p.add_argument("--wandb",         action="store_true")
    p.add_argument("--wandb_project", default="slr-mamba-distill")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════
# Matrix collection
# ══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def collect_matrices(student, teacher, x, device):
    """
    Returns:
        teacher_attn  : list[n_blocks] of (BM, V, H, T+1, T+1)
        student_trans : list[n_blocks] of (BM, V, H, T+1, T+1)
        frob_per_block: list[float]
    """
    x = x.to(device)
    t_out = teacher(x, return_attn=True, return_hidden_states=True)
    teacher_attn   = t_out["temporal_attn_matrices"]
    teacher_hidden = t_out["hidden_states"]

    student.eval()
    student_trans  = []
    frob_per_block = []

    n = min(len(student.blocks), len(teacher_attn))
    for l in range(n):
        s_out = student.blocks[l](
            hidden_states=teacher_hidden[l].to(device),
            run_mlp_component=False,
            return_transfer_matrix=True,
        )
        tm_s = s_out["transfer_matrix"].cpu()
        tm_t = teacher_attn[l].cpu()
        frob = torch.linalg.matrix_norm(tm_s - tm_t, ord="fro").mean().item()
        student_trans.append(tm_s)
        frob_per_block.append(frob)

    teacher_attn_cpu = [m.cpu() for m in teacher_attn]
    return teacher_attn_cpu, student_trans, frob_per_block


# ══════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════

def _np(t: torch.Tensor) -> np.ndarray:
    return t.float().numpy()


def _norm01(mat: np.ndarray) -> np.ndarray:
    """Normalize matrix về [0, 1] theo min-max. Trả về bản copy, không thay đổi gốc."""
    lo, hi = mat.min(), mat.max()
    if hi - lo < 1e-8:          # matrix phẳng (tất cả giá trị bằng nhau)
        return np.zeros_like(mat)
    return (mat - lo) / (hi - lo)


def _avg_joints(mat: torch.Tensor, joint_indices) -> torch.Tensor:
    """Average (BM, V, H, L, L) over selected joint indices → (BM, H, L, L)."""
    return mat[:, joint_indices, :, :, :].mean(dim=1)


def _imshow(ax, data, cmap="Blues", vmin=None, vmax=None, title=""):
    im = ax.imshow(data, aspect="auto", cmap=cmap,
                   vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title, fontsize=8)
    ax.set_xlabel("Source frame", fontsize=7)
    ax.set_ylabel("Query frame",  fontsize=7)
    ax.tick_params(labelsize=6)
    return im


def _colorbar(fig, ax, im):
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cb.ax.tick_params(labelsize=6)


# ══════════════════════════════════════════════════════════════════════
# Panel A — CLS attention row
# ══════════════════════════════════════════════════════════════════════

def plot_cls_attention(block_idx, teacher_tm, student_tm, sample_idx):
    """
    T[b, :, h, 0, 1:] averaged over V joints → (H, T) bar chart.
    Row 0 = CLS token, columns 1.. = real frames.

    Cho biết: mỗi head tập trung vào frame nào khi đưa ra prediction.
    Teacher (softmax attention) vs Student (SSM transfer matrix).
    """
    BM, V, H, L, _ = teacher_tm.shape
    b = min(sample_idx, BM - 1)
    T = L - 1   # exclude CLS slot itself

    # Average CLS row over all V joints → (H, T)
    t_cls = teacher_tm[b, :, :, 0, 1:].mean(dim=0).numpy()   # (H, T)
    s_cls = student_tm[b, :, :, 0, 1:].mean(dim=0).numpy()   # (H, T)

    fig, axes = plt.subplots(H, 2, figsize=(14, 2 * H), squeeze=False)
    fig.suptitle(
        f"Panel A — CLS token attention row  (Block {block_idx})\n"
        f"Trung bình qua V={V} joints — mỗi head tập trung vào frame nào",
        fontsize=11, fontweight="bold",
    )

    for h in range(H):
        # Normalize mỗi row về [0,1] riêng — so sánh shape phân phối, không bị lệch scale
        t_row = _norm01(t_cls[h])   # (T,)
        s_row = _norm01(s_cls[h])   # (T,)
        frames = np.arange(T)

        for col, (row, label, color) in enumerate([
            (t_row, f"Teacher — head {h}  [norm 0-1]", "steelblue"),
            (s_row, f"Student — head {h}  [norm 0-1]", "tomato"),
        ]):
            ax = axes[h][col]
            ax.bar(frames, row, color=color, alpha=0.8, width=0.9)
            ax.set_xlim(-0.5, T - 0.5)
            ax.set_ylim(0, 1.05)
            ax.set_title(label, fontsize=8)
            ax.set_xlabel("Frame index", fontsize=7)
            ax.set_ylabel("Normalized weight", fontsize=7)
            ax.tick_params(labelsize=6)

    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════
# Panel B — Joint group attention
# ══════════════════════════════════════════════════════════════════════

def plot_joint_group_attention(block_idx, teacher_tm, student_tm, sample_idx):
    """
    Chia V=55 joints thành 3 nhóm: body, left_hand, right_hand.
    Với mỗi nhóm: average over joints + average over heads → (L, L) matrix.

    Cho biết: vùng cơ thể nào tập trung vào khoảng thời gian nào.
    """
    BM, V, H, L, _ = teacher_tm.shape
    b = min(sample_idx, BM - 1)

    groups = list(JOINT_GROUPS.items())   # [("body", [...]), ...]
    n_groups = len(groups)

    fig, axes = plt.subplots(n_groups, 3, figsize=(14, 4 * n_groups), squeeze=False)
    fig.suptitle(
        f"Panel B — Joint group attention  (Block {block_idx})\n"
        f"Trung bình trong mỗi nhóm khớp, trung bình qua H={H} heads",
        fontsize=11, fontweight="bold",
    )

    for row, (grp_name, grp_idx) in enumerate(groups):
        # (H, L, L) → average over H → (L, L)
        t_mat_raw = _np(_avg_joints(teacher_tm, grp_idx)[b].mean(dim=0))
        s_mat_raw = _np(_avg_joints(student_tm, grp_idx)[b].mean(dim=0))

        # Normalize riêng từng matrix về [0,1] để so sánh pattern hình học
        t_mat = _norm01(t_mat_raw)
        s_mat = _norm01(s_mat_raw)
        # Diff tính trên bản đã normalize → sai lệch pattern, không phải scale
        d_mat = s_mat - t_mat
        abs_d = max(np.abs(d_mat).max(), 1e-8)
        cmap  = GROUP_COLORS[grp_name]

        for col, (mat, title, cm, vlo, vhi) in enumerate([
            (t_mat, f"Teacher — {grp_name}  [0-1]",  cmap,    0, 1),
            (s_mat, f"Student — {grp_name}  [0-1]",  cmap,    0, 1),
            (d_mat, f"Diff  — {grp_name}",            "RdBu_r", -abs_d, abs_d),
        ]):
            ax  = axes[row][col]
            im  = _imshow(ax, mat, cm, vlo, vhi, title)
            _colorbar(fig, ax, im)

    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════
# Panel C — Head diversity
# ══════════════════════════════════════════════════════════════════════

def plot_head_diversity(block_idx, teacher_tm, student_tm, sample_idx):
    """
    Average over V joints, show mỗi head riêng → (H, L, L) grid.
    Mỗi cột = 1 head, hàng trên = teacher, hàng dưới = student.

    Cho biết: mỗi head chuyên về pattern thời gian gì (local vs global,
    early vs late, diagonal vs long-range).
    """
    BM, V, H, L, _ = teacher_tm.shape
    b = min(sample_idx, BM - 1)

    # Average over all V joints → (H, L, L)
    t_all = _np(teacher_tm[b].mean(dim=0))   # (H, L, L)
    s_all = _np(student_tm[b].mean(dim=0))   # (H, L, L)

    fig, axes = plt.subplots(2, H, figsize=(2.5 * H, 6), squeeze=False)
    fig.suptitle(
        f"Panel C — Head diversity  (Block {block_idx})\n"
        f"Trung bình qua V={V} joints — hàng trên = Teacher, hàng dưới = Student",
        fontsize=11, fontweight="bold",
    )

    for h in range(H):
        # Normalize riêng từng matrix để mỗi head hiển thị đầy đủ range màu
        # → dễ thấy pattern local/global của từng head dù magnitude khác nhau
        t_mat = _norm01(t_all[h])
        s_mat = _norm01(s_all[h])

        im_t = _imshow(axes[0][h], t_mat, "Blues", 0, 1, f"Teacher h{h}\n[norm 0-1]")
        im_s = _imshow(axes[1][h], s_mat, "Reds",  0, 1, f"Student h{h}\n[norm 0-1]")
        _colorbar(fig, axes[0][h], im_t)
        _colorbar(fig, axes[1][h], im_s)

    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════
# Panel D — Teacher vs Student summary + Frobenius
# ══════════════════════════════════════════════════════════════════════

def plot_summary_comparison(teacher_attn_all, student_trans_all, frob_per_block,
                             blocks_show, sample_idx):
    """
    Mỗi block được show bằng 1 matrix duy nhất: average over V + average over H.
    Columns: teacher | student | difference.
    Thêm Frobenius distance bar chart ở cuối.

    Cho biết: nhanh nhất về chất lượng distillation tổng thể.
    """
    n = len(blocks_show)
    fig = plt.figure(figsize=(14, 4 * n + 4))
    gs  = gridspec.GridSpec(n + 1, 3, figure=fig, hspace=0.45, wspace=0.35)

    fig.suptitle(
        "Panel D — Summary: Teacher vs Student  (avg over all joints & heads)\n"
        "Nhanh nhất để đánh giá chất lượng distillation mỗi block",
        fontsize=11, fontweight="bold",
    )

    for row, l in enumerate(blocks_show):
        if l >= len(teacher_attn_all):
            continue
        BM, V, H, L, _ = teacher_attn_all[l].shape
        b = min(sample_idx, BM - 1)

        # Average over V joints and H heads → (L, L)
        t_mat_raw = _np(teacher_attn_all[l][b].mean(dim=(0, 1)))
        s_mat_raw = _np(student_trans_all[l][b].mean(dim=(0, 1)))

        # Normalize riêng về [0,1] → so sánh pattern, không bị lệch scale
        t_mat = _norm01(t_mat_raw)
        s_mat = _norm01(s_mat_raw)
        d_mat = s_mat - t_mat
        abs_d = max(np.abs(d_mat).max(), 1e-8)
        frob  = frob_per_block[l] if l < len(frob_per_block) else 0.0

        for col, (mat, title, cm, vlo, vhi) in enumerate([
            (t_mat, f"Teacher  block {l}  [0-1]",       "Blues",  0, 1),
            (s_mat, f"Student  block {l}  [0-1]",       "Blues",  0, 1),
            (d_mat, f"Diff  (Frob={frob:.3f})",         "RdBu_r", -abs_d, abs_d),
        ]):
            ax = fig.add_subplot(gs[row, col])
            im = _imshow(ax, mat, cm, vlo, vhi, title)
            _colorbar(fig, ax, im)

    # Frobenius bar chart — last row spans all 3 columns
    ax_bar = fig.add_subplot(gs[n, :])
    x    = list(range(len(frob_per_block)))
    bars = ax_bar.bar(x, frob_per_block, color="steelblue", edgecolor="white")
    ax_bar.set_xlabel("Block index", fontsize=9)
    ax_bar.set_ylabel("Frobenius distance", fontsize=9)
    ax_bar.set_title("Frobenius distance per block", fontsize=10)
    ax_bar.set_xticks(x)
    for bar, val in zip(bars, frob_per_block):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(frob_per_block) * 0.01,
            f"{val:.3f}", ha="center", va="bottom", fontsize=7,
        )

    return fig


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    args   = get_args()
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    # ── wandb ────────────────────────────────────────────────────────
    run = None
    if args.wandb:
        import wandb
        run = wandb.init(
            project=args.wandb_project,
            name="matrix-visualization",
            config=vars(args),
        )

    # ── Models ───────────────────────────────────────────────────────
    print("Loading teacher...")
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
    teacher.to(device).eval()

    print("Loading student...")
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
    ckpt = torch.load(args.student_ckpt, map_location=device, weights_only=False)
    student.load_state_dict(ckpt.get("model_state_dict", ckpt))
    student.to(device).eval()

    # ── Data ─────────────────────────────────────────────────────────
    print("Loading one batch...")
    try:
        import json
        from functools import partial
        from torch.utils.data import DataLoader
        from sstan.dataset import Sign_Dataset
        from sstan.datamodule import collate_fn

        with open(args.split_file) as f:
            content = json.load(f)
        num_classes = len(sorted(set(e["gloss"] for e in content)))
        _collate = partial(collate_fn, num_classes=num_classes)

        ds = Sign_Dataset(
            index_file_path=args.split_file,
            pose_root=args.pose_root,
            split="val",
            num_samples=args.seq_len,
            num_copies=1,
            sample_strategy="seq",
            skeleton_augmentation=False,
        )
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=_collate)
        batch = next(iter(loader))
        x = batch["skeleton_data"].float()
    except Exception as e:
        print(f"[WARNING] Dataset failed ({e}). Dùng random input.")
        x = torch.randn(args.batch_size, 2, args.seq_len, args.n_joints, 1)

    print(f"Input shape: {x.shape}")

    # ── Collect ──────────────────────────────────────────────────────
    print("Collecting matrices (forward pass)...")
    teacher_attn, student_trans, frob_per_block = collect_matrices(
        student, teacher, x, device
    )

    print("\nFrobenius distances per block:")
    for i, d in enumerate(frob_per_block):
        bar = "█" * int(d * 20)
        print(f"  Block {i:2d}: {d:.4f}  {bar}")

    blocks_show = [b for b in args.blocks if b < len(frob_per_block)]
    bidx        = min(args.sample_idx, x.shape[0] - 1)

    # ── Generate all panels ───────────────────────────────────────────
    # Dùng commit=False cho mỗi log → tích lũy cùng 1 step trên wandb
    # Dùng try/except cho mỗi panel → panel lỗi không làm crash toàn bộ

    def _log(key, path):
        """Log 1 image lên wandb với commit=False (chưa flush)."""
        if run:
            import wandb
            run.log({key: wandb.Image(path)}, commit=False)

    # Panel A: CLS attention row
    print("\nPanel A — CLS attention row...")
    for l in blocks_show:
        try:
            fig  = plot_cls_attention(l, teacher_attn[l], student_trans[l], bidx)
            path = os.path.join(args.out_dir, f"panelA_block{l:02d}_cls_attention.png")
            fig.savefig(path, dpi=150, bbox_inches="tight");  plt.close(fig)
            print(f"  Saved: {path}")
            _log(f"A_cls/block_{l:02d}", path)
        except Exception as e:
            print(f"  [ERROR] Panel A block {l}: {e}")

    # Panel B: Joint group attention
    print("\nPanel B — Joint group attention...")
    for l in blocks_show:
        try:
            fig  = plot_joint_group_attention(l, teacher_attn[l], student_trans[l], bidx)
            path = os.path.join(args.out_dir, f"panelB_block{l:02d}_joint_groups.png")
            fig.savefig(path, dpi=150, bbox_inches="tight");  plt.close(fig)
            print(f"  Saved: {path}")
            _log(f"B_groups/block_{l:02d}", path)
        except Exception as e:
            print(f"  [ERROR] Panel B block {l}: {e}")

    # Panel C: Head diversity
    print("\nPanel C — Head diversity...")
    for l in blocks_show:
        try:
            fig  = plot_head_diversity(l, teacher_attn[l], student_trans[l], bidx)
            path = os.path.join(args.out_dir, f"panelC_block{l:02d}_head_diversity.png")
            fig.savefig(path, dpi=150, bbox_inches="tight");  plt.close(fig)
            print(f"  Saved: {path}")
            _log(f"C_heads/block_{l:02d}", path)
        except Exception as e:
            print(f"  [ERROR] Panel C block {l}: {e}")

    # Panel D: Summary + Frobenius bar
    print("\nPanel D — Summary comparison...")
    try:
        fig  = plot_summary_comparison(
            teacher_attn, student_trans, frob_per_block, blocks_show, bidx,
        )
        path = os.path.join(args.out_dir, "panelD_summary_frobenius.png")
        fig.savefig(path, dpi=150, bbox_inches="tight");  plt.close(fig)
        print(f"  Saved: {path}")
        _log("D_summary", path)
    except Exception as e:
        print(f"  [ERROR] Panel D: {e}")

    # ── Commit tất cả images cùng 1 step + flush ─────────────────────
    mean_frob = sum(frob_per_block) / max(len(frob_per_block), 1)
    print(f"\nMean Frobenius distance: {mean_frob:.4f}")
    if run:
        # commit=True (default) → flush toàn bộ buffer commit=False ở trên
        run.log({"frobenius/mean": mean_frob})
        run.finish()


if __name__ == "__main__":
    main()
