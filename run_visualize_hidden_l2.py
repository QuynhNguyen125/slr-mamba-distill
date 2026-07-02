"""
run_visualize_hidden_l2.py
──────────────────────────
So sánh L2 distance per block giữa teacher và student hidden states,
TRƯỚC và SAU Stage 2 — đây là metric đúng để đánh giá Stage 2.

Sự khác biệt so với run_visualize_stage2.py (hiện tại chỉ plot transfer matrices):
    run_visualize_stage2.py  → Frobenius(attention_teacher, mixer_student)   ← Stage 1 metric
    Script này              → L2(block_output_student, block_output_teacher) ← Stage 2 metric ✓

Cách dùng:
    python run_visualize_hidden_l2.py

Output: visualizations_hidden_l2/hidden_l2_comparison.png
"""

import os, sys, warnings, logging

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

import compat  # inject torchvision stub

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ══════════════════════════════════════════════════════════════════════
# CONFIG — chỉnh paths theo máy bạn
# ══════════════════════════════════════════════════════════════════════

TEACHER_CKPT       = os.path.expanduser(
    "~/sign-language-recognition/skeleton-slr-transformer-main"
    "/scripts/outputs/2026-06-04/16-23-19/checkpoints"
    "/epoch=1400-valid_loss=1.1588-valid_accuracy_PI@01=0.8254.ckpt"
)
STUDENT_STAGE1_CKPT = "checkpoints/student_stage1_best.pth"
STUDENT_STAGE2_CKPT = "checkpoints/student_stage2_best_nofreeze.pth"  # freeze_mlp=False

# Nếu muốn so sánh cả phiên bản freeze_mlp=True:
# STUDENT_STAGE2_CKPT = "checkpoints/student_stage2_best.pth"

SPLIT_FILE = os.path.expanduser("~/slr-mamba-distill/data/splits/splits/asl100.json")
POSE_ROOT  = os.path.expanduser("~/slr-mamba-distill/data/pose_per_individual_videos")
SSTAN_SRC  = os.path.expanduser(
    "~/sign-language-recognition/skeleton-slr-transformer-main/src"
)

SEQ_LEN       = 50
N_JOINTS      = 55
IN_CHANNELS   = 2
EMBEDDING_DIM = 128
N_BLOCKS      = 10
HEAD_DIM      = 64
N_HEADS       = 8
NORM_TYPE     = "batchnorm"
FFN_EXPAND    = 4.0
FFN_DROPOUT   = 0.25
MAX_STOCH     = 0.25
D_STATE       = 64
D_CONV        = 3
CHUNK_SIZE    = 16

# Số batch để tính trung bình L2 distance (tăng = chính xác hơn, chậm hơn)
N_BATCHES  = 20
BATCH_SIZE = 4
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "visualizations_hidden_l2"

# Stage 2 mode — phải khớp với cách bạn train Stage 2:
#   freeze_mlp=False → target = block_outputs[l]   (full block output)
#   freeze_mlp=True  → target = pre_ffn_states[l]  (trước FFN)
FREEZE_MLP = False   # ← đổi thành True nếu bạn train với freeze_mlp=True

# ══════════════════════════════════════════════════════════════════════


def get_target_key(freeze_mlp: bool) -> str:
    return "pre_ffn_states" if freeze_mlp else "block_outputs"


@torch.no_grad()
def compute_l2_per_block(student, teacher, loader, n_batches, device, freeze_mlp):
    """
    Tính L2 distance per block trung bình trên n_batches batch.

    Theo đúng Stage 2 logic:
        student_input[l]  = teacher.hidden_states[l]   (input đến block l)
        student_output[l] = student.blocks[l](student_input[l])
        teacher_target[l] = teacher.block_outputs[l]   (hoặc pre_ffn_states nếu freeze_mlp=True)
        L2[l] = mean over tokens of ||student_output[l] - teacher_target[l]||_2
    """
    target_key = get_target_key(freeze_mlp)
    student.eval()
    teacher.eval()

    block_l2_sum   = np.zeros(N_BLOCKS)
    block_l2_count = 0

    for i, batch in enumerate(loader):
        if i >= n_batches:
            break

        if isinstance(batch, dict):
            x = batch["skeleton_data"].to(device).float()
        else:
            x = batch[0].to(device).float()

        # Teacher forward — lấy block_inputs và targets
        t_out = teacher(x, return_attn=False, return_hidden_states=True)
        block_inputs   = t_out["hidden_states"]   # list[N_BLOCKS]: input đến block l
        teacher_targets = t_out[target_key]        # list[N_BLOCKS]: target cho student

        n = min(N_BLOCKS, len(block_inputs), len(teacher_targets))

        for l in range(n):
            s_input  = block_inputs[l].to(device)
            t_target = teacher_targets[l].to(device)

            # Student block l forward (dùng TEACHER input — giống Stage 2 training)
            s_out = student.blocks[l](
                hidden_states=s_input,
                run_mlp_component=not freeze_mlp,
                return_transfer_matrix=False,
            )
            s_hidden = s_out["hidden_states"]   # (BM, T+1, V, D)

            # L2 norm per token, average over all positions and batch
            # shape: (BM, T+1, V)
            l2_per_token = torch.norm(s_hidden - t_target, p=2, dim=-1)
            block_l2_sum[l] += l2_per_token.mean().item()

        block_l2_count += 1

    return block_l2_sum / max(block_l2_count, 1)  # (N_BLOCKS,)


def load_student(ckpt_path, num_classes, device):
    from models.student import BiMambaSLR
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
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    student.load_state_dict(ckpt.get("model_state_dict", ckpt))
    return student.to(device).eval()


def plot_results(l2_stage1, l2_stage2, freeze_mlp, output_path):
    """
    Grouped bar chart: L2 distance per block, Stage 1 vs Stage 2.
    Thêm delta (% improvement) trên mỗi cặp cột.
    """
    blocks = np.arange(N_BLOCKS)
    bar_w  = 0.35
    target_label = "pre_ffn_states (freeze_mlp=True)" if freeze_mlp else "block_outputs (freeze_mlp=False)"

    # ── Tính improvement ──────────────────────────────────────────────
    improvement = (l2_stage1 - l2_stage2) / (l2_stage1 + 1e-8) * 100  # % giảm

    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[2, 1], hspace=0.45)

    # ── Panel trên: L2 distance grouped bar ───────────────────────────
    ax1 = fig.add_subplot(gs[0])
    b1 = ax1.bar(blocks - bar_w/2, l2_stage1, bar_w,
                 label="After Stage 1", color="#5b8dd9", alpha=0.85, edgecolor="white")
    b2 = ax1.bar(blocks + bar_w/2, l2_stage2, bar_w,
                 label="After Stage 2", color="#e07b54", alpha=0.85, edgecolor="white")

    # Annotate values
    for rect in b1:
        h = rect.get_height()
        ax1.text(rect.get_x() + rect.get_width()/2, h + 0.005,
                 f"{h:.3f}", ha="center", va="bottom", fontsize=7.5, color="#333")
    for rect in b2:
        h = rect.get_height()
        ax1.text(rect.get_x() + rect.get_width()/2, h + 0.005,
                 f"{h:.3f}", ha="center", va="bottom", fontsize=7.5, color="#333")

    ax1.set_title(
        f"Hidden State L2 Distance: Teacher vs Student — Stage 1 vs Stage 2\n"
        f"(Target: {target_label}  |  avg over {N_BATCHES} batches)",
        fontsize=11, fontweight="bold", pad=12
    )
    ax1.set_xlabel("Block index", fontsize=10)
    ax1.set_ylabel("Mean L2 distance per token", fontsize=10)
    ax1.set_xticks(blocks)
    ax1.legend(fontsize=10)
    ax1.grid(axis="y", alpha=0.3, linestyle="--")
    ax1.set_xlim(-0.6, N_BLOCKS - 0.4)

    # ── Panel dưới: % improvement ─────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    colors = ["#3aab6b" if v >= 0 else "#d94f4f" for v in improvement]
    ax2.bar(blocks, improvement, color=colors, alpha=0.85, edgecolor="white")

    for i, v in enumerate(improvement):
        ax2.text(i, v + (0.5 if v >= 0 else -2.5),
                 f"{v:+.1f}%", ha="center", va="bottom", fontsize=8.5,
                 color="#333", fontweight="bold")

    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_title("L2 Improvement after Stage 2 (% reduction, green=better)", fontsize=10)
    ax2.set_xlabel("Block index", fontsize=10)
    ax2.set_ylabel("% L2 reduction", fontsize=10)
    ax2.set_xticks(blocks)
    ax2.grid(axis="y", alpha=0.3, linestyle="--")
    ax2.set_xlim(-0.6, N_BLOCKS - 0.4)

    # ── Summary stats ─────────────────────────────────────────────────
    mean_s1 = l2_stage1.mean()
    mean_s2 = l2_stage2.mean()
    overall  = (mean_s1 - mean_s2) / mean_s1 * 100
    best_block = int(np.argmax(improvement))

    fig.text(
        0.5, 0.01,
        f"Overall L2 — Stage1: {mean_s1:.4f}  →  Stage2: {mean_s2:.4f}  "
        f"({overall:+.1f}% change)   |   Best block: {best_block} ({improvement[best_block]:+.1f}%)",
        ha="center", fontsize=10, color="#555", style="italic"
    )

    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[Plot] Saved → {output_path}")
    return mean_s1, mean_s2, overall


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if SSTAN_SRC not in sys.path:
        sys.path.insert(0, SSTAN_SRC)

    import json
    from functools import partial
    from torch.utils.data import DataLoader
    from sstan.dataset import Sign_Dataset
    from sstan.datamodule import collate_fn
    from models.teacher import TeacherModel

    # ── Dataset ───────────────────────────────────────────────────────
    print("Loading dataset...")
    with open(SPLIT_FILE) as f:
        content = json.load(f)
    glosses     = sorted(set(e["gloss"] for e in content))
    num_classes = len(glosses)

    val_dataset = Sign_Dataset(
        index_file_path=SPLIT_FILE,
        pose_root=POSE_ROOT,
        split="val",
        num_samples=SEQ_LEN,
        num_copies=1,
        sample_strategy="rnd_start",
        skeleton_augmentation=False,
    )
    _collate = partial(collate_fn, num_classes=num_classes)
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, collate_fn=_collate, drop_last=False,
    )
    print(f"Val samples: {len(val_dataset)}  |  Using first {N_BATCHES} batches")

    # ── Teacher ───────────────────────────────────────────────────────
    print("\nLoading teacher...")
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
    ).to(DEVICE).eval()
    print("Teacher loaded ✓")

    # ── Stage 1 student ───────────────────────────────────────────────
    print(f"\nLoading Stage 1 student: {STUDENT_STAGE1_CKPT}")
    if not os.path.exists(STUDENT_STAGE1_CKPT):
        print(f"[ERROR] Không tìm thấy: {STUDENT_STAGE1_CKPT}")
        sys.exit(1)
    student1 = load_student(STUDENT_STAGE1_CKPT, num_classes, DEVICE)
    print("Stage 1 student loaded ✓")

    # ── Stage 2 student ───────────────────────────────────────────────
    print(f"\nLoading Stage 2 student: {STUDENT_STAGE2_CKPT}")
    if not os.path.exists(STUDENT_STAGE2_CKPT):
        print(f"[ERROR] Không tìm thấy: {STUDENT_STAGE2_CKPT}")
        sys.exit(1)
    student2 = load_student(STUDENT_STAGE2_CKPT, num_classes, DEVICE)
    print("Stage 2 student loaded ✓")

    target_key = get_target_key(FREEZE_MLP)
    print(f"\nTarget key : {target_key}  (freeze_mlp={FREEZE_MLP})")
    print(f"Computing L2 distance over {N_BATCHES} val batches...")

    # ── Compute L2 per block ──────────────────────────────────────────
    print("\n[1/2] Stage 1 L2 distance...")
    l2_s1 = compute_l2_per_block(student1, teacher, val_loader, N_BATCHES, DEVICE, FREEZE_MLP)

    print("[2/2] Stage 2 L2 distance...")
    l2_s2 = compute_l2_per_block(student2, teacher, val_loader, N_BATCHES, DEVICE, FREEZE_MLP)

    # ── Console summary ───────────────────────────────────────────────
    print("\n" + "="*65)
    print(f"{'Block':>6} | {'Stage1 L2':>10} | {'Stage2 L2':>10} | {'Δ%':>8}")
    print("-"*65)
    for l in range(N_BLOCKS):
        delta = (l2_s1[l] - l2_s2[l]) / (l2_s1[l] + 1e-8) * 100
        mark = "▲ better" if delta > 0 else "▼ worse "
        print(f"{l:>6} | {l2_s1[l]:>10.4f} | {l2_s2[l]:>10.4f} | {delta:>+7.1f}%  {mark}")
    print("="*65)
    mean_s1 = l2_s1.mean(); mean_s2 = l2_s2.mean()
    overall = (mean_s1 - mean_s2) / mean_s1 * 100
    print(f"{'MEAN':>6} | {mean_s1:>10.4f} | {mean_s2:>10.4f} | {overall:>+7.1f}%")
    print("="*65)

    # ── Judgment ──────────────────────────────────────────────────────
    print()
    if overall > 10:
        print("✓ Stage 2 hiệu quả: L2 giảm đáng kể toàn bộ (>10%).")
    elif overall > 0:
        print("~ Stage 2 cải thiện nhẹ. Xem xét tăng epochs hoặc LR.")
    else:
        print("✗ Stage 2 không giúp ích — L2 không giảm. Kiểm tra FREEZE_MLP và checkpoint.")

    # ── Plot ──────────────────────────────────────────────────────────
    out_path = os.path.join(OUTPUT_DIR, "hidden_l2_comparison.png")
    plot_results(l2_s1, l2_s2, FREEZE_MLP, out_path)


if __name__ == "__main__":
    main()
