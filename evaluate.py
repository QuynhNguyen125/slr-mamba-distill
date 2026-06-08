"""
Evaluate BiMambaSLR student on a test set.

Usage
-----
python evaluate.py \\
    --checkpoint  checkpoints/student_best.pth \\
    --data_root   /data/wlasl \\
    --subset      asl100 \\
    --num_classes 100 \\
    --device      cuda
"""

import argparse
import sys
import os
import torch
from torch.utils.data import DataLoader

_TEACHER_SRC = os.path.join(
    os.path.dirname(__file__), "..",
    "skeleton-slr-transformer-main (1)",
    "skeleton-slr-transformer-main", "src",
)
if _TEACHER_SRC not in sys.path:
    sys.path.insert(0, _TEACHER_SRC)

from models.student import BiMambaSLR


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",  required=True)
    p.add_argument("--data_root",   required=True)
    p.add_argument("--subset",      default="asl100")
    p.add_argument("--num_classes", type=int, default=100)
    p.add_argument("--seq_len",     type=int, default=50)
    p.add_argument("--n_joints",    type=int, default=55)
    p.add_argument("--in_channels", type=int, default=2)
    p.add_argument("--embedding_dim", type=int, default=128)
    p.add_argument("--n_blocks",    type=int, default=10)
    p.add_argument("--n_heads",     type=int, default=8)
    p.add_argument("--d_state",     type=int, default=16)
    p.add_argument("--d_conv",      type=int, default=3)
    p.add_argument("--batch_size",  type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device",      default="cuda")
    p.add_argument("--top_k",       type=int, default=5)
    return p.parse_args()


@torch.no_grad()
def evaluate(model, loader, device, top_k=5):
    model.eval()
    correct1 = correctk = total = 0
    for batch in loader:
        x = batch[0].to(device).float()
        labels = batch[1].to(device).long()
        logits = model(x)

        correct1  += (logits.argmax(-1) == labels).sum().item()
        _, pred_k  = logits.topk(min(top_k, logits.size(-1)), dim=-1)
        correctk  += (pred_k == labels.unsqueeze(-1)).any(-1).sum().item()
        total     += labels.size(0)

    return correct1 / total, correctk / total


def main():
    args = get_args()
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"

    # Dataset
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
        loader = dm.test_dataloader()
    except Exception as e:
        raise RuntimeError(f"Could not build test DataLoader: {e}")

    model = BiMambaSLR(
        in_channels=args.in_channels,
        num_classes=args.num_classes,
        seq_len=args.seq_len,
        n_joints=args.n_joints,
        embedding_dim=args.embedding_dim,
        n_blocks=args.n_blocks,
        n_heads=args.n_heads,
        d_state=args.d_state,
        d_conv=args.d_conv,
    )
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.to(device)

    top1, topk = evaluate(model, loader, device, top_k=args.top_k)
    print(f"Top-1  Accuracy : {top1 * 100:.2f}%")
    print(f"Top-{args.top_k} Accuracy : {topk * 100:.2f}%")


if __name__ == "__main__":
    main()
