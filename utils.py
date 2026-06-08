"""Utility helpers shared across training scripts."""

import random
import numpy as np
import torch
from torch.utils.data import Subset


def balance_val_split(dataset, test_split=0.2):
    """
    Stratified train/val split preserving class distribution.
    Adapted from skeleton project's __balance_val_split.
    """
    targets = np.array(dataset.targets)
    classes = np.unique(targets)

    train_idx, val_idx = [], []
    for cls in classes:
        idx = np.where(targets == cls)[0].tolist()
        random.shuffle(idx)
        split_pt = max(1, int(len(idx) * test_split))
        val_idx.extend(idx[:split_pt])
        train_idx.extend(idx[split_pt:])

    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def log_class_statistics(dataset):
    targets = np.array(dataset.targets if hasattr(dataset, "targets") else
                       [dataset[i][3].item() for i in range(len(dataset))])
    classes, counts = np.unique(targets, return_counts=True)
    print(f"Classes: {len(classes)}  Samples: {len(targets)}")
    print(f"  Min per class: {counts.min()}  Max: {counts.max()}  Mean: {counts.mean():.1f}")


def save_checkpoint(model, optimizer, epoch, path, **extra):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        **extra,
    }, path)


def load_checkpoint(model, optimizer, path, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt.get("epoch", 0)
