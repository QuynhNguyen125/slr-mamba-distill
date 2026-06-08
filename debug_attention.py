"""
Debug script to diagnose teacher attention CUDA device-side assert.

Usage: python debug_attention.py
"""

import os
import sys

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import torch
import torch.nn as nn

print("=" * 80)
print("DEBUG: Teacher Attention Device Issue")
print("=" * 80)

# Step 1: Check SSTAN path
SSTAN_SRC = os.path.expanduser(
    "~/sign-language-recognition/skeleton-slr-transformer-main/src"
)
print(f"\n[Step 1] SSTAN path: {SSTAN_SRC}")
print(f"  Exists: {os.path.exists(SSTAN_SRC)}")

if not os.path.exists(SSTAN_SRC):
    print("  ERROR: SSTAN path does not exist!")
    sys.exit(1)

if SSTAN_SRC not in sys.path:
    sys.path.insert(0, SSTAN_SRC)
    print(f"  ✓ Added to sys.path")

# Step 2: Try to import attention module
print("\n[Step 2] Import attention module")
try:
    from sstan.models.transformers.modules.attention import (
        RelativePositionalEncodeMultiHeadSelfAttention,
    )
    print("  ✓ Successfully imported RelativePositionalEncodeMultiHeadSelfAttention")
except ImportError as e:
    print(f"  ✗ Failed to import: {e}")
    sys.exit(1)

# Step 3: Check original method
print("\n[Step 3] Check original compute_relative_positions")
original_method = RelativePositionalEncodeMultiHeadSelfAttention.compute_relative_positions
print(f"  Method: {original_method}")
print(f"  Location: {original_method.__code__.co_filename}:{original_method.__code__.co_firstlineno}")

# Step 4: Test original method to see the issue
print("\n[Step 4] Test original method (this might fail)")
try:
    # Create a simple attention instance
    attn = RelativePositionalEncodeMultiHeadSelfAttention(
        input_dim=128,
        head_dim=64,
        n_heads=2,
        bias=False,
    )

    # Move to CUDA if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")
    attn = attn.to(device)

    # Check table device
    table_device = attn.relative_position_bias_table.device
    print(f"  relative_position_bias_table device: {table_device}")

    # Try compute_relative_positions
    print(f"\n  Calling compute_relative_positions(5)...")
    indices = attn.compute_relative_positions(5)
    print(f"  Indices shape: {indices.shape}")
    print(f"  Indices device: {indices.device}")
    print(f"  Indices dtype: {indices.dtype}")

    if indices.device != table_device:
        print(f"  ✗ DEVICE MISMATCH: indices on {indices.device}, table on {table_device}")

    # Try to index with these indices
    print(f"\n  Attempting to index table with indices...")
    result = attn.relative_position_bias_table[indices]
    print(f"  ✓ Indexing succeeded!")
    print(f"  Result shape: {result.shape}")

except Exception as e:
    print(f"  ✗ Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

# Step 5: Apply patch
print("\n[Step 5] Apply patch")
print("  Patching compute_relative_positions...")

def compute_relative_positions_fixed(self, seq_len):
    """Fixed version with correct device handling."""
    device = self.relative_position_bias_table.device
    dtype = self.relative_position_bias_table.dtype

    range_vec = torch.arange(seq_len, device=device, dtype=torch.long)
    rel_pos_matrix = range_vec[:, None] - range_vec[None, :]
    rel_pos_matrix = rel_pos_matrix + seq_len - 1

    return rel_pos_matrix

RelativePositionalEncodeMultiHeadSelfAttention.compute_relative_positions = (
    compute_relative_positions_fixed
)
print("  ✓ Patch applied")

# Step 6: Test patched method
print("\n[Step 6] Test patched method")
try:
    attn2 = RelativePositionalEncodeMultiHeadSelfAttention(
        input_dim=128,
        head_dim=64,
        n_heads=2,
        bias=False,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    attn2 = attn2.to(device)

    table_device = attn2.relative_position_bias_table.device
    print(f"  relative_position_bias_table device: {table_device}")

    print(f"  Calling compute_relative_positions(5)...")
    indices = attn2.compute_relative_positions(5)
    print(f"  Indices shape: {indices.shape}")
    print(f"  Indices device: {indices.device}")

    if indices.device == table_device:
        print(f"  ✓ Devices match!")
    else:
        print(f"  ✗ DEVICE MISMATCH: {indices.device} vs {table_device}")

    print(f"  Attempting to index table with indices...")
    result = attn2.relative_position_bias_table[indices]
    print(f"  ✓ Indexing succeeded!")
    print(f"  Result shape: {result.shape}")

except Exception as e:
    print(f"  ✗ Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

# Step 7: Test full attention forward
print("\n[Step 7] Test full attention forward pass")
try:
    batch_size = 2
    seq_len = 10
    input_dim = 128

    attn3 = RelativePositionalEncodeMultiHeadSelfAttention(
        input_dim=input_dim,
        head_dim=64,
        n_heads=2,
        bias=False,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    attn3 = attn3.to(device)

    # Create dummy input
    x = torch.randn(batch_size, seq_len, input_dim, device=device)

    print(f"  Input shape: {x.shape}, device: {x.device}")
    print(f"  Calling forward...")

    out, attn_weights = attn3(x)

    print(f"  ✓ Forward pass succeeded!")
    print(f"  Output shape: {out.shape}")
    print(f"  Attention weights shape: {attn_weights.shape}")

except Exception as e:
    print(f"  ✗ Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("Debug complete!")
print("=" * 80)
