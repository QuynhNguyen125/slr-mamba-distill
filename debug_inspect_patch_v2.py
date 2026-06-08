"""
Fixed version: Inspect patch status with correct attention signature.
"""

import os
import sys
import inspect

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

print("=" * 80)
print("DEBUG: Inspect Patch Status (v2 - Fixed)")
print("=" * 80)

# Setup
SSTAN_SRC = os.path.expanduser(
    "~/sign-language-recognition/skeleton-slr-transformer-main/src"
)

if SSTAN_SRC not in sys.path:
    sys.path.insert(0, SSTAN_SRC)

# Import
print("\n[Step 1] Import attention module")
try:
    from sstan.models.transformers.modules.attention import (
        RelativePositionalEncodeMultiHeadSelfAttention,
    )
    print("  ✓ Imported")
except Exception as e:
    print(f"  ✗ Error: {e}")
    sys.exit(1)

# Check signature
print("\n[Step 2] Check __init__ signature")
try:
    import inspect
    sig = inspect.signature(RelativePositionalEncodeMultiHeadSelfAttention.__init__)
    print(f"  Signature: {sig}")
except Exception as e:
    print(f"  Error: {e}")

# Apply patch
print("\n[Step 3] Apply patch")
try:
    from patch_attention import patch_attention_module
    success = patch_attention_module()
    print(f"  Patch result: {success}")
except Exception as e:
    print(f"  ✗ Error: {e}")
    sys.exit(1)

# Test method after patch
print("\n[Step 4] Test patched method")
try:
    import torch

    # Create instance with correct args (from the signature)
    # RelativePositionalEncodeMultiHeadSelfAttention(input_dim, head_dim, n_heads, seq_len, bias)
    seq_len = 52  # T+1 from the data

    attn = RelativePositionalEncodeMultiHeadSelfAttention(
        input_dim=128,
        head_dim=64,
        n_heads=8,
        seq_len=seq_len,
        bias=False,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    attn = attn.to(device)

    print(f"  Instance created on {device}")
    print(f"  relative_position_bias_table device: {attn.relative_position_bias_table.device}")

    # Call the method
    print(f"\n  Calling compute_relative_positions(seq_len)...")
    indices = attn.compute_relative_positions(seq_len)

    print(f"  Indices shape: {indices.shape}")
    print(f"  Indices device: {indices.device}")
    print(f"  Indices dtype: {indices.dtype}")

    table_device = attn.relative_position_bias_table.device
    print(f"  Table device: {table_device}")

    if indices.device == table_device:
        print(f"  ✓ DEVICES MATCH!")
    else:
        print(f"  ✗ DEVICE MISMATCH")

except Exception as e:
    print(f"  ✗ Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

# Test forward pass
print("\n[Step 5] Test attention forward pass")
try:
    batch_size = 2
    x = torch.randn(batch_size, seq_len, 128, device=device)

    print(f"  Input shape: {x.shape}, device: {x.device}")
    print(f"  Calling attn(x)...")

    out, weights = attn(x)

    print(f"  ✓ Forward pass succeeded!")
    print(f"  Output shape: {out.shape}")
    print(f"  Weights shape: {weights.shape}")

except Exception as e:
    print(f"  ✗ Error: {type(e).__name__}: {e}")
    if "device-side assert" in str(e):
        print("  This is the CUDA device-side assert error!")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("Debug complete!")
print("=" * 80)
