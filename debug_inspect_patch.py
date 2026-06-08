"""
Inspect whether the patch is actually being applied and used.

Usage: python debug_inspect_patch.py
"""

import os
import sys
import inspect

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

print("=" * 80)
print("DEBUG: Inspect Patch Status")
print("=" * 80)

# Setup
SSTAN_SRC = os.path.expanduser(
    "~/sign-language-recognition/skeleton-slr-transformer-main/src"
)

if SSTAN_SRC not in sys.path:
    sys.path.insert(0, SSTAN_SRC)

# Step 1: Import attention module BEFORE patch
print("\n[Step 1] Import attention module (no patch yet)")
try:
    from sstan.models.transformers.modules.attention import (
        RelativePositionalEncodeMultiHeadSelfAttention,
    )
    print("  ✓ Imported")

    # Get the method
    original = RelativePositionalEncodeMultiHeadSelfAttention.compute_relative_positions
    print(f"  Method: {original}")
    print(f"  Source file: {inspect.getsourcefile(original)}")
    print(f"  Source lines: {inspect.getsourcelines(original)[1]}")

    # Print the actual source
    print("\n  Method source (first 300 chars):")
    try:
        source = inspect.getsource(original)
        print("  " + "\n  ".join(source[:300].split("\n")))
    except Exception as e:
        print(f"  Could not get source: {e}")

except Exception as e:
    print(f"  ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 2: Apply patch
print("\n[Step 2] Apply patch")
try:
    from patch_attention import patch_attention_module
    success = patch_attention_module()
    if success:
        print("  ✓ Patch applied")
    else:
        print("  ✗ Patch returned False")
except Exception as e:
    print(f"  ✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Step 3: Check method after patch
print("\n[Step 3] Inspect method after patch")
try:
    # Re-get the method
    patched = RelativePositionalEncodeMultiHeadSelfAttention.compute_relative_positions
    print(f"  Method: {patched}")
    print(f"  Same object as original? {patched is original}")
    print(f"  Source file: {inspect.getsourcefile(patched)}")

    # Print source
    print("\n  Method source (first 400 chars):")
    try:
        source = inspect.getsource(patched)
        print("  " + "\n  ".join(source[:400].split("\n")))
    except Exception as e:
        print(f"  Could not get source: {e}")

except Exception as e:
    print(f"  ✗ Error: {e}")

# Step 4: Check if there's a cached/original version
print("\n[Step 4] Check for cached methods")
try:
    if hasattr(RelativePositionalEncodeMultiHeadSelfAttention, "_original_compute_relative_positions"):
        print("  ✓ Found _original_compute_relative_positions")
        orig_method = RelativePositionalEncodeMultiHeadSelfAttention._original_compute_relative_positions
        print(f"    Method: {orig_method}")
    else:
        print("  No _original_compute_relative_positions found")

except Exception as e:
    print(f"  ✗ Error: {e}")

# Step 5: Create instance and check method binding
print("\n[Step 5] Create instance and test method binding")
try:
    import torch

    attn = RelativePositionalEncodeMultiHeadSelfAttention(
        input_dim=128,
        head_dim=64,
        n_heads=2,
        bias=False,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    attn = attn.to(device)

    print(f"  Instance created on {device}")

    # Get the bound method
    bound_method = attn.compute_relative_positions
    print(f"  Bound method: {bound_method}")

    # Try to get source
    try:
        source = inspect.getsource(bound_method)
        print("\n  Bound method source (first 300 chars):")
        print("  " + "\n  ".join(source[:300].split("\n")))
    except Exception as e:
        print(f"  Could not get source: {e}")

    # Call it and check behavior
    print("\n  Calling compute_relative_positions(5)...")
    indices = attn.compute_relative_positions(5)

    print(f"  Result device: {indices.device}")
    print(f"  Result shape: {indices.shape}")
    print(f"  Result dtype: {indices.dtype}")

    table_device = attn.relative_position_bias_table.device
    print(f"  Table device: {table_device}")

    if indices.device == table_device:
        print(f"  ✓ Devices match!")
    else:
        print(f"  ✗ Device mismatch: indices on {indices.device}, table on {table_device}")

except Exception as e:
    print(f"  ✗ Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("Inspection complete!")
print("=" * 80)
