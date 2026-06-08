"""
Step-by-step debug of teacher forward pass to find exact location of CUDA error.
"""

import os
import sys

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

print("=" * 80)
print("DEBUG: Teacher Forward Pass Step-by-Step")
print("=" * 80)

# Setup
SSTAN_SRC = os.path.expanduser(
    "~/sign-language-recognition/skeleton-slr-transformer-main/src"
)

print(f"\nSetup:")
print(f"  SSTAN_SRC: {SSTAN_SRC}")
print(f"  Exists: {os.path.exists(SSTAN_SRC)}")

if SSTAN_SRC not in sys.path:
    sys.path.insert(0, SSTAN_SRC)

# Step 1: Patch FIRST
print("\n[Step 1] Apply patch BEFORE importing teacher")
try:
    from patch_attention import patch_attention_module
    success = patch_attention_module()
    if success:
        print("  ✓ Patch applied successfully")
    else:
        print("  ✗ Patch failed")
except Exception as e:
    print(f"  ✗ Error: {e}")
    sys.exit(1)

# Step 2: Import and check
import torch
print(f"\n[Step 2] PyTorch info")
print(f"  Version: {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"  Device: {device}")

# Step 3: Import teacher
print(f"\n[Step 3] Import TeacherModel")
try:
    from models.teacher import TeacherModel
    print("  ✓ Imported")
except Exception as e:
    print(f"  ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 4: Create teacher
print(f"\n[Step 4] Create teacher model")
try:
    teacher = TeacherModel(
        embedding_dim=128,
        n_blocks=10,
        head_dim=64,
        n_heads=8,
        norm_type="batchnorm",
        ffn_expand_ratio=4.0,
        ffn_dropout_ratio=0.25,
        max_stochastic_depth=0.25,
    )
    print("  ✓ Created")

    teacher = teacher.to(device)
    teacher.eval()
    print(f"  ✓ Moved to {device} and set to eval mode")

except Exception as e:
    print(f"  ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 5: Create dummy input
print(f"\n[Step 5] Create dummy input")
try:
    batch_size = 1
    n_frames = 51
    n_joints = 55
    in_channels = 2

    x = torch.randn(batch_size, in_channels, n_frames, n_joints, 1, device=device)
    print(f"  Input shape: {x.shape}")
    print(f"  ✓ Input created")

except Exception as e:
    print(f"  ✗ Error: {e}")
    sys.exit(1)

# Step 6: Test forward pass
print(f"\n[Step 6] Test teacher forward pass")
print(f"  This is where CUDA error should occur if patch didn't work...")

try:
    with torch.no_grad():
        print(f"  Calling teacher(x)...")
        output = teacher(x)

    print(f"  ✓ SUCCESS! No CUDA error!")
    print(f"  Output type: {type(output)}")

except RuntimeError as e:
    error_str = str(e)

    if "device-side assert" in error_str:
        print(f"  ✗ CUDA DEVICE-SIDE ASSERT ERROR!")
        print(f"  Error message: {error_str[:300]}")

        # Try to extract line info
        import traceback
        tb = traceback.format_exc()
        print(f"\n  Traceback (last 500 chars):")
        print("  " + "\n  ".join(tb[-500:].split("\n")))

    elif "CUDA" in error_str:
        print(f"  ✗ CUDA ERROR!")
        print(f"  Error: {error_str[:300]}")

    else:
        print(f"  ✗ OTHER ERROR!")
        print(f"  Error: {error_str[:300]}")

    import traceback
    traceback.print_exc()

except Exception as e:
    print(f"  ✗ ERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

# Step 7: If it worked, try with larger batch
if "output" in locals():
    print(f"\n[Step 7] Try with larger batch")
    try:
        batch_size = 4
        x_large = torch.randn(batch_size, in_channels, n_frames, n_joints, 1, device=device)

        with torch.no_grad():
            output_large = teacher(x_large)

        print(f"  ✓ Larger batch (B={batch_size}) also works!")

    except Exception as e:
        print(f"  ✗ Larger batch failed: {type(e).__name__}")

print("\n" + "=" * 80)
print("Debug complete!")
print("=" * 80)
