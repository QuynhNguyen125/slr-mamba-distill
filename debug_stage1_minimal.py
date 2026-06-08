"""
Minimal stage1 debug: Load teacher and try one forward pass.

Usage: python debug_stage1_minimal.py
"""

import os
import sys

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

print("=" * 80)
print("DEBUG: Minimal Stage1 Forward Pass")
print("=" * 80)

# Setup path
SSTAN_SRC = os.path.expanduser(
    "~/sign-language-recognition/skeleton-slr-transformer-main/src"
)
TEACHER_CKPT = os.path.expanduser(
    "~/sign-language-recognition/skeleton-slr-transformer-main"
    "/scripts/outputs/2026-06-04/16-23-19/checkpoints"
    "/epoch=1400-valid_loss=1.1588-valid_accuracy_PI@01=0.8254.ckpt"
)

print(f"\n[Setup] SSTAN path: {SSTAN_SRC}")
print(f"[Setup] Teacher checkpoint: {TEACHER_CKPT}")
print(f"[Setup] Checkpoint exists: {os.path.exists(TEACHER_CKPT)}")

if SSTAN_SRC not in sys.path:
    sys.path.insert(0, SSTAN_SRC)

# Step 1: Import patch FIRST
print("\n[Step 1] Import and apply patch")
try:
    from patch_attention import patch_attention_module
    patched = patch_attention_module()
    if patched:
        print("  ✓ Patch applied successfully")
    else:
        print("  ✗ Patch failed!")
except Exception as e:
    print(f"  ✗ Error importing patch: {e}")
    import traceback
    traceback.print_exc()

# Step 2: Import torch and check CUDA
print("\n[Step 2] Check PyTorch and CUDA")
import torch
print(f"  PyTorch version: {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  CUDA device: {torch.cuda.get_device_name(0)}")

device = "cuda" if torch.cuda.is_available() else "cpu"

# Step 3: Load teacher model
print("\n[Step 3] Load teacher model")
try:
    # Patch must be done before importing teacher
    import compat
    compat._patch_relative_position_attention()

    from models.teacher import TeacherModel

    print("  Creating teacher model...")
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

    print("  Loading checkpoint...")
    if os.path.exists(TEACHER_CKPT):
        checkpoint = torch.load(TEACHER_CKPT, map_location=device)
        if "state_dict" in checkpoint:
            teacher.load_state_dict(checkpoint["state_dict"], strict=False)
        else:
            teacher.load_state_dict(checkpoint, strict=False)
        print("  ✓ Checkpoint loaded")
    else:
        print(f"  ✗ Checkpoint not found at {TEACHER_CKPT}")

    teacher = teacher.to(device)
    teacher.eval()
    print("  ✓ Teacher model ready")

except Exception as e:
    print(f"  ✗ Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 4: Create dummy input
print("\n[Step 4] Create dummy input")
try:
    batch_size = 2
    n_frames = 51
    n_joints = 55
    in_channels = 2

    x = torch.randn(batch_size, in_channels, n_frames, n_joints, 1, device=device)
    print(f"  Input shape: {x.shape}")
    print(f"  Input device: {x.device}")

except Exception as e:
    print(f"  ✗ Error: {e}")
    sys.exit(1)

# Step 5: Forward pass
print("\n[Step 5] Test forward pass")
print("  This is where the CUDA error should occur if patch didn't work...")
try:
    with torch.no_grad():
        print("  Calling teacher(x)...")
        output = teacher(x)
        print(f"  ✓ Forward pass succeeded!")
        print(f"  Output type: {type(output)}")
        if isinstance(output, dict):
            print(f"  Output keys: {output.keys()}")
        elif isinstance(output, (tuple, list)):
            print(f"  Output length: {len(output)}")

except RuntimeError as e:
    error_msg = str(e)
    if "device-side assert" in error_msg:
        print(f"  ✗ CUDA device-side assert error (patch didn't work):")
        print(f"     {error_msg[:200]}")
    elif "CUDA" in error_msg:
        print(f"  ✗ CUDA error:")
        print(f"     {error_msg[:200]}")
    else:
        print(f"  ✗ Runtime error:")
        print(f"     {error_msg[:200]}")

    import traceback
    print("\n  Full traceback:")
    traceback.print_exc()

except Exception as e:
    print(f"  ✗ Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("Debug complete!")
print("=" * 80)
