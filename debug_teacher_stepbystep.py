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

# Step 1: Add sstan to path FIRST (nhưng chưa import)
print("\n[Step 1] Add SSTAN to sys.path")
if SSTAN_SRC not in sys.path:
    sys.path.insert(0, SSTAN_SRC)
    print("  ✓ Added to path")

# Step 1b: Patch NGAY TRƯỚC KHI BẤT KỲ SSTAN MODULE NÀO ĐƯỢC IMPORT
print("\n[Step 1b] Apply patch NGAY (trước khi import sstan)")
import torch

def patch_attention_before_import():
    """Patch forward method để intercept lúc gọi - cách này chắc chắn work."""
    try:
        # Import module
        from sstan.models.transformers.modules.attention import (
            RelativePositionalEncodeMultiHeadSelfAttention,
        )

        print("  Imported RelativePositionalEncodeMultiHeadSelfAttention")

        # Cách 1: Patch compute_relative_positions
        def compute_relative_positions_fixed(self, seq_len):
            """Fixed: Create indices on correct device."""
            device = self.relative_position_bias_table.device
            range_vec = torch.arange(seq_len, device=device, dtype=torch.long)
            rel_pos_matrix = range_vec[:, None] - range_vec[None, :]
            return rel_pos_matrix + seq_len - 1

        RelativePositionalEncodeMultiHeadSelfAttention.compute_relative_positions = (
            compute_relative_positions_fixed
        )
        print("  ✓ Patched compute_relative_positions")

        # Cách 2: THÊM patch forward để chắc chắn
        original_forward = RelativePositionalEncodeMultiHeadSelfAttention.forward

        def forward_with_device_fix(self, x, mask=None):
            """Forward với fix: ensure compute_relative_positions uses correct device."""
            # Force patch lần nữa (vì method có thể bị cache)
            device = self.relative_position_bias_table.device

            def compute_rel_pos_fixed(this, seq_len):
                """Fixed method - nhận self làm tham số đầu."""
                range_vec = torch.arange(seq_len, device=device, dtype=torch.long)
                rel_pos_matrix = range_vec[:, None] - range_vec[None, :]
                return rel_pos_matrix + seq_len - 1

            # Tạm thời replace method
            old_compute = self.compute_relative_positions
            import types
            self.compute_relative_positions = types.MethodType(compute_rel_pos_fixed, self)

            try:
                result = original_forward(self, x, mask)
            finally:
                # Restore
                self.compute_relative_positions = old_compute

            return result

        RelativePositionalEncodeMultiHeadSelfAttention.forward = forward_with_device_fix
        print("  ✓ Patched forward method (thêm layer bảo vệ)")

        return True

    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

success = patch_attention_before_import()
if not success:
    print("  PATCH FAILED!")
    sys.exit(1)

# Step 2: Check PyTorch
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
    # Find checkpoint
    TEACHER_CKPT = os.path.expanduser(
        "~/sign-language-recognition/skeleton-slr-transformer-main"
        "/scripts/outputs/2026-06-04/16-23-19/checkpoints"
        "/epoch=1400-valid_loss=1.1588-valid_accuracy_PI@01=0.8254.ckpt"
    )
    print(f"  Checkpoint: {TEACHER_CKPT}")
    print(f"  Exists: {os.path.exists(TEACHER_CKPT)}")

    if not os.path.exists(TEACHER_CKPT):
        print("  ✗ Checkpoint not found!")
        # Try to find alternative
        print("  Searching for any .ckpt file...")
        import glob
        ckpts = glob.glob(os.path.expanduser("~/sign-language-recognition/**/epoch=*.ckpt"), recursive=True)
        if ckpts:
            TEACHER_CKPT = ckpts[0]
            print(f"  Found: {TEACHER_CKPT}")
        else:
            print("  No checkpoint found!")
            sys.exit(1)

    teacher = TeacherModel(
        checkpoint_path=TEACHER_CKPT,
        num_classes=100,
        in_channels=2,
        seq_len=50,
        n_joints=55,
        embedding_dim=128,
        n_blocks=10,
        head_dim=64,
        n_heads=8,
        norm_type="batchnorm",
        ffn_expand_ratio=4.0,
        ffn_dropout_ratio=0.25,
        max_stochastic_depth_rate=0.25,
        device=device,
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
