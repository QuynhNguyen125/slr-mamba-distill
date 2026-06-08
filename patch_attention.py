"""
Direct patch for SSTAN attention module to fix CUDA device mismatch in compute_relative_positions.

This must be imported before any teacher model is instantiated.
"""

import sys
import torch


def patch_attention_module():
    """
    Patch the sstan attention module to fix CUDA device-side assert.

    The original compute_relative_positions creates indices on CPU,
    but relative_position_bias_table is on CUDA → device mismatch.
    """
    try:
        # Try to import the attention module
        from sstan.models.transformers.modules import attention

        # Check if we can find the RelativePositionalEncodeMultiHeadSelfAttention class
        if not hasattr(attention, 'RelativePositionalEncodeMultiHeadSelfAttention'):
            print("[patch_attention] Class not found in attention module")
            return False

        AttentionClass = attention.RelativePositionalEncodeMultiHeadSelfAttention

        # Store original method
        if hasattr(AttentionClass, '_original_compute_relative_positions'):
            print("[patch_attention] Already patched, skipping")
            return True

        original_method = AttentionClass.compute_relative_positions
        AttentionClass._original_compute_relative_positions = original_method

        # Create fixed version
        def compute_relative_positions_fixed(self, seq_len):
            """
            Fixed version: Create indices on same device as relative_position_bias_table.

            Args:
                seq_len: sequence length

            Returns:
                Relative position indices tensor on CUDA/CPU (same device as table)
            """
            device = self.relative_position_bias_table.device
            dtype = self.relative_position_bias_table.dtype

            # Create range tensor on correct device
            range_vec = torch.arange(seq_len, device=device, dtype=torch.long)

            # Compute relative positions
            rel_pos_matrix = range_vec[:, None] - range_vec[None, :]
            rel_pos_matrix = rel_pos_matrix + seq_len - 1

            return rel_pos_matrix

        # Apply patch
        AttentionClass.compute_relative_positions = compute_relative_positions_fixed

        print("[patch_attention] ✓ Successfully patched compute_relative_positions")
        return True

    except ImportError as e:
        print(f"[patch_attention] Could not import sstan attention module: {e}")
        return False
    except Exception as e:
        print(f"[patch_attention] Error during patching: {e}")
        import traceback
        traceback.print_exc()
        return False


# Try to patch immediately on import
if __name__ != "__main__":
    patch_attention_module()
