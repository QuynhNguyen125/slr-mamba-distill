"""
Fix for SSTAN attention: ensure compute_relative_positions creates indices on correct device.

The problem: relative_position_bias_table stays on CPU after model.to(cuda)
The solution: Create indices on the SAME device as relative_position_bias_table
"""

import torch


def patch_attention_module():
    """
    Patch compute_relative_positions to handle device correctly.
    Indices must be created on the same device as relative_position_bias_table.
    """
    try:
        from sstan.models.transformers.modules import attention

        if not hasattr(attention, 'RelativePositionalEncodeMultiHeadSelfAttention'):
            print("[patch_attention] Class not found")
            return False

        AttentionClass = attention.RelativePositionalEncodeMultiHeadSelfAttention

        # Check if already patched
        if hasattr(AttentionClass, '_patched_device_fix'):
            print("[patch_attention] Already patched")
            return True

        # Save original
        original_compute = AttentionClass.compute_relative_positions

        # Create fixed version
        def compute_relative_positions_fixed(self, seq_len):
            """
            Fixed: Create indices on SAME device as relative_position_bias_table.

            This fixes the device mismatch error when:
            - relative_position_bias_table is on CUDA
            - but indices are created on CPU
            """
            # KEY FIX: Get device from the table itself
            device = self.relative_position_bias_table.device

            # Create range_vec on the CORRECT device
            range_vec = torch.arange(seq_len, device=device, dtype=torch.long)

            # Compute relative positions
            rel_pos_matrix = range_vec[:, None] - range_vec[None, :]
            rel_pos_matrix = rel_pos_matrix + seq_len - 1

            return rel_pos_matrix

        # Apply patch
        AttentionClass.compute_relative_positions = compute_relative_positions_fixed
        AttentionClass._patched_device_fix = True

        print("[patch_attention] ✓ Patched compute_relative_positions for device consistency")
        return True

    except Exception as e:
        print(f"[patch_attention] Error: {e}")
        import traceback
        traceback.print_exc()
        return False


# Auto-patch on import
if __name__ != "__main__":
    patch_attention_module()
