"""
Fix for SSTAN attention module.

Issues fixed:
1. Device mismatch: indices created on CPU, table on CUDA
2. Out of bounds: validation data seq_len > relative_position_bias_table size
"""

import torch


def patch_attention_module():
    """
    Patch compute_relative_positions and forward method to handle:
    - Device mismatch (indices on wrong device)
    - Sequence length mismatch (validation data longer than training)
    """
    try:
        from sstan.models.transformers.modules import attention

        if not hasattr(attention, 'RelativePositionalEncodeMultiHeadSelfAttention'):
            print("[patch_attention] Class not found")
            return False

        AttentionClass = attention.RelativePositionalEncodeMultiHeadSelfAttention

        # Check if already patched
        if hasattr(AttentionClass, '_patched_comprehensive'):
            print("[patch_attention] Already patched")
            return True

        # Save original forward
        original_forward = AttentionClass.forward

        # Create patched forward
        def forward_patched(self, x, mask=None):
            """
            Patched forward: truncate x if seq_len exceeds relative_position_bias_table size.
            """
            # Get current seq_len
            seq_len = x.shape[-2]
            table_size = self.relative_position_bias_table.shape[0]
            max_seq_len = (table_size + 1) // 2

            # Truncate if needed
            if seq_len > max_seq_len:
                print(f"[patch] WARNING: seq_len={seq_len} > max={max_seq_len}, truncating to {max_seq_len}...")
                x = x[..., :max_seq_len, :]
                if mask is not None:
                    mask = mask[..., :max_seq_len, :max_seq_len]

            # Now call original forward with corrected input
            return original_forward(self, x, mask)

        # Patch forward
        AttentionClass.forward = forward_patched

        # Also patch compute_relative_positions for direct use
        def compute_relative_positions_fixed(self, seq_len):
            """Fixed version with device handling."""
            device = self.relative_position_bias_table.device
            range_vec = torch.arange(seq_len, device=device, dtype=torch.long)
            rel_pos_matrix = range_vec[:, None] - range_vec[None, :]
            rel_pos_matrix = rel_pos_matrix + seq_len - 1
            return rel_pos_matrix

        AttentionClass.compute_relative_positions = compute_relative_positions_fixed
        AttentionClass._patched_comprehensive = True

        print("[patch_attention] ✓ Comprehensive patch applied (device + bounds handling)")
        return True

    except Exception as e:
        print(f"[patch_attention] Error: {e}")
        import traceback
        traceback.print_exc()
        return False


# Auto-patch on import
if __name__ != "__main__":
    patch_attention_module()
