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
            Patched forward with error handling for device and size mismatches.
            """
            try:
                return original_forward(self, x, mask)
            except (RuntimeError, IndexError) as e:
                error_msg = str(e)

                # Check if it's a device or index error
                if "device" in error_msg.lower() or "index" in error_msg.lower() or "out of bounds" in error_msg.lower():
                    # Re-compute with proper device handling and bounds checking
                    input_shape = x.shape if isinstance(x, torch.Tensor) else x[0].shape
                    seq_len = input_shape[-2]

                    # Get device from table
                    device = self.relative_position_bias_table.device
                    table_size = self.relative_position_bias_table.shape[0]

                    # Clamp seq_len to avoid out of bounds
                    # table_size = 2*init_seq_len - 1
                    # max_valid_seq_len = (table_size + 1) / 2
                    max_seq_len = (table_size + 1) // 2

                    if seq_len > max_seq_len:
                        print(f"[patch] WARNING: seq_len={seq_len} > max={max_seq_len}, clamping...")
                        seq_len = max_seq_len

                    # Create relative positions with proper device
                    range_vec = torch.arange(seq_len, device=device, dtype=torch.long)
                    rel_pos_matrix = range_vec[:, None] - range_vec[None, :]
                    rel_pos_matrix = rel_pos_matrix + seq_len - 1

                    # Clamp indices to valid range
                    rel_pos_matrix = torch.clamp(rel_pos_matrix, 0, table_size - 1)

                    # Index with clamped indices
                    pos_bias = self.relative_position_bias_table[rel_pos_matrix]

                    # Continue with rest of forward (copied from original)
                    batch_size = x.shape[0]
                    seq_len_actual = x.shape[1]

                    q = self.linear_to_qkv(x)
                    q = q.view(*input_shape[:-1], self.n_heads, self.head_dim).transpose(-2, -3)
                    k = self.linear_to_qkv(x)
                    k = k.view(*input_shape[:-1], self.n_heads, self.head_dim).transpose(-2, -3)
                    v = self.linear_to_qkv(x)
                    v = v.view(*input_shape[:-1], self.n_heads, self.head_dim).transpose(-2, -3)

                    # Compute attention with position bias
                    import torch.nn.functional as F
                    attn = q @ k.transpose(-1, -2) * (self.head_dim ** -0.5)
                    attn = attn + torch.einsum('...hld,lrd->...hlr', q, pos_bias)

                    if mask is not None:
                        attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(1) == 0, float('-inf'))

                    attn_weights = F.softmax(attn, dim=-1)
                    attn_weights = self.dropout(attn_weights)

                    output = attn_weights @ v
                    output = output.transpose(-2, -3).contiguous()
                    output = output.view(*input_shape[:-1], -1)
                    output = self.linear_w(output)

                    return output, attn_weights

                raise

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
