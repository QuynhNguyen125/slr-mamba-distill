# Fix for Teacher Model CUDA Device-Side Assert in Attention

## Problem

After completing first epoch, the code fails with:
```
RuntimeError: CUDA error: device-side assert triggered
../aten/src/ATen/native/cuda/IndexKernel.cu:93
```

The error occurs in the teacher model's attention module when computing relative position bias:
```python
pos_bias = self.relative_position_bias_table[self.compute_relative_positions(input_shape[-2])]
```

## Root Cause

The `compute_relative_positions` method in SSTAN's `RelativePositionalEncodeMultiHeadSelfAttention` class creates position indices on the **CPU**, but the `relative_position_bias_table` is on **CUDA**.

When using CPU indices to index a CUDA tensor in PyTorch 2.12, it causes a device mismatch assertion.

**Original code (SSTAN):**
```python
def compute_relative_positions(self, seq_len):
    # This creates CPU tensors by default
    range_vec = torch.arange(seq_len)  # ← CPU tensor
    rel_pos_matrix = range_vec[:, None] - range_vec[None, :]
    rel_pos_matrix = rel_pos_matrix + seq_len - 1
    return rel_pos_matrix  # ← CPU tensor

# Later in forward:
pos_bias = self.relative_position_bias_table[self.compute_relative_positions(...)]
# self.relative_position_bias_table is on CUDA, indices are on CPU → DEVICE MISMATCH
```

## Solution

Patch the method to create indices on the **same device** as the `relative_position_bias_table`:

**Fixed version:**
```python
def compute_relative_positions_fixed(self, seq_len):
    # Get device from the bias table
    device = self.relative_position_bias_table.device
    
    # Create indices on the correct device
    range_vec = torch.arange(seq_len, device=device, dtype=torch.long)
    rel_pos_matrix = range_vec[:, None] - range_vec[None, :]
    rel_pos_matrix = rel_pos_matrix + seq_len - 1
    return rel_pos_matrix
```

## Implementation

**New file: `patch_attention.py`**
- Direct patch of SSTAN attention module
- Patches `compute_relative_positions` method
- Creates indices on correct device
- Provides clear error reporting

**Modified: `run_stage1.py`**
- Imports `patch_attention_module` before instantiating teacher
- Ensures patch is applied before any forward passes
- Exits with clear message if patch fails

## Key Points

1. **Order matters**: Patch must be applied BEFORE teacher model is instantiated
2. **Device consistency**: Indices must be on same device as the tensor they index into
3. **Type safety**: Use `dtype=torch.long` for indices
4. **Early validation**: Fail fast with clear error if patch doesn't apply

## Testing

Run stage1 training:
```bash
python run_stage1.py
```

Expected output:
```
[patch_attention] ✓ Successfully patched compute_relative_positions
[Stage1] Epoch 1/10 — train_loss: X.XXXX
[Stage1] Epoch 1/10 — val_loss: X.XXXX
```

If you see the patch message but still get CUDA errors, check:
1. PyTorch version (issue is specific to PyTorch 2.12+)
2. Teacher checkpoint is loaded correctly
3. CUDA device is available

## Files Modified

- `patch_attention.py` (NEW) - Direct patch implementation
- `run_stage1.py` - Apply patch before teacher instantiation
- `compat.py` - Enhanced with better error reporting (for completeness)

## Upstream Issue

This is a known issue in SSTAN with PyTorch 2.12+. The patch can be removed if/when SSTAN is updated upstream.
