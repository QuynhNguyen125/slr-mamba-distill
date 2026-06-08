# Fix for Index Out of Bounds Error in Transfer Matrix Computation

## Problem
After successfully completing the first epoch, the code crashes with:
```
RuntimeError: CUDA error: device-side assert triggered
../aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): Assertion
`-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
```

This occurs during validation loss computation after epoch 1 when computing transfer matrices.

## Root Causes

### Issue 1: Unsafe Fancy Indexing in Diagonal Addition (bi_mamba2.py)
**Original code (line 121-122):**
```python
idx = torch.arange(L, device=D.device)
T[:, :, idx, idx] += D.view(1, H, 1)
```

**Problems:**
- Fancy indexing with paired indices `[idx, idx]` can cause CUDA kernel issues
- In-place operations with complex broadcasting rules may fail
- Memory layout issues with advanced indexing

**Fix:**
```python
for i in range(L):
    T[:, :, i, i] = T[:, :, i, i] + D
```

Simple element-wise assignment is more reliable and avoids CUDA indexing issues.

### Issue 2: Shape Mismatch in Transfer Matrix Reshape (student.py)
**Original code (line 183-188):**
```python
if return_transfer_matrix:
    H = tm_out["transfer_matrix"].shape[1]
    # No validation that shapes actually match!
    outputs["transfer_matrix"] = (
        tm_out["transfer_matrix"].view(BM, V, H, T1, T1)
    )
```

**Problems:**
- No validation that input shape (BM*V, H, L, L) can be reshaped to (BM, V, H, T1, T1)
- If L ≠ T1, the view operation fails silently or with cryptic errors
- Silent shape mismatches can corrupt data or cause downstream CUDA errors

**Fix:**
```python
if return_transfer_matrix:
    tm_matrix = tm_out["transfer_matrix"]  # (BM*V, H, L, L)
    B_total, H, L_actual, _ = tm_matrix.shape

    # Explicit validation
    assert B_total == BM * V, (
        f"Batch mismatch: expected {BM * V}, got {B_total}"
    )
    assert L_actual == T1, (
        f"Sequence length mismatch: expected {T1}, got {L_actual}. "
        "This may indicate padding inconsistency between training/validation data."
    )

    # Safe reshape
    outputs["transfer_matrix"] = tm_matrix.view(BM, V, H, T1, T1)
```

### Issue 3: Missing Tensor Contiguity Check (bi_mamba2.py)
**Problem:**
- After einsum and rearrange operations, tensor memory layout might not be contiguous
- Non-contiguous tensors can cause CUDA kernel failures

**Fix:**
```python
# Ensure T is contiguous before diagonal operations
if not T.is_contiguous():
    T = T.contiguous()
```

## Why This Error Appeared After Epoch 1

The error manifests during **validation** (after first epoch) because:

1. **Training uses padding dynamically**: Each batch might have slightly different sequence lengths, and the mixer pads to the nearest multiple of chunk_size
2. **Validation uses different data**: Validation data might have different sequence length characteristics
3. **Shape mismatch accumulates**: The mismatch between padded length and original length compounds when reshaping

The error triggers specifically during validation because:
- `_compute_val_loss()` calls transfer matrix computation
- Validation data might have different sequence statistics
- The view operation finally fails when reshaping attempt is made

## Files Modified

1. **models/mixers/bi_mamba2.py**:
   - Replaced unsafe fancy indexing with safe loop for diagonal addition
   - Added shape validation in `_materialize_transfer_matrix`
   - Added contiguity check before tensor operations

2. **models/student.py**:
   - Added explicit shape assertions before reshape
   - Provides helpful error messages about padding/sequence length mismatches

## Testing

After these fixes, the code should:
- ✓ Complete training epoch without CUDA errors
- ✓ Successfully compute validation loss
- ✓ Generate transfer matrices with correct shapes
- ✓ Provide clear error messages if shape mismatches occur

## Key Lesson

Always validate shapes before reshape/view operations, especially when:
- Tensors have been padded or sliced
- Different data distributions exist (train vs val)
- Fancy indexing is involved in CUDA operations

Explicit assertions are better than silent failures!
