# Debug Guide: CUDA Device-Side Assert in Teacher Attention

## Quick Start

Pull latest changes:
```bash
cd ~/slr-mamba-distill
git pull origin main
```

Run debug scripts in this order:

### 1. Test Patch Application
```bash
python debug_inspect_patch.py
```

**What it checks:**
- Whether patch is being applied to the method
- Whether the patched method is actually being used
- Source code of the method before and after patch

**Expected output (SUCCESS):**
```
✓ Imported
✓ Patch applied
Same object as original? False
✓ Devices match!
```

**If you see:**
- `Same object as original? True` → Patch didn't replace the method
- `Device mismatch: indices on cpu, table on cuda` → Patch didn't work

### 2. Test Attention Module Directly
```bash
python debug_attention.py
```

**What it checks:**
- Can import attention module
- Original method has device mismatch
- After patching, method works correctly

**Look for:**
```
[Step 6] Test patched method
  ✓ Devices match!
  ✓ Indexing succeeded!
```

### 3. Test Full Teacher Forward
```bash
python debug_stage1_minimal.py
```

**What it checks:**
- Patch is applied before teacher loads
- Teacher model can be loaded
- Forward pass works without CUDA errors

**Expected output:**
```
[Step 5] Test forward pass
  ✓ Forward pass succeeded!
```

---

## Troubleshooting

### Case 1: Patch not applied (Same object as original? True)

**Problem:** The method wasn't actually replaced

**Solutions:**
1. Check that patch_attention.py is in the same directory as run_stage1.py
2. Verify sstan is in sys.path BEFORE patch is applied
3. Add debug print in run_stage1.py:

```python
from patch_attention import patch_attention_module
result = patch_attention_module()
print(f"Patch result: {result}")  # Should print True
if not result:
    print("ERROR: Patch failed!")
    sys.exit(1)
```

### Case 2: Device mismatch persists (indices on cpu, table on cuda)

**Problem:** Patch is applied but indices still on CPU

**Possible causes:**
- Method is cached somewhere
- Multiple imports are creating different classes
- PyTorch compiled/JIT code is bypassing the patch

**Debug:**
```python
# In debug_inspect_patch.py, check:
print(inspect.getsource(patched))  # Should show device= in torch.arange
```

**Solution:**
Try this alternative patch in run_stage1.py:

```python
# Immediately after adding sstan to sys.path
from sstan.models.transformers.modules import attention

# Patch the forward method instead of compute_relative_positions
original_forward = attention.RelativePositionalEncodeMultiHeadSelfAttention.forward

def forward_with_device_fix(self, x, mask=None):
    # Temporarily patch compute_relative_positions for this call
    device = self.relative_position_bias_table.device
    
    def compute_rel_pos_fixed(seq_len):
        range_vec = torch.arange(seq_len, device=device, dtype=torch.long)
        rel_pos_matrix = range_vec[:, None] - range_vec[None, :]
        return rel_pos_matrix + seq_len - 1
    
    old_method = self.compute_relative_positions
    self.compute_relative_positions = compute_rel_pos_fixed
    try:
        result = original_forward(self, x, mask)
    finally:
        self.compute_relative_positions = old_method
    
    return result

attention.RelativePositionalEncodeMultiHeadSelfAttention.forward = forward_with_device_fix
```

### Case 3: CUDA error still occurs

**Problem:** Even with patch, CUDA assert happens

**Possible causes:**
1. **Teacher checkpoint issue** - wrong device or state
2. **Input data issue** - shapes don't match teacher expectations
3. **PyTorch version incompatibility** - 2.12 might need different fix

**Debug steps:**

1. Check teacher loads without error:
```python
from models.teacher import TeacherModel
teacher = TeacherModel(...)
print("Teacher loaded OK")
```

2. Check input shapes:
```python
x = torch.randn(2, 2, 51, 55, 1)  # (B, C, T, V, M)
print(f"Input shape: {x.shape}")
# Should match teacher's expected shape
```

3. Run with stack trace:
```bash
python -u debug_stage1_minimal.py 2>&1 | tail -50
```

---

## What Each Debug Script Does

### debug_inspect_patch.py
- Imports attention module
- Shows method before/after patch
- Compares source code
- Tests method behavior on GPU

**Fastest way to check if patch works.**

### debug_attention.py
- Tests original method (shows device mismatch)
- Tests patched method (should work)
- Full forward pass test

**Most comprehensive attention testing.**

### debug_stage1_minimal.py
- Loads actual checkpoint
- Tests real input shapes
- Minimal version of run_stage1.py

**Tests the actual training scenario.**

---

## If Nothing Works

Try the nuclear option - rewrite the method inline:

In `run_stage1.py`, right after adding sstan to sys.path:

```python
import torch
from sstan.models.transformers.modules.attention import (
    RelativePositionalEncodeMultiHeadSelfAttention,
)

# Completely replace the problematic method
def safe_compute_relative_positions(self, seq_len):
    """Always create indices on correct device."""
    device = self.relative_position_bias_table.device
    range_vec = torch.arange(seq_len, device=device, dtype=torch.long)
    rel_pos_matrix = range_vec[:, None] - range_vec[None, :]
    return rel_pos_matrix + seq_len - 1

# Force it
RelativePositionalEncodeMultiHeadSelfAttention.compute_relative_positions = safe_compute_relative_positions
print("✓ Forcefully patched compute_relative_positions")
```

Then run: `python run_stage1.py`

---

## Debug Output Interpretation

| Output | Meaning | Action |
|--------|---------|--------|
| `✓ Patch applied` | Patch function returned True | Continue testing |
| `✗ Patch failed` | Patch function returned False | Check sstan path |
| `Same object as original? False` | Method was replaced | Good sign |
| `✓ Devices match!` | indices.device == table.device | Patch working |
| `✗ Device mismatch` | indices on CPU, table on CUDA | Patch not working |
| `✓ Forward pass succeeded!` | No CUDA errors | Issue fixed! |
| `device-side assert` | CUDA error still occurs | Patch didn't work |

---

## Next Steps

After fixing the attention patch, you can:

1. Run minimal test:
   ```bash
   python debug_stage1_minimal.py
   ```

2. If that works, run full training:
   ```bash
   python run_stage1.py
   ```

3. If still fails, collect full error log:
   ```bash
   python run_stage1.py 2>&1 | tee stage1_error.log
   ```
   And share the log file.
