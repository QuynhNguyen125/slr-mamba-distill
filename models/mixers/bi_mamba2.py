"""
Bidirectional Discrete Mamba-2 Mixer using mamba_ssm kernels.

Key design decisions
--------------------
1. Uses `mamba_chunk_scan_combined` from mamba_ssm (Triton CUDA kernel).
2. Bidirectional — two passes:
     forward  : standard left-to-right causal SSM scan
     backward : reverse the sequence, run causal SSM, reverse output back
     output   : y = y_forward + y_backward
3. No Δ (discrete-time): A_log acts as per-token per-head time step;
   A is fixed at -1 for all v_heads (same as phi-mamba).
4. Symmetric Conv1d with padding='same' — no causal masking.
5. n_qk_heads = n_v_heads = n_heads so transfer-matrix shape (B, H, L, L)
   maps 1-to-1 with the teacher's attention matrices.

Transfer matrix for Stage-1 distillation
-----------------------------------------
  Non-causal bidirectional decay (NO torch.tril, NO masked_fill):

    powers[b, h, i, j] = exp( −|cumsum(A)[b,h,i] − cumsum(A)[b,h,j]| )
                        ∈ (0, 1]   for all (i, j) pairs

  This is:
  • Symmetric    : powers[i,j] = powers[j,i]
  • Full matrix  : valid for every (i, j), no masking
  • Stable decay : values always in (0,1]
  • At i=j       : powers = 1 (identity, no decay)

  T[b, h, i, j]  = (C[b,i,h,:] · B[b,j,h,:]) * powers[b,h,i,j]
  T final         = T + D * I   (D skip added once on diagonal)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# ── mamba_ssm kernels ──────────────────────────────────────────────────
# Triton SSD kernel yêu cầu GPU compute capability >= 8.0 (Ampere+).
# RTX Titan = cc 7.5 → dùng pure-PyTorch fallback thay thế.
def _check_gpu_compatible() -> bool:
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        cc_major, _ = torch.cuda.get_device_capability(0)
        if cc_major < 8:
            print(f"[BiMamba2] GPU compute capability {cc_major}.x < 8.0 "
                  f"— dùng PyTorch fallback (Triton không hỗ trợ cc7.x)")
            return False
        return True
    except Exception:
        return False

_GPU_COMPATIBLE = _check_gpu_compatible()

try:
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
    _HAS_MAMBA_SSM = _GPU_COMPATIBLE   # chỉ dùng kernel nếu GPU tương thích
except ImportError:
    _HAS_MAMBA_SSM = False


# ── Non-causal segment sum (bidirectional, no tril, no masked_fill) ────

def _segsum_noncausal(x: torch.Tensor) -> torch.Tensor:
    """
    Non-causal bidirectional decay matrix — replaces the causal segsum.

    NO torch.tril, NO masked_fill.

    For each pair (i, j):
        result[..., i, j] = −|cumsum(x)[..., i] − cumsum(x)[..., j]|

    Properties:
      • Full matrix  : valid for ALL (i, j), not just i >= j
      • Symmetric    : result[i,j] == result[j,i]
      • Non-positive : result[i,j] <= 0  →  exp(result) ∈ (0, 1]
      • At diagonal  : result[i,i] == 0  →  exp = 1  (no decay)
      • Decays with "distance" in cumsum(A) space (larger gap → stronger decay)
    """
    cumsum = torch.cumsum(x, dim=-1)        # (..., L)
    cs_i = cumsum.unsqueeze(-1)             # (..., L, 1)  broadcast over j
    cs_j = cumsum.unsqueeze(-2)             # (..., 1, L)  broadcast over i
    return -(cs_i - cs_j).abs()             # (..., L, L)  all values ≤ 0


def _materialize_transfer_matrix(
    A_log: torch.Tensor,   # (B, L, H)
    B: torch.Tensor,       # (B, L, H, S)
    C: torch.Tensor,       # (B, L, H, S)
    D: torch.Tensor,       # (H,)
) -> torch.Tensor:
    """
    Compute the non-causal bidirectional transfer matrix.

    Uses _segsum_noncausal — no torch.tril, no masked_fill, no flip tricks.
    Single pass, full (L×L) matrix, both directions treated equally.

        powers[b, h, i, j] = exp(−|cumsum(A)[b,h,i] − cumsum(A)[b,h,j]|)
        T[b, h, i, j]      = (C[b,i,h,:] · B[b,j,h,:]) * powers[b,h,i,j]
        T                 += D[h] on diagonal

    Returns: (B, H, L, L)
    """
    batch, L, H, S = B.shape

    # Validate input shapes
    assert A_log.shape == (batch, L, H), f"A_log shape mismatch: {A_log.shape} vs ({batch}, {L}, {H})"
    assert C.shape == (batch, L, H, S), f"C shape mismatch: {C.shape} vs ({batch}, {L}, {H}, {S})"

    # A: (B, H, L)  — negative values ensure decay
    A_neg = rearrange(-F.softplus(A_log), "b l h -> b h l")

    # Non-causal decay: (B, H, L, L) — full symmetric matrix, no masking
    powers = torch.exp(_segsum_noncausal(A_neg))

    # T[b,h,s,l] = sum_n  C[b,l,h,n] * B[b,s,h,n] * powers[b,h,l,s]
    T_raw = torch.einsum("blhn,bshn,bhls->bhsl", C, B, powers)
    T = rearrange(T_raw, "b h s l -> b h l s")                # (B, H, L, L)

    # Ensure T is contiguous before diagonal operations
    if not T.is_contiguous():
        T = T.contiguous()

    # D skip connection on diagonal (added once, vectorized)
    # T.diagonal(dim1=-2, dim2=-1) → view (B, H, L) — không copy memory
    # D.view(1, H, 1) broadcast qua B và L
    if D is not None:
        assert D.shape[0] == H, f"D shape mismatch: {D.shape[0]} vs {H}"
        T.diagonal(dim1=-2, dim2=-1).add_(D.view(1, H, 1))

    return T


# ── Pure-PyTorch fallback scan (used when mamba_ssm is unavailable) ────

def _ssm_scan_pytorch(
    x: torch.Tensor,        # (B, L, H, P)
    A_bar: torch.Tensor,    # (B, L, H)  — decay ∈ (0,1)
    B: torch.Tensor,        # (B, L, H, S)
    C: torch.Tensor,        # (B, L, H, S)
    reverse: bool = False,
) -> torch.Tensor:
    """Sequential SSM scan (fallback; slow for long sequences)."""
    if reverse:
        x, A_bar, B, C = x.flip(1), A_bar.flip(1), B.flip(1), C.flip(1)

    B_sz, L, H, P = x.shape
    S = B.shape[-1]
    h = x.new_zeros(B_sz, H, P, S)
    ys = []
    for t in range(L):
        a = A_bar[:, t, :, None, None]           # (B,H,1,1)
        b = B[:, t, :, None, :]                  # (B,H,1,S)
        xt = x[:, t, :, :, None]                 # (B,H,P,1)
        h = a * h + xt * b                       # (B,H,P,S)
        c = C[:, t, :, None, :]                  # (B,H,1,S)
        ys.append((h * c).sum(-1))               # (B,H,P)
    y = torch.stack(ys, dim=1)                   # (B,L,H,P)
    if reverse:
        y = y.flip(1)
    return y


# ── Main mixer ─────────────────────────────────────────────────────────

class BiMamba2Mixer(nn.Module):
    """
    Bidirectional Discrete Mamba-2 Mixer.

    Parameters
    ----------
    d_model    : model dimension
    d_state    : SSM state (N) — controls expressiveness; independent of teacher
    n_heads    : MUST equal teacher attention heads (n_v_heads = n_qk_heads = n_heads)
    d_conv     : depthwise conv kernel; use odd number for exact 'same' padding
    expand     : d_inner = expand * d_model
    chunk_size : chunk length for mamba_ssm kernel.
                 Sequence is padded to the next multiple of chunk_size automatically.
                 For sign language: spatial L=55 joints, temporal L=51 frames —
                 chunk_size=16 works for both (55→64, 51→64).
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        n_heads: int = 8,          # ← must match teacher attention heads
        d_conv: int = 3,
        expand: int = 1,
        chunk_size: int = 32,
        bias: bool = False,
        conv_bias: bool = True,
        dropout: float = 0.0,
        layer_idx: int = None,
        **kwargs,
    ):
        super().__init__()
        self.d_model    = d_model
        self.d_state    = d_state
        self.n_heads    = n_heads   # = n_v_heads = n_qk_heads
        self.d_inner    = expand * d_model
        assert self.d_inner % n_heads == 0, "d_inner must be divisible by n_heads"
        self.headdim    = self.d_inner // n_heads
        self.d_conv     = d_conv
        self.chunk_size = chunk_size
        self.layer_idx  = layer_idx

        # ── In-projection: → [xBC (for conv), z (gate), A_log] ───────
        # xBC : d_inner + 2 * n_heads * d_state
        # z   : d_inner
        # A_log: n_heads  (per-token, per-head)
        conv_dim = self.d_inner + 2 * n_heads * d_state
        proj_dim = conv_dim + self.d_inner + n_heads
        self.in_proj = nn.Linear(d_model, proj_dim, bias=bias)

        # Initialize weights with proper scaling
        with torch.no_grad():
            # Standard initialization for xBC and z parts
            nn.init.normal_(self.in_proj.weight[:conv_dim + self.d_inner], std=1.0 / (d_model ** 0.5))
            # A_log should map to values where softplus(A_log) ∈ (0.1, 1)
            # This typically means A_log ∈ (-1, 0)
            nn.init.uniform_(self.in_proj.weight[-n_heads:], a=-1.0, b=0.0)
            if bias:
                nn.init.zeros_(self.in_proj.bias)

        # ── Symmetric depthwise Conv1d (NOT causal) ──────────────────
        # padding='same' ensures output length == input length
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            kernel_size=d_conv,
            padding="same",          # symmetric padding — no causal mask
            groups=conv_dim,
            bias=conv_bias,
        )

        # ── D skip connection (per v_head) ────────────────────────────
        # Khởi tạo ngẫu nhiên trong [0, 1] → học bởi AdamW trong quá trình training.
        # Không dùng ones() vì tất cả head bắt đầu giống nhau → mất symmetry-breaking.
        self.D = nn.Parameter(torch.empty(n_heads).uniform_(0.0, 1.0))

        # ── z bias (from phi-mamba) ───────────────────────────────────
        self.z_bias = nn.Parameter(torch.zeros(self.d_inner)) if not bias else 0

        # ── Out-projection ───────────────────────────────────────────
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)
        self.dropout  = nn.Dropout(dropout)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        u: torch.Tensor,                      # (B, L, d_model)
        return_transfer_matrix: bool = False,
        **kwargs,
    ) -> dict:
        batch, L, _ = u.shape
        conv_dim = self.d_inner + 2 * self.n_heads * self.d_state

        # Input sanitization (replace NaN/Inf để tránh crash training)
        if torch.isnan(u).any() or torch.isinf(u).any():
            u = torch.nan_to_num(u, nan=0.0, posinf=1.0, neginf=-1.0)

        # Pad to nearest multiple of chunk_size for mamba_ssm kernel
        padded_L = ((L - 1) // self.chunk_size + 1) * self.chunk_size
        u_pad = F.pad(u, (0, 0, 0, padded_L - L))

        # ── Project ──────────────────────────────────────────────────
        xBCzA = self.in_proj(u_pad)                              # (B, padded_L, proj_dim)
        xBC   = xBCzA[..., :conv_dim]
        z     = xBCzA[..., conv_dim : conv_dim + self.d_inner]
        A_log = xBCzA[..., conv_dim + self.d_inner :]           # (B, padded_L, H)

        # Validate A_log range
        A_softplus = F.softplus(A_log)
        assert not torch.isnan(A_softplus).any(), f"softplus(A_log) contains NaN. A_log range: [{A_log.min():.4f}, {A_log.max():.4f}]"
        assert not torch.isinf(A_softplus).any(), f"softplus(A_log) contains Inf. A_log range: [{A_log.min():.4f}, {A_log.max():.4f}]"

        # ── Symmetric depthwise Conv1d ────────────────────────────────
        xBC = F.silu(
            self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)    # (B, padded_L, conv_dim)
        )

        # ── Fix 2: Padding leak — zero conv-bias contamination ────────
        # Conv1d bias adds a non-zero constant to ALL positions including
        # the zero-padded ones.  The backward scan then carries that bias
        # back into the real sequence.  Zero out every padded slot here,
        # before the values ever enter the SSM scan.
        if padded_L > L:
            seq_mask_2d = xBC.new_ones(batch, padded_L, 1)
            seq_mask_2d[:, L:] = 0.0
            xBC   = xBC   * seq_mask_2d          # (B, padded_L, conv_dim)
            z     = z     * seq_mask_2d          # (B, padded_L, d_inner)
            A_log = A_log * seq_mask_2d[..., :self.n_heads]  # (B, padded_L, H)

        # ── Split x / B / C ──────────────────────────────────────────
        x_flat, B_flat, C_flat = torch.split(
            xBC,
            [self.d_inner,
             self.n_heads * self.d_state,
             self.n_heads * self.d_state],
            dim=-1,
        )

        # Reshape to (B, padded_L, H, P/S)
        x      = x_flat.view(batch, padded_L, self.n_heads, self.headdim)
        B_ssm  = B_flat.view(batch, padded_L, self.n_heads, self.d_state)
        C_ssm  = C_flat.view(batch, padded_L, self.n_heads, self.d_state)

        # Normalize values: x / softplus(A_log)  [SSD parameterization]
        # Add epsilon for numerical stability
        x_norm = x / (F.softplus(A_log).unsqueeze(-1) + 1e-6)

        # ── Fix 2 (cont.): mask x_norm before scan ────────────────────
        # Even after zeroing xBC above, apply an explicit 4-D mask to
        # x_norm (and B_ssm / C_ssm) so no residual leaks through the
        # normalization step or the scan kernel at padded positions.
        if padded_L > L:
            seq_mask_4d = x_norm.new_ones(batch, padded_L, 1, 1)
            seq_mask_4d[:, L:] = 0.0
            x_norm = x_norm * seq_mask_4d        # (B, padded_L, H, P)
            B_ssm  = B_ssm  * seq_mask_4d        # (B, padded_L, H, S)
            C_ssm  = C_ssm  * seq_mask_4d        # (B, padded_L, H, S)

        # ── Bidirectional SSM ─────────────────────────────────────────
        if _HAS_MAMBA_SSM:
            try:
                y_fwd, y_bwd = self._scan_mamba_ssm(x_norm, A_log, B_ssm, C_ssm, padded_L)
            except Exception:
                # Runtime fallback nếu Triton kernel vẫn lỗi
                y_fwd, y_bwd = self._scan_pytorch(x_norm, A_log, B_ssm, C_ssm)
        else:
            y_fwd, y_bwd = self._scan_pytorch(x_norm, A_log, B_ssm, C_ssm)

        # ── Fix 1: Diagonal doubling ──────────────────────────────────
        # Both y_fwd[i] and y_bwd[i] include the "self" contribution
        #   C[i] · B[i] * x_norm[i]
        # (the diagonal of the SSM transfer matrix, where source==query).
        # When we sum the two passes this term is counted twice.
        # Subtract it once to restore the correct bidirectional output.
        #
        #   diagonal_term[b,l,h,p] = (Σ_s C[b,l,h,s]*B[b,l,h,s]) * x_norm[b,l,h,p]
        #                           = CB_dot[b,l,h,1]            * x_norm[b,l,h,p]
        CB_dot = (C_ssm * B_ssm).sum(-1, keepdim=True)          # (B, padded_L, H, 1)
        y = y_fwd + y_bwd - CB_dot * x_norm                     # (B, padded_L, H, P)

        # D skip (added once, not doubled)
        Du = torch.einsum("h,blhp->blhp", self.D, x_norm)
        y  = y + Du

        # Reshape, gate, project
        y   = y.reshape(batch, padded_L, self.d_inner)
        out = self.out_proj(
            self.dropout(y * F.silu(z + self.z_bias))
        )                                                        # (B, padded_L, d_model)

        result = {"hidden_states": out[:, :L, :]}

        if return_transfer_matrix:
            result["transfer_matrix"] = _materialize_transfer_matrix(
                A_log[:, :L, :],
                B_ssm[:, :L, :, :],
                C_ssm[:, :L, :, :],
                self.D,
            )

        return result

    # ------------------------------------------------------------------
    # Internal scan implementations
    # ------------------------------------------------------------------

    def _scan_mamba_ssm(
        self,
        x_norm: torch.Tensor,   # (B, padded_L, H, P)  — already masked
        A_log:  torch.Tensor,   # (B, padded_L, H)
        B_ssm:  torch.Tensor,   # (B, padded_L, H, S)  — already masked
        C_ssm:  torch.Tensor,   # (B, padded_L, H, S)  — already masked
        padded_L: int,
    ) -> tuple:
        """
        Two-pass bidirectional scan using mamba_ssm's causal kernel.

        Returns (y_fwd, y_bwd) separately so the caller can apply the
        diagonal-doubling correction before summing.

        Forward  : standard order
        Backward : flip sequence → causal scan → flip output
        """
        A_fixed = -torch.ones(self.n_heads, device=x_norm.device)

        # ── Forward pass ─────────────────────────────────────────────
        y_fwd = mamba_chunk_scan_combined(
            x   = x_norm,
            dt  = A_log,
            A   = A_fixed,
            B   = B_ssm,
            C   = C_ssm,
            chunk_size = self.chunk_size,
            dt_softplus = True,
        )                                                        # (B, padded_L, H, P)

        # ── Backward pass (reverse → scan → reverse) ─────────────────
        y_bwd = mamba_chunk_scan_combined(
            x   = x_norm.flip(1),
            dt  = A_log.flip(1),
            A   = A_fixed,
            B   = B_ssm.flip(1),
            C   = C_ssm.flip(1),
            chunk_size = self.chunk_size,
            dt_softplus = True,
        ).flip(1)                                                # (B, padded_L, H, P)

        return y_fwd, y_bwd

    def _scan_pytorch(
        self,
        x_norm: torch.Tensor,
        A_log:  torch.Tensor,
        B_ssm:  torch.Tensor,
        C_ssm:  torch.Tensor,
    ) -> tuple:
        """Pure-PyTorch fallback — returns (y_fwd, y_bwd) separately."""
        A_bar = torch.exp(-F.softplus(A_log))                   # (B, L, H)
        y_fwd = _ssm_scan_pytorch(x_norm, A_bar, B_ssm, C_ssm, reverse=False)
        y_bwd = _ssm_scan_pytorch(x_norm, A_bar, B_ssm, C_ssm, reverse=True)
        return y_fwd, y_bwd

    @property
    def d_output(self):
        return self.d_model
