"""
Debug script to diagnose CUDA errors in BiMamba2Mixer.

Run with: CUDA_LAUNCH_BLOCKING=1 python debug_cuda_error.py
"""

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import sys
import torch
import torch.nn.functional as F
import numpy as np

# Add teacher path
_TEACHER_SRC = os.path.join(
    os.path.dirname(__file__), "..",
    "skeleton-slr-transformer-main (1)",
    "skeleton-slr-transformer-main", "src",
)
if _TEACHER_SRC not in sys.path:
    sys.path.insert(0, _TEACHER_SRC)

from models.student import HybridB2TPostNormBlock
from models.mixers.bi_mamba2 import BiMamba2Mixer


def test_mixer_shapes():
    """Test mixer with debug prints."""
    print("=" * 80)
    print("Testing BiMamba2Mixer shapes and values")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Simple test case
    batch_size = 4
    seq_len = 51  # temporal
    d_model = 256
    n_heads = 8
    d_state = 64

    mixer = BiMamba2Mixer(
        d_model=d_model,
        d_state=d_state,
        n_heads=n_heads,
        d_conv=3,
        chunk_size=32,
    ).to(device)

    # Input: (B, L, D)
    u = torch.randn(batch_size, seq_len, d_model, device=device)

    print(f"\nInput shape: {u.shape}")
    print(f"Input min/max: {u.min():.4f} / {u.max():.4f}")
    print(f"Input has NaN: {torch.isnan(u).any()}")
    print(f"Input has Inf: {torch.isinf(u).any()}")

    try:
        with torch.no_grad():
            # Manually step through forward to catch where error occurs
            xBCzA = mixer.in_proj(u)  # (B, L, proj_dim)
            print(f"\nAfter in_proj shape: {xBCzA.shape}")
            print(f"After in_proj min/max: {xBCzA.min():.4f} / {xBCzA.max():.4f}")
            print(f"After in_proj has NaN: {torch.isnan(xBCzA).any()}")

            conv_dim = mixer.d_inner + 2 * mixer.n_heads * mixer.d_state
            xBC = xBCzA[..., :conv_dim]
            z = xBCzA[..., conv_dim : conv_dim + mixer.d_inner]
            A_log = xBCzA[..., conv_dim + mixer.d_inner :]

            print(f"\nA_log shape: {A_log.shape}")
            print(f"A_log min/max: {A_log.min():.4f} / {A_log.max():.4f}")
            print(f"A_log has NaN: {torch.isnan(A_log).any()}")

            # Check softplus
            softplus_A = F.softplus(A_log)
            print(f"\nsoftplus(A_log) min/max: {softplus_A.min():.6f} / {softplus_A.max():.6f}")
            print(f"softplus(A_log) has NaN: {torch.isnan(softplus_A).any()}")
            print(f"softplus(A_log) < 1e-6: {(softplus_A < 1e-6).sum().item()} values")

            # Conv1d
            xBC = F.silu(mixer.conv1d(xBC.transpose(1, 2)).transpose(1, 2))
            print(f"\nAfter conv1d shape: {xBC.shape}")
            print(f"After conv1d min/max: {xBC.min():.4f} / {xBC.max():.4f}")
            print(f"After conv1d has NaN: {torch.isnan(xBC).any()}")

            # Split
            x_flat, B_flat, C_flat = torch.split(
                xBC,
                [mixer.d_inner,
                 mixer.n_heads * mixer.d_state,
                 mixer.n_heads * mixer.d_state],
                dim=-1,
            )

            x = x_flat.view(batch_size, seq_len, mixer.n_heads, mixer.headdim)
            B_ssm = B_flat.view(batch_size, seq_len, mixer.n_heads, mixer.d_state)
            C_ssm = C_flat.view(batch_size, seq_len, mixer.n_heads, mixer.d_state)

            print(f"\nx shape: {x.shape}")
            print(f"B_ssm shape: {B_ssm.shape}")
            print(f"C_ssm shape: {C_ssm.shape}")

            # Normalize
            x_norm = x / F.softplus(A_log).unsqueeze(-1)
            print(f"\nx_norm min/max: {x_norm.min():.4f} / {x_norm.max():.4f}")
            print(f"x_norm has NaN: {torch.isnan(x_norm).any()}")
            print(f"x_norm has Inf: {torch.isinf(x_norm).any()}")

            # Check A_bar
            A_bar = torch.exp(-F.softplus(A_log))  # (B, L, H)
            print(f"\nA_bar min/max: {A_bar.min():.6f} / {A_bar.max():.6f}")
            print(f"A_bar has NaN: {torch.isnan(A_bar).any()}")
            print(f"A_bar has Inf: {torch.isinf(A_bar).any()}")

            # Try forward
            out = mixer(u)
            print(f"\nForward output shape: {out['hidden_states'].shape}")
            print(f"Forward output has NaN: {torch.isnan(out['hidden_states']).any()}")
            print(f"Forward output has Inf: {torch.isinf(out['hidden_states']).any()}")

            print("\n✓ Forward pass succeeded!")

    except Exception as e:
        print(f"\n✗ Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_mixer_shapes()
