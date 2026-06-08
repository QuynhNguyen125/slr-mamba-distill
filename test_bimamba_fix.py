"""
Minimal test for BiMamba2Mixer to verify the CUDA error fix.

Run with: python test_bimamba_fix.py
"""

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import torch
import torch.nn as nn

# Add model path
import sys
sys.path.insert(0, os.path.dirname(__file__))

from models.mixers.bi_mamba2 import BiMamba2Mixer


def test_bimamba_initialization():
    """Test that BiMamba2Mixer initializes correctly."""
    print("Testing BiMamba2Mixer initialization...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    mixer = BiMamba2Mixer(
        d_model=256,
        d_state=64,
        n_heads=8,
        d_conv=3,
        chunk_size=16,
    ).to(device)

    # Check initialization of in_proj weights for A_log
    with torch.no_grad():
        a_log_weight = mixer.in_proj.weight[-8:]  # Last 8 rows for A_log
        print(f"\nA_log weight initialization:")
        print(f"  Min: {a_log_weight.min():.4f}, Max: {a_log_weight.max():.4f}")
        assert a_log_weight.min() >= -1.0, "A_log weights should be >= -1.0"
        assert a_log_weight.max() <= 0.0, "A_log weights should be <= 0.0"
        print("  ✓ A_log weights are in correct range [-1.0, 0.0]")


def test_forward_pass():
    """Test forward pass without errors."""
    print("\nTesting forward pass...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    mixer = BiMamba2Mixer(
        d_model=256,
        d_state=64,
        n_heads=8,
        d_conv=3,
        chunk_size=16,
    ).to(device)
    mixer.eval()

    batch_size = 4
    seq_len = 51  # Temporal sequence length for skeleton data
    d_model = 256

    # Create input
    x = torch.randn(batch_size, seq_len, d_model, device=device)

    print(f"Input shape: {x.shape}")
    print(f"Input stats: min={x.min():.4f}, max={x.max():.4f}")

    try:
        with torch.no_grad():
            out = mixer(x, return_transfer_matrix=False)
            hidden = out["hidden_states"]

            print(f"\nOutput shape: {hidden.shape}")
            print(f"Output stats: min={hidden.min():.4f}, max={hidden.max():.4f}")

            # Check for NaN/Inf
            assert not torch.isnan(hidden).any(), "Output contains NaN"
            assert not torch.isinf(hidden).any(), "Output contains Inf"

            print("✓ Forward pass succeeded without NaN/Inf")

    except AssertionError as e:
        print(f"✗ Assertion failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Error during forward pass: {type(e).__name__}: {e}")
        return False

    return True


def test_transfer_matrix():
    """Test transfer matrix computation."""
    print("\nTesting transfer matrix computation...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mixer = BiMamba2Mixer(
        d_model=256,
        d_state=64,
        n_heads=8,
        d_conv=3,
        chunk_size=16,
    ).to(device)
    mixer.eval()

    batch_size = 2
    seq_len = 51
    d_model = 256

    x = torch.randn(batch_size, seq_len, d_model, device=device)

    try:
        with torch.no_grad():
            out = mixer(x, return_transfer_matrix=True)

            transfer_matrix = out["transfer_matrix"]
            print(f"\nTransfer matrix shape: {transfer_matrix.shape}")
            print(f"Expected shape: ({batch_size}, {mixer.n_heads}, {seq_len}, {seq_len})")

            assert transfer_matrix.shape == (batch_size, mixer.n_heads, seq_len, seq_len), \
                f"Shape mismatch: {transfer_matrix.shape}"

            print(f"Transfer matrix stats: min={transfer_matrix.min():.4f}, max={transfer_matrix.max():.4f}")

            # Check for NaN/Inf
            assert not torch.isnan(transfer_matrix).any(), "Transfer matrix contains NaN"
            assert not torch.isinf(transfer_matrix).any(), "Transfer matrix contains Inf"

            print("✓ Transfer matrix computation succeeded")

    except Exception as e:
        print(f"✗ Error: {type(e).__name__}: {e}")
        return False

    return True


def test_backward_pass():
    """Test backward pass for training."""
    print("\nTesting backward pass...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mixer = BiMamba2Mixer(
        d_model=256,
        d_state=64,
        n_heads=8,
        d_conv=3,
        chunk_size=16,
    ).to(device)
    mixer.train()

    batch_size = 2
    seq_len = 51
    d_model = 256

    x = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)

    try:
        out = mixer(x)
        loss = out["hidden_states"].sum()
        loss.backward()

        print("✓ Backward pass succeeded")
        return True

    except Exception as e:
        print(f"✗ Error: {type(e).__name__}: {e}")
        return False


if __name__ == "__main__":
    print("=" * 80)
    print("BiMamba2Mixer CUDA Error Fix Tests")
    print("=" * 80)

    try:
        test_bimamba_initialization()

        passed = 0
        total = 3

        if test_forward_pass():
            passed += 1

        if test_transfer_matrix():
            passed += 1

        if test_backward_pass():
            passed += 1

        print("\n" + "=" * 80)
        print(f"Results: {passed}/{total} tests passed")
        print("=" * 80)

        if passed == total:
            print("\n✓ All tests passed! The CUDA error should be fixed.")
        else:
            print(f"\n✗ {total - passed} test(s) failed.")

    except Exception as e:
        print(f"\nFatal error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
