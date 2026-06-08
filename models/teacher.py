"""
Teacher model wrapper — SpatialTemporalTransformerWithClassToken (PostNorm).

Default from README / default.yaml:
    model=postnorm_transformer
    → model_name: SpatialTemporalTransformerWithClassToken
    → norm_type:  batchnorm   (BatchNorm2d, NOT LayerNorm)
    → block type: B2TPostNormSpatialTemporalTransformerBlock
      (PostNorm: norm AFTER residual; B2T: extra skip in FFN sub-block)

Attention shapes per block  (same as prenorm, same attention module):
  Spatial  (MultiHeadSelfAttention):                 (BM, T+1, H, V,   V  )
  Temporal (RelativePositionalEncodeMultiHeadSelfAttention): (BM, V,   H, T+1, T+1)
"""

import sys
import os
import torch
import torch.nn as nn
from typing import List, Dict

_TEACHER_SRC = os.path.join(
    os.path.dirname(__file__), "..", "..",
    "skeleton-slr-transformer-main (1)",
    "skeleton-slr-transformer-main", "src",
)
if _TEACHER_SRC not in sys.path:
    sys.path.insert(0, _TEACHER_SRC)

def _load_lightning_state_dict(ckpt: dict) -> dict:
    """
    Extract model weights from a PyTorch Lightning checkpoint.

    Lightning wraps the model as `self.model`, so all keys are prefixed
    with "model." in the saved state_dict.  This function strips that
    prefix so the weights can be loaded directly into the raw nn.Module.

    Supports:
      - Lightning .ckpt  →  ckpt["state_dict"]  with "model." prefix
      - plain torch .pth →  ckpt["model_state_dict"] or ckpt["state_dict"]
      - bare state dict  →  ckpt itself
    """
    if "state_dict" in ckpt:
        raw = ckpt["state_dict"]
        # Lightning prefix: strip "model."
        if any(k.startswith("model.") for k in raw):
            return {k[len("model."):]: v for k, v in raw.items()
                    if k.startswith("model.")}
        return raw
    if "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    # bare dict
    return ckpt


from sstan.models.transformers.postnorm_transformer import (
    SpatialTemporalTransformerWithClassToken,
)
from sstan.models.transformers.modules.attention import (
    MultiHeadSelfAttention,
    RelativePositionalEncodeMultiHeadSelfAttention,
)


class TeacherModel(nn.Module):
    """
    Frozen PostNorm teacher with forward hooks that expose per-block
    attention matrices and hidden states for MOHAWK distillation.

    Usage
    -----
        teacher = TeacherModel(
            checkpoint_path="checkpoints/teacher_best.pth",
            num_classes=100,
        )
        teacher.eval()

        out = teacher(skeleton_data, return_attn=True, return_hidden_states=True)
        out["logits"]                   # (B, num_classes)
        out["temporal_attn_matrices"]   # list[n_blocks] of (BM, V, H, T+1, T+1)
        out["hidden_states"]            # list[n_blocks] of (BM, T+1, V, D)
                                        # hidden_states[l] = INPUT to teacher's block l
                                        #                  = student_input for block l in MOHAWK
    """

    def __init__(
        self,
        checkpoint_path: str,
        num_classes: int = 100,
        in_channels: int = 2,
        seq_len: int = 50,
        n_joints: int = 55,
        embedding_dim: int = 128,
        n_blocks: int = 10,
        head_dim: int = 64,
        n_heads: int = 8,
        norm_type: str = "batchnorm",     # default from README/config
        ffn_expand_ratio: float = 4.0,
        ffn_dropout_ratio: float = 0.25,
        max_stochastic_depth_rate: float = 0.25,
        use_bias: bool = False,
        device: str = "cpu",
    ):
        super().__init__()

        self.model = SpatialTemporalTransformerWithClassToken(
            in_channels=in_channels,
            num_classes=num_classes,
            seq_len=seq_len,
            n_joints=n_joints,
            embedding_dim=embedding_dim,
            n_blocks=n_blocks,
            head_dim=head_dim,
            n_heads=n_heads,
            norm_type=norm_type,
            ffn_expand_ratio=ffn_expand_ratio,
            ffn_dropout_ratio=ffn_dropout_ratio,
            max_stochastic_depth_rate=max_stochastic_depth_rate,
            use_bias=use_bias,
        )

        ckpt = torch.load(checkpoint_path, map_location=device)
        state = _load_lightning_state_dict(ckpt)
        self.model.load_state_dict(state, strict=True)

        self.model.eval()
        self.model.requires_grad_(False)

        self._temporal_attn:  List[torch.Tensor] = []
        self._block_inputs:   List[torch.Tensor] = []  # INPUT  to block l  (pre-hook)
        self._block_outputs:  List[torch.Tensor] = []  # OUTPUT of block l  (post-hook)
        self._hooks: List = []
        self._register_hooks()

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def _register_hooks(self):
        for block in self.model.blocks:
            # ── Pre-hook: capture INPUT to each block ─────────────────
            # teacher_hidden_states[l] = input to block l
            #   = embedding output for l=0
            #   = output of block l-1 for l>0
            # This is exactly what MOHAWK uses as student_input for block l.
            def make_pre_hook():
                def hook(mod, inp):
                    # PyTorch pre-hook: inp luôn là tuple các positional args.
                    # Block được gọi là block(x, mask=None) → inp = (x,) hoặc (x, mask)
                    tensor_in = inp[0] if isinstance(inp, tuple) else inp
                    if not isinstance(tensor_in, torch.Tensor):
                        return
                    self._block_inputs.append(tensor_in.detach())
                return hook

            # ── Post-hook: capture OUTPUT of each block ───────────────
            # block_outputs[l] = output of block l
            #                  = TARGET for student block l in Stage 2
            #
            # Teacher block (B2TPostNormSpatialTemporalTransformerBlock)
            # trả về plain Tensor. Nhưng hook được viết robust để xử lý
            # mọi kiểu trả về:
            #   Tensor          → dùng trực tiếp
            #   tuple / list    → lấy phần tử đầu (hidden_states theo convention)
            #   dict            → lấy key "hidden_states"
            def make_post_hook():
                def hook(mod, inp, out):
                    if isinstance(out, torch.Tensor):
                        tensor = out
                    elif isinstance(out, (tuple, list)):
                        # First element is always hidden_states by convention
                        tensor = out[0]
                        if not isinstance(tensor, torch.Tensor):
                            return          # không nhận dạng được, bỏ qua
                    elif isinstance(out, dict):
                        tensor = out.get("hidden_states", None)
                        if tensor is None:
                            return
                    else:
                        return
                    self._block_outputs.append(tensor.detach())
                return hook

            # ── Forward hook: temporal attention matrix ────────────────
            def make_tm_hook():
                def hook(mod, inp, out):
                    _z, attn = out          # (BM, V, H, T+1, T+1)
                    self._temporal_attn.append(attn.detach())
                return hook

            h1 = block.register_forward_pre_hook(make_pre_hook())
            h2 = block.register_forward_hook(make_post_hook())
            h3 = block.multihead_self_attention2.register_forward_hook(make_tm_hook())
            self._hooks.extend([h1, h2, h3])

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        skeleton_data: torch.Tensor,     # (B, C, T, V, M)
        return_attn: bool = False,
        return_hidden_states: bool = False,
    ) -> Dict:
        self._temporal_attn.clear()
        self._block_inputs.clear()
        self._block_outputs.clear()

        with torch.no_grad():
            logits = self.model(skeleton_data)

        result: Dict = {"logits": logits}
        if return_attn:
            # list[n_blocks] of (BM, V, H, T+1, T+1)
            result["temporal_attn_matrices"] = list(self._temporal_attn)
        if return_hidden_states:
            # hidden_states[l]  = INPUT  to teacher block l   → student_input  for block l
            # block_outputs[l]  = OUTPUT of teacher block l   → target for student block l
            result["hidden_states"]  = list(self._block_inputs)
            result["block_outputs"]  = list(self._block_outputs)
        return result
