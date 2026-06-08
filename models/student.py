"""
BiMambaSLR — Student model for skeleton-based Sign Language Recognition.

Strategy: keep the teacher architecture exactly, replace ONLY the temporal
attention sub-block with BiMamba2Mixer.

Teacher block (B2TPostNormSpatialTemporalTransformerBlock):
  spatial MHA → residual → norm1   (PostNorm, BatchNorm2d)
  temporal RelPosBias MHA → residual → norm2
  FFN + residual + input → norm3   (B2T: extra skip from block input)

Student block (HybridB2TPostNormBlock):
  spatial MHA    [UNCHANGED]  → residual → norm1
  temporal Bi-Mamba2 [NEW]    → residual → norm2
  FFN + residual + input      [UNCHANGED] → norm3  (B2T preserved)

Everything else (input format, embedding, cls_token, spatial PE, pooling,
classifier, norm_type=batchnorm) is identical to the teacher.

Input: (B, C=2, T, V=55, M=1)  — same as SpatialTemporalTransformerWithClassToken

Distillation:
  Stage 1 — temporal BiMamba2 transfer matrix  ↔ teacher temporal attention (BM, V, H, T+1, T+1)
  Stage 2 — per-block hidden states aligned
  Stage 3 — full KL + CE
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── sstan imports ──────────────────────────────────────────────────────
_TEACHER_SRC = os.path.join(
    os.path.dirname(__file__), "..", "..",
    "skeleton-slr-transformer-main (1)",
    "skeleton-slr-transformer-main", "src",
)
if _TEACHER_SRC not in sys.path:
    sys.path.insert(0, _TEACHER_SRC)

from sstan.models.transformers.modules.attention import MultiHeadSelfAttention
from sstan.models.transformers.postnorm_transformer import FeedForwardNetwork

from .mixers.bi_mamba2 import BiMamba2Mixer
from .pos_encode import SinusoidalPositionalEncoding


def _apply_norm(norm_layer: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Apply norm_layer to x: (BM, T+1, V, D).
    LayerNorm: applied directly.
    BatchNorm2d: transpose to (BM, D, V, T+1), apply, transpose back.
    """
    if isinstance(norm_layer, nn.LayerNorm):
        return norm_layer(x)
    else:
        # BatchNorm2d expects (N, C, H, W)
        return norm_layer(x.transpose(-1, 1)).transpose(-1, 1)


# ──────────────────────────────────────────────────────────────────────
# Hybrid block: PostNorm + B2T, spatial MHA unchanged, temporal → BiMamba2
# ──────────────────────────────────────────────────────────────────────

class HybridB2TPostNormBlock(nn.Module):
    """
    Mirrors B2TPostNormSpatialTemporalTransformerBlock exactly except
    `multihead_self_attention2` (temporal) is replaced by BiMamba2Mixer.

    PostNorm order (norm AFTER residual):
        z, _ = spatial_attn(input)
        x = norm1(input + z)

        x = x.transpose(-2,-3)
        z  = temporal_mamba(x)
        x = norm2(x + z)
        x = x.transpose(-2,-3)

        x = norm3(x + ffn(x) + input)   ← B2T extra skip from block input
    """

    def __init__(
        self,
        embedding_dim: int,
        head_dim: int,
        n_heads: int,
        d_state: int,
        d_conv: int,
        chunk_size: int,
        seq_len: int,       # T+1
        n_joints: int,      # V
        ffn_expand_ratio: float = 4.0,
        ffn_dropout_ratio: float = 0.25,
        norm_type: str = "batchnorm",
        stochastic_depth_rate: float = 0.0,
        use_bias: bool = False,
        layer_idx: int = None,
    ):
        super().__init__()
        import torchvision

        norm_cls = nn.BatchNorm2d if norm_type == "batchnorm" else nn.LayerNorm

        # ── Spatial MHA (identical to teacher) ───────────────────────
        self.multihead_self_attention1 = MultiHeadSelfAttention(
            input_dim=embedding_dim,
            head_dim=head_dim,
            n_heads=n_heads,
            bias=use_bias,
        )
        self.norm_layer1 = norm_cls(embedding_dim)

        # ── Temporal BiMamba2 (replaces temporal MHA) ─────────────────
        self.temporal_mamba = BiMamba2Mixer(
            d_model=embedding_dim,
            d_state=d_state,
            n_heads=n_heads,       # matches teacher n_heads for 1-to-1 alignment
            d_conv=d_conv,
            chunk_size=chunk_size,
            layer_idx=layer_idx,
        )
        self.norm_layer2 = norm_cls(embedding_dim)

        # ── FFN (identical to teacher) ────────────────────────────────
        self.feed_forward_network = FeedForwardNetwork(
            in_channels=embedding_dim,
            expand_ratio=ffn_expand_ratio,
            dropout_ratio=ffn_dropout_ratio,
            bias=use_bias,
        )
        self.norm_layer3 = norm_cls(embedding_dim)

        self.stochastic_depth = (
            torchvision.ops.StochasticDepth(stochastic_depth_rate, mode="row")
            if stochastic_depth_rate > 0 else nn.Identity()
        )

    def forward(
        self,
        hidden_states: torch.Tensor,            # (BM, T+1, V, D)
        return_transfer_matrix: bool = False,
        run_mlp_component: bool = True,
    ) -> dict:
        BM, T1, V, D = hidden_states.shape
        outputs = {}

        # ── 1. Spatial MHA — PostNorm (unchanged from teacher) ────────
        z, _sp_attn = self.multihead_self_attention1(hidden_states)
        x = hidden_states + self.stochastic_depth(z)
        x = _apply_norm(self.norm_layer1, x)

        # ── 2. Temporal BiMamba2 — PostNorm ───────────────────────────
        x = x.transpose(-2, -3).contiguous()    # (BM, V, T+1, D)
        _x_flat = x.reshape(BM * V, T1, D)

        tm_out = self.temporal_mamba(
            _x_flat,
            return_transfer_matrix=return_transfer_matrix,
        )
        z_tm = tm_out["hidden_states"].reshape(BM, V, T1, D)
        x = x + self.stochastic_depth(z_tm)
        x = _apply_norm(self.norm_layer2, x)
        x = x.transpose(-2, -3).contiguous()    # (BM, T+1, V, D)

        if return_transfer_matrix:
            H = tm_out["transfer_matrix"].shape[1]
            # "transfer_matrix" key — shape (BM, V, H, T+1, T+1)
            outputs["transfer_matrix"] = (
                tm_out["transfer_matrix"].view(BM, V, H, T1, T1)
            )

        # ── 3. FFN — B2T PostNorm (unchanged from teacher) ───────────
        if run_mlp_component:
            # B2T: extra skip from block input (hidden_states)
            x = x + self.stochastic_depth(self.feed_forward_network(x)) + hidden_states
            x = _apply_norm(self.norm_layer3, x)

        outputs["hidden_states"] = x
        return outputs


# ──────────────────────────────────────────────────────────────────────
# Full student model
# ──────────────────────────────────────────────────────────────────────

class BiMambaSLR(nn.Module):
    """
    Hybrid student — identical to SpatialTemporalTransformerWithClassToken
    except every temporal attention sub-block is replaced by BiMamba2Mixer.

    ALL structural parameters must match the teacher:
        embedding_dim, n_blocks, head_dim, n_heads, norm_type,
        ffn_expand_ratio, ffn_dropout_ratio, max_stochastic_depth_rate, use_bias

    Student-only (BiMamba) parameters:
        d_state, d_conv, chunk_size
    """

    def __init__(
        self,
        # ── Must match teacher ─────────────────────────────────────
        in_channels: int = 2,
        num_classes: int = 100,
        seq_len: int = 50,
        n_joints: int = 55,
        embedding_dim: int = 128,
        n_blocks: int = 10,
        head_dim: int = 64,
        n_heads: int = 8,
        norm_type: str = "batchnorm",     # "batchnorm" | "layernorm"
        ffn_expand_ratio: float = 4.0,
        ffn_dropout_ratio: float = 0.25,
        max_stochastic_depth_rate: float = 0.25,
        use_bias: bool = False,
        # ── BiMamba-only ───────────────────────────────────────────
        d_state: int = 64,
        d_conv: int = 3,
        chunk_size: int = 16,
        **kwargs,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_joints      = n_joints
        self.seq_len       = seq_len + 1   # +1 for class token (same as teacher)

        # ── Embedding (same as teacher) ───────────────────────────────
        self.embedding = nn.Sequential(
            nn.Linear(in_channels, embedding_dim, bias=use_bias)
        )

        # ── Class token (same as teacher) ────────────────────────────
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))

        # ── Spatial PE (same as teacher) ─────────────────────────────
        self.spatial_positional_encode = SinusoidalPositionalEncoding(
            embedding_dim, n_joints
        )

        # ── Blocks (same count + stochastic depth schedule) ──────────
        stoch_rates = np.linspace(0.0, max_stochastic_depth_rate, n_blocks)
        self.blocks = nn.ModuleList([
            HybridB2TPostNormBlock(
                embedding_dim=embedding_dim,
                head_dim=head_dim,
                n_heads=n_heads,
                d_state=d_state,
                d_conv=d_conv,
                chunk_size=chunk_size,
                seq_len=self.seq_len,
                n_joints=n_joints,
                ffn_expand_ratio=ffn_expand_ratio,
                ffn_dropout_ratio=ffn_dropout_ratio,
                norm_type=norm_type,
                stochastic_depth_rate=float(stoch_rates[i]),
                use_bias=use_bias,
                layer_idx=i,
            )
            for i in range(n_blocks)
        ])

        # ── Classifier (same as teacher) ─────────────────────────────
        self.fc = nn.Linear(embedding_dim, num_classes, bias=use_bias)

        self._init_weights()

    def _init_weights(self):
        n_blocks = len(self.blocks)

        # cls_token: small random (standard ViT practice)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        for name, m in self.named_modules():
            # ── Linear layers ────────────────────────────────────────
            if isinstance(m, nn.Linear):
                if "out_proj" in name:
                    # Depth-scaled init for output projections
                    # (GPT-2 / Mamba practice: std / sqrt(2 * n_layers))
                    std = 0.02 / (2 * n_blocks) ** 0.5
                    nn.init.trunc_normal_(m.weight, std=std)
                else:
                    nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            # ── Conv1d (depthwise conv in BiMamba2Mixer) ─────────────
            elif isinstance(m, nn.Conv1d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            # ── Norm layers ───────────────────────────────────────────
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Weight transfer from teacher
    # ------------------------------------------------------------------

    def load_teacher_weights(self, teacher_model: "nn.Module"):
        """
        Bơm trọng số từ Teacher sang Student.

        Sao chép:  embedding, cls_token, spatial PE,
                   spatial MHA (multihead_self_attention1),
                   FFN (feed_forward_network),
                   norm_layer1/2/3, classifier (fc).

        Bỏ qua:    temporal attention (multihead_self_attention2)
                   → student dùng BiMamba2 thay thế.

        Sau khi copy, đóng băng tất cả tham số đã copy.
        Chỉ để lại temporal_mamba trainable → sẵn sàng cho Stage 1.
        """
        # Unwrap TeacherModel wrapper nếu cần
        raw_teacher = teacher_model.model if hasattr(teacher_model, "model") else teacher_model
        teacher_dict = raw_teacher.state_dict()
        student_dict = self.state_dict()

        copied, skipped_temporal, skipped_shape, skipped_missing = [], [], [], []

        for key, param in teacher_dict.items():
            # Bỏ qua toàn bộ temporal attention của teacher
            if "multihead_self_attention2" in key:
                skipped_temporal.append(key)
                continue

            if key in student_dict:
                if param.shape == student_dict[key].shape:
                    student_dict[key].copy_(param)
                    copied.append(key)
                else:
                    skipped_shape.append(f"{key}: teacher={tuple(param.shape)} student={tuple(student_dict[key].shape)}")
            else:
                skipped_missing.append(key)

        self.load_state_dict(student_dict, strict=False)

        print(f"[WeightTransfer] Sao chép thành công : {len(copied)} tensors")
        print(f"[WeightTransfer] Bỏ qua (temporal)   : {len(skipped_temporal)} tensors")
        if skipped_shape:
            print(f"[WeightTransfer] Bỏ qua (shape khác): {len(skipped_shape)} tensors")
            for s in skipped_shape:
                print(f"    {s}")
        if skipped_missing:
            print(f"[WeightTransfer] Bỏ qua (không tồn tại): {len(skipped_missing)}")

        # Đóng băng tất cả trừ temporal_mamba
        frozen = trainable = 0
        for name, param in self.named_parameters():
            if "temporal_mamba" in name:
                param.requires_grad_(True)
                trainable += param.numel()
            else:
                param.requires_grad_(False)
                frozen += param.numel()

        print(f"[WeightTransfer] Đóng băng  : {frozen:,} params")
        print(f"[WeightTransfer] Trainable  : {trainable:,} params  (temporal_mamba only)")

    # ------------------------------------------------------------------
    # Feature extraction — mirrors teacher.extract_feature exactly
    # ------------------------------------------------------------------

    def extract_feature(
        self,
        skeleton_data: torch.Tensor,           # (B, C, T, V, M)
        return_transfer_matrix: bool = False,
        run_mlp_component: bool = True,
        collect_hidden_states: bool = False,
    ):
        B, C, T, V, M = skeleton_data.size()

        x = skeleton_data.transpose(1, -1).contiguous().view(-1, T, V, C)
        x = self.embedding(x)                          # (BM, T, V, D)

        cls = self.cls_token.expand(B * M, 1, V, -1)
        x = torch.cat([cls, x], dim=1)                # (BM, T+1, V, D)

        x = self.spatial_positional_encode(x)

        temporal_tms, all_hidden = [], []

        for block in self.blocks:
            out = block(
                hidden_states=x,
                return_transfer_matrix=return_transfer_matrix,
                run_mlp_component=run_mlp_component,
            )
            x = out["hidden_states"]
            if return_transfer_matrix:
                temporal_tms.append(out["transfer_matrix"])
            if collect_hidden_states:
                all_hidden.append(x)

        return x, temporal_tms, all_hidden

    # ------------------------------------------------------------------
    # Forward — mirrors teacher.forward exactly
    # ------------------------------------------------------------------

    def forward(self, skeleton_data: torch.Tensor, **kwargs) -> torch.Tensor:
        B, C, T, V, M = skeleton_data.size()
        x, _, _ = self.extract_feature(skeleton_data)
        # cls token at T=0, mean over V joints (same as teacher)
        x = x[:, 0].contiguous().view(B, V, -1).mean(dim=1)
        return self.fc(x)

    # ------------------------------------------------------------------
    # Distillation helper
    # ------------------------------------------------------------------

    def forward_with_intermediates(
        self,
        skeleton_data: torch.Tensor,
        return_transfer_matrix: bool = False,
        run_mlp_component: bool = True,
        collect_hidden_states: bool = True,
    ) -> dict:
        B, C, T, V, M = skeleton_data.size()
        x, tm_tms, all_hs = self.extract_feature(
            skeleton_data,
            return_transfer_matrix=return_transfer_matrix,
            run_mlp_component=run_mlp_component,
            collect_hidden_states=collect_hidden_states,
        )
        cls_feat = x[:, 0].contiguous().view(B, V, -1).mean(dim=1)
        logits   = self.fc(cls_feat)

        return {
            "logits":        logits,
            "temporal_tms":  tm_tms,     # list[n_blocks] of (BM, V, H, T+1, T+1)
            "hidden_states": all_hs,     # list[n_blocks] of (BM, T+1, V, D)
        }
