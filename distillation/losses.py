"""
Loss functions for MOHAWK-style distillation.

Stage 1 — Matrix alignment:  Frobenius norm  ||M_student - M_teacher||_F
Stage 2 — Hidden alignment:  L2 norm         ||h_student - h_teacher||_2
Stage 3 — Full distillation: KL divergence (temperature-scaled) + CE on hard labels
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def frobenius_loss(student_matrix: torch.Tensor, teacher_matrix: torch.Tensor) -> torch.Tensor:
    """
    Mean Frobenius norm between student transfer matrix and teacher attention matrix.

    Both tensors: (B, H, L, L)
    Returns scalar loss.
    """
    assert student_matrix.shape == teacher_matrix.shape, (
        f"Shape mismatch: student {student_matrix.shape} vs teacher {teacher_matrix.shape}"
    )
    return torch.linalg.matrix_norm(student_matrix - teacher_matrix, ord="fro").mean()


def hidden_state_l2_loss(
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
) -> torch.Tensor:
    """
    Per-token L2 norm giữa student và teacher hidden states.

    Theo phi-mamba mohawk_stage2.py dòng 79-81:
        loss = torch.norm(student - teacher, p=2, dim=(-1,)).mean()

    Với shape (BM, T+1, V, D):
        - Tính ||h_s[b,t,v,:] - h_t[b,t,v,:]||_2  per (b, t, v) tuple
        - mean() qua tất cả (b, t, v)

    Khác với F.mse_loss (mean of squared elements):
        L2 norm per token đo khoảng cách Euclidean giữa các representation
        vectors, nhạy hơn với lệch hướng trong feature space.
    """
    return torch.norm(student_hidden - teacher_hidden, p=2, dim=-1).mean()


def kl_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 4.0,
) -> torch.Tensor:
    """
    KL divergence loss (soft targets) between student and teacher logits.

    Returns scalar loss scaled by T^2 (Hinton et al.).
    """
    T = temperature
    s_log_prob = F.log_softmax(student_logits / T, dim=-1)
    t_prob     = F.softmax(teacher_logits / T, dim=-1)
    return F.kl_div(s_log_prob, t_prob, reduction="batchmean") * (T ** 2)


def classification_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Standard cross-entropy loss."""
    return F.cross_entropy(logits, labels.long().squeeze(-1))


def combined_stage3_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.5,
    temperature: float = 4.0,
) -> torch.Tensor:
    """
    Stage-3 combined loss: alpha * KL + (1-alpha) * CE.

    alpha=0.5 balances soft-label distillation with hard-label supervision.
    """
    kl   = kl_distillation_loss(student_logits, teacher_logits, temperature)
    ce   = classification_loss(student_logits, labels)
    return alpha * kl + (1 - alpha) * ce
