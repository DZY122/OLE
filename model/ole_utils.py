import math
from typing import Optional

import torch


def spectral_normalize_matrix(matrix: torch.Tensor, n_power_iter: int = 2, eps: float = 1e-6) -> torch.Tensor:
    if matrix.ndim != 2:
        raise ValueError("spectral_normalize_matrix expects a 2D tensor")
    with torch.no_grad():
        v = torch.randn(matrix.shape[1], device=matrix.device, dtype=matrix.dtype)
        v = v / (v.norm() + eps)
        for _ in range(max(1, n_power_iter)):
            u = matrix @ v
            u = u / (u.norm() + eps)
            v = matrix.transpose(0, 1) @ u
            v = v / (v.norm() + eps)
        sigma = (u * (matrix @ v)).sum().abs()
    return matrix / (sigma + eps)


def nuclear_norm_fn(matrix: torch.Tensor, mode: str = "exact", eps: float = 1e-8) -> torch.Tensor:
    if mode != "exact":
        raise ValueError(f"Unsupported nuclear norm mode: {mode}")
    # torch.linalg.svdvals is not implemented for float16 on some CUDA backends.
    matrix_svd = matrix.float() if matrix.dtype in (torch.float16, torch.bfloat16) else matrix
    svals = torch.linalg.svdvals(matrix_svd)
    norm = svals.sum()
    return norm / math.sqrt(matrix.shape[1] + eps)


def effective_rank(matrix: torch.Tensor, eps: float = 1e-8) -> float:
    matrix_svd = matrix.float() if matrix.dtype in (torch.float16, torch.bfloat16) else matrix
    svals = torch.linalg.svdvals(matrix_svd)
    total = svals.sum() + eps
    p = svals / total
    entropy = -(p * torch.log(p + eps)).sum()
    return torch.exp(entropy).item()


def mean_pairwise_cosine(head_features: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # head_features: [H, M, D]
    h = head_features.shape[0]
    flat = head_features.reshape(h, -1)
    flat = flat / (flat.norm(dim=1, keepdim=True) + eps)
    sim = flat @ flat.transpose(0, 1)
    return (sim.sum() - torch.diag(sim).sum()) / max(h * (h - 1), 1)
