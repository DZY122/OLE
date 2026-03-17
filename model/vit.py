# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import math
from functools import partial
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

import timm.models.vision_transformer
from timm.layers.attention import maybe_add_mask

from model.ole_utils import spectral_normalize_matrix


def nuclear_norm_fn(matrix: torch.Tensor, mode: str = "exact", eps: float = 1e-8, approx_rank: int = 8) -> torch.Tensor:
    # torch.linalg.svdvals is not implemented for float16 on some CUDA backends.
    matrix_svd = matrix.float() if matrix.dtype in (torch.float16, torch.bfloat16) else matrix

    if mode == "exact":
        svals = torch.linalg.svdvals(matrix_svd)
        norm = svals.sum()
        return norm / math.sqrt(matrix.shape[1] + eps)

    if mode == "approx":
        m, n = matrix_svd.shape
        k = max(1, min(int(approx_rank), m, n))
        omega = torch.randn(n, k, device=matrix_svd.device, dtype=matrix_svd.dtype)
        y = matrix_svd @ omega
        q, _ = torch.linalg.qr(y, mode="reduced")
        b = q.transpose(0, 1) @ matrix_svd
        svals = torch.linalg.svdvals(b)
        norm = svals.sum()
        return norm / math.sqrt(matrix.shape[1] + eps)

    raise ValueError(f"Unsupported nuclear norm mode: {mode}")


def effective_rank(matrix: torch.Tensor, eps: float = 1e-8) -> float:
    matrix_svd = matrix.float() if matrix.dtype in (torch.float16, torch.bfloat16) else matrix
    svals = torch.linalg.svdvals(matrix_svd)
    total = svals.sum() + eps
    p = svals / total
    entropy = -(p * torch.log(p + eps)).sum()
    return torch.exp(entropy).item()


def centered_mean_pairwise_cosine(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    x: [H, M, D]
       H = num_heads
       M = B*N (sample axis)
       D = head_dim

    For each head:
      1) subtract mean over M
      2) flatten to [H, M*D]
      3) compute pairwise cosine
      4) return mean off-diagonal cosine
    """
    h = x.shape[0]
    if h <= 1:
        return x.new_tensor(0.0)

    x = x.float()
    x_centered = x - x.mean(dim=1, keepdim=True)
    x_flat = x_centered.reshape(h, -1)
    x_flat = x_flat / x_flat.norm(dim=1, keepdim=True).clamp_min(eps)
    sim = x_flat @ x_flat.transpose(0, 1)

    mask = ~torch.eye(h, dtype=torch.bool, device=sim.device)
    return sim[mask].mean().abs_()


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def nuclear_norm_subgrad(A: torch.Tensor, delta: float = 1e-6) -> torch.Tensor:
    """
    One simple valid subgradient of ||A||_*:
        G = U_r V_r^T
    where r counts singular values > delta.

    A: [m, n]
    return: [m, n]
    """
    A_svd = A.float() if A.dtype in (torch.float16, torch.bfloat16) else A
    U, S, Vh = torch.linalg.svd(A_svd, full_matrices=False)
    r = int((S > delta).sum().item())
    if r == 0:
        G = torch.zeros_like(A_svd)
    else:
        G = U[:, :r] @ Vh[:r, :]
    return G.to(A.dtype) if A.dtype in (torch.float16, torch.bfloat16) else G


class OLEAttention(timm.models.vision_transformer.Attention):
    def __init__(self, *args, layer_index=0, ole_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        ole_config = ole_config or {}
        self.layer_index = layer_index
        self.ole_mode = ole_config.get("ole_mode", "none")
        self.ole_lambda_sum = ole_config.get("ole_lambda_sum", 1.0)

        # 改小：更接近 notebook 里的 one-step inner lr
        self.ole_solver_step_size = ole_config.get("ole_solver_step_size", 1e-3)

        self.nuclear_norm_mode = ole_config.get("nuclear_norm_mode", "exact")
        self.ole_nuclear_rank = ole_config.get("ole_nuclear_rank", 8)
        self.ole_log_head_stats = ole_config.get("ole_log_head_stats", False)
        self.selected_layers = set(ole_config.get("ole_layers", []))
        self.ole_subgrad_delta = ole_config.get("ole_subgrad_delta", 1e-6)

        # alternating-style: solve T on fixed current features
        self.ole_detach_solver_features = ole_config.get("ole_detach_solver_features", True)
        # 每多少个 train forward 更新一次 T
        self.ole_update_interval = max(1, int(ole_config.get("ole_update_interval", 1)))

        self._ole_enabled = self.ole_mode != "none" and self.layer_index in self.selected_layers
        self.last_ole_loss = None
        self.last_ole_stats = {}
        self.last_out_heads = None
        self.last_t_eff = None
        self.register_buffer("_ole_forward_count", torch.zeros((), dtype=torch.long), persistent=False)

        if self._ole_enabled:
            eye = torch.eye(self.head_dim)
            noise = 0.01 * torch.randn(self.head_dim, self.head_dim)
            self.ole_t = nn.Parameter(eye + noise)
        else:
            self.register_parameter("ole_t", None)

    # ------------------------------------------------------------------
    # Build matrices in CCCP / OLT convention:
    # each head is one "class"
    #   Y_h in R^{D x (B*N)}
    # all heads concatenated by columns:
    #   Y   in R^{D x (H*B*N)}
    # ------------------------------------------------------------------
    def _build_head_matrices(self, out_heads: torch.Tensor):
        # out_heads: [B, H, N, D]
        b, h, n, d = out_heads.shape

        # [H, D, B*N]
        y_heads = out_heads.permute(1, 3, 0, 2).contiguous().reshape(h, d, b * n)

        # [D, H*B*N]
        y_all = y_heads.permute(1, 0, 2).contiguous().reshape(d, h * b * n)

        return y_heads, y_all

    def _compute_layer_ole_from_mats(
        self,
        y_heads: torch.Tensor,   # [H, D, B*N]
        y_all: torch.Tensor,     # [D, H*B*N]
        t_eff: torch.Tensor,     # [D, D]
    ) -> torch.Tensor:
        ty_heads = torch.matmul(t_eff.unsqueeze(0), y_heads)   # [H, D, B*N]

        head_terms = torch.stack(
            [
                nuclear_norm_fn(
                    ty_heads[i],
                    mode=self.nuclear_norm_mode,
                    approx_rank=self.ole_nuclear_rank,
                )
                for i in range(ty_heads.shape[0])
            ],
            dim=0,
        )

        union_term = nuclear_norm_fn(
            t_eff @ y_all,
            mode=self.nuclear_norm_mode,
            approx_rank=self.ole_nuclear_rank,
        )

        if self.ole_log_head_stats:
            x = ty_heads.permute(0, 2, 1).contiguous()  # [H, B*N, D]
            centered_cos = centered_mean_pairwise_cosine(x.detach())
            self.last_ole_stats = {
                "head_nuclear": head_terms.mean().detach().item(),
                "union_nuclear": union_term.detach().item(),
                "head_cosine": centered_cos.item(),
                "head_centered_cosine": centered_cos.item(),
            }

        return head_terms.sum() - self.ole_lambda_sum * union_term

    def _compute_layer_ole(self, out_heads: torch.Tensor, t_eff: torch.Tensor) -> torch.Tensor:
        y_heads, y_all = self._build_head_matrices(out_heads)
        return self._compute_layer_ole_from_mats(y_heads, y_all, t_eff)

    # ------------------------------------------------------------------
    # One-step CCCP update without spectral normalization
    #
    # objective:
    #   J(T) = sum_h ||T Y_h||_* - lambda ||T Y||_*
    #
    # CCCP surrogate at current T^(t):
    #   J_sur(T; T^(t)) = sum_h ||T Y_h||_* - trace(M^T T)
    # where
    #   M = lambda * G_A * Y^T
    #   G_A in ∂||T^(t)Y||_*
    #
    # We do ONLY ONE gradient step on this surrogate.
    # ------------------------------------------------------------------
    def _cccp_one_step(self, out_heads: torch.Tensor, t_base: torch.Tensor) -> torch.Tensor:
        solver_heads = out_heads.detach() if self.ole_detach_solver_features else out_heads
        y_heads, y_all = self._build_head_matrices(solver_heads)  # [H,D,BN], [D,HBN]

        # 1) linearize the concave term at current T^(t)
        with torch.no_grad():
            a_all = t_base.detach() @ y_all                                   # [D, HBN]
            g_a = nuclear_norm_subgrad(a_all, delta=self.ole_subgrad_delta)   # [D, HBN]
            m = self.ole_lambda_sum * (g_a @ y_all.transpose(0, 1))           # [D, D]

            # 2) explicit subgradient for sum_h ||T Y_h||_* term:
            #    if G_h in ∂||T Y_h||_*, then
            #      ∂_T ||T Y_h||_* contains G_h Y_h^T
            grad_within = torch.zeros_like(t_base)
            for i in range(y_heads.shape[0]):
                a_h = t_base.detach() @ y_heads[i]                            # [D, BN]
                g_h = nuclear_norm_subgrad(a_h, delta=self.ole_subgrad_delta) # [D, BN]
                grad_within = grad_within + (g_h @ y_heads[i].transpose(0, 1))

            # surrogate gradient:
            #   ∂_T [sum_h ||T Y_h||_* - trace(M^T T)] = grad_within - M
            grad_t = grad_within - m

            # no spectral normalization
            t_eff = t_base.detach() - self.ole_solver_step_size * grad_t

        return t_eff.to(t_base.dtype)

    def _should_update_t_this_forward(self) -> bool:
        return bool((self._ole_forward_count.remainder(self.ole_update_interval) == 0).item())

    def _get_t_eff(self, out_heads: torch.Tensor):
        if not self._ole_enabled:
            return None, None

        # 取消 spectral normalization，直接用当前参数
        t_base = self.ole_t

        if self.ole_mode == "learned_t":
            return t_base, self._compute_layer_ole(out_heads, t_base)

        if self.ole_mode == "solver_t":
            # eval/test: directly use stored T
            if (not self.training) or (not torch.is_grad_enabled()):
                return t_base, self._compute_layer_ole(out_heads, t_base)

            should_update = self._should_update_t_this_forward()
            self._ole_forward_count.add_(1)

            if should_update:
                t_eff = self._cccp_one_step(out_heads, t_base)

                with torch.no_grad():
                    self.ole_t.copy_(t_eff.detach())

                self.last_t_eff = t_eff.detach()
                t_use = t_eff.detach()
            else:
                self.last_t_eff = t_base.detach()
                t_use = t_base.detach()

            layer_ole = self._compute_layer_ole(out_heads, t_use)
            return t_use, layer_ole

        raise ValueError(f"Unsupported ole_mode: {self.ole_mode}")

    def commit_last_t_eff(self):
        """
        Keep for compatibility.
        Since we already write back self.ole_t every train step in solver_t mode,
        this function mainly acts as a final safeguard.
        """
        if (not self._ole_enabled) or (self.ole_mode != "solver_t") or (self.last_t_eff is None):
            return False

        with torch.no_grad():
            self.ole_t.copy_(self.last_t_eff)

        return True

    def forward(self, x: torch.Tensor, attn_mask=None) -> torch.Tensor:
        B, N, _ = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            out_heads = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = maybe_add_mask(attn, attn_mask)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            out_heads = attn @ v

        if self._ole_enabled:
            t_eff, layer_ole = self._get_t_eff(out_heads)

            # use the updated T on current batch
            out_heads = torch.matmul(out_heads, t_eff.transpose(0, 1))

            self.last_ole_loss = layer_ole
        else:
            self.last_ole_loss = None
            self.last_ole_stats = {}

        self.last_out_heads = out_heads.detach()

        x = out_heads.transpose(1, 2).reshape(B, N, self.attn_dim)
        x = self.norm(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class OLEBlock(timm.models.vision_transformer.Block):
    def __init__(self, *args, ole_config=None, depth=0, **kwargs):
        super().__init__(*args, depth=depth, **kwargs)
        self.attn = OLEAttention(
            dim=kwargs["dim"] if "dim" in kwargs else args[0],
            num_heads=kwargs.get("num_heads", args[1] if len(args) > 1 else 8),
            qkv_bias=kwargs.get("qkv_bias", False),
            qk_norm=kwargs.get("qk_norm", False),
            scale_norm=kwargs.get("scale_attn_norm", False),
            proj_bias=kwargs.get("proj_bias", True),
            proj_drop=kwargs.get("proj_drop", 0.0),
            attn_drop=kwargs.get("attn_drop", 0.0),
            norm_layer=kwargs.get("norm_layer", nn.LayerNorm),
            layer_index=depth,
            ole_config=ole_config,
        )


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """Vision Transformer with support for global average pooling + optional OLE attention."""

    def __init__(
        self,
        global_pool=False,
        ole_mode="none",
        ole_loss_weight=0.0,
        ole_lambda_sum=1.0,
        ole_layers="all",
        ole_solver_step_size=0.1,
        ole_solver_second_order=False,
        ole_log_head_stats=False,
        nuclear_norm_mode="exact",
        ole_update_interval=1,
        ole_nuclear_rank=8,
        **kwargs,
    ):
        depth = kwargs.get("depth", 12)
        embed_dim = kwargs.get("embed_dim", 768)
        selected_layers = self._parse_ole_layers(ole_layers, depth)
        ole_cfg = {
            "ole_mode": ole_mode,
            "ole_lambda_sum": ole_lambda_sum,
            "ole_solver_step_size": ole_solver_step_size,
            "ole_solver_second_order": ole_solver_second_order,
            "ole_log_head_stats": ole_log_head_stats,
            "ole_layers": selected_layers,
            "nuclear_norm_mode": nuclear_norm_mode,
            "ole_update_interval": ole_update_interval,
            "ole_nuclear_rank": ole_nuclear_rank,
        }
        super(VisionTransformer, self).__init__(
            block_fn=partial(OLEBlock, ole_config=ole_cfg),
            global_pool="token",
            **kwargs,
        )

        self.global_pool = global_pool
        self.ole_mode = ole_mode
        self.ole_loss_weight = ole_loss_weight
        self.ole_layers = selected_layers
        self.ole_aux_loss = torch.tensor(0.0)
        self.ole_head_stats: Dict[int, Dict[str, float]] = {}

        if self.global_pool:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
            self.fc_norm = norm_layer(embed_dim)
            del self.norm

    @staticmethod
    def _parse_ole_layers(ole_layers, depth: int) -> List[int]:
        if isinstance(ole_layers, str):
            if ole_layers == "last3":
                return list(range(max(depth - 3, 0), depth))
            if ole_layers == "all":
                return list(range(depth))
            return [int(x.strip()) for x in ole_layers.split(",") if x.strip()]
        if ole_layers is None:
            return list(range(max(depth - 3, 0), depth))
        return [int(x) for x in ole_layers]

    def get_aux_loss(self):
        return self.ole_aux_loss

    def get_ole_head_stats(self):
        return self.ole_head_stats

    def commit_solver_t(self):
        """
        Commit the LAST train-time t_eff into self.ole_t for all OLE-attention blocks.
        Call once after training, before final eval/test.
        """
        committed = 0
        for blk in self.blocks:
            if hasattr(blk, "attn") and hasattr(blk.attn, "commit_last_t_eff"):
                committed += int(blk.attn.commit_last_t_eff())
        return committed

    def forward_head(self, x, pre_logits: bool = False):
        if hasattr(self, "fc_norm"):
            x = self.fc_norm(x)
        elif hasattr(self, "norm"):
            x = self.norm(x)
        return self.head(x)

    def forward_features(self, x, attn_mask=None):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        ole_losses = []
        ole_stats = {}
        for idx, blk in enumerate(self.blocks):
            x = blk(x, attn_mask=attn_mask)
            layer_loss = getattr(blk.attn, "last_ole_loss", None)
            if layer_loss is not None:
                ole_losses.append(layer_loss)
                if blk.attn.last_ole_stats:
                    ole_stats[idx] = blk.attn.last_ole_stats

        if ole_losses:
            self.ole_aux_loss = torch.stack(ole_losses).mean()
        else:
            self.ole_aux_loss = x.new_zeros(())
        self.ole_head_stats = ole_stats

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]
        return outcome


def vit_tiny_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def vit_small_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model
