# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

from functools import partial
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

import timm.models.vision_transformer
from timm.layers.attention import maybe_add_mask

from model.ole_utils import mean_pairwise_cosine, nuclear_norm_fn, spectral_normalize_matrix


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class OLEAttention(timm.models.vision_transformer.Attention):
    def __init__(self, *args, layer_index=0, ole_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        ole_config = ole_config or {}
        self.layer_index = layer_index
        self.ole_mode = ole_config.get("ole_mode", "none")
        self.ole_lambda_sum = ole_config.get("ole_lambda_sum", 1.0)
        self.ole_solver_step_size = ole_config.get("ole_solver_step_size", 0.1)
        self.ole_solver_second_order = ole_config.get("ole_solver_second_order", False)
        self.nuclear_norm_mode = ole_config.get("nuclear_norm_mode", "exact")
        self.ole_log_head_stats = ole_config.get("ole_log_head_stats", False)
        self.selected_layers = set(ole_config.get("ole_layers", []))

        self._ole_enabled = self.ole_mode != "none" and self.layer_index in self.selected_layers
        self.last_ole_loss = None
        self.last_ole_stats = {}
        self.last_out_heads = None

        if self._ole_enabled:
            eye = torch.eye(self.head_dim)
            noise = 0.01 * torch.randn(self.head_dim, self.head_dim)
            self.ole_t = nn.Parameter(eye + noise)
        else:
            self.register_parameter("ole_t", None)

    def _compute_layer_ole(self, out_heads: torch.Tensor, t_eff: torch.Tensor) -> torch.Tensor:
        b, h, n, d = out_heads.shape
        transformed = torch.matmul(out_heads, t_eff.transpose(0, 1))
        x = transformed.permute(1, 0, 2, 3).reshape(h, b * n, d)
        head_terms = torch.stack([nuclear_norm_fn(xi, mode=self.nuclear_norm_mode) for xi in x], dim=0)
        x_sum = x.sum(dim=0)
        sum_term = nuclear_norm_fn(x_sum, mode=self.nuclear_norm_mode)
        if self.ole_log_head_stats:
            self.last_ole_stats = {
                "head_nuclear": head_terms.mean().detach().item(),
                "sum_nuclear": sum_term.detach().item(),
                "head_cosine": mean_pairwise_cosine(x.detach()).item(),
            }
        return head_terms.sum() - sum_term
        # return head_terms.mean() - self.ole_lambda_sum * sum_term

    def _get_t_eff(self, out_heads: torch.Tensor):
        if not self._ole_enabled:
            return None, None
        t_base = spectral_normalize_matrix(self.ole_t)
        if self.ole_mode == "learned_t":
            return t_base, self._compute_layer_ole(out_heads, t_base)
        if self.ole_mode == "solver_t":
            local_obj = self._compute_layer_ole(out_heads, t_base)
            grad_t = torch.autograd.grad(
                local_obj,
                t_base,
                retain_graph=True,
                create_graph=self.ole_solver_second_order,
                allow_unused=False,
            )[0]
            grad_t_used = grad_t if self.ole_solver_second_order else grad_t.detach()
            t_eff = spectral_normalize_matrix(t_base - self.ole_solver_step_size * grad_t_used)
            return t_eff, self._compute_layer_ole(out_heads, t_eff)
        raise ValueError(f"Unsupported ole_mode: {self.ole_mode}")

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
    def __init__(self, global_pool=False, ole_mode="none", ole_loss_weight=0.0, ole_lambda_sum=1.0,
                 ole_layers="last3", ole_solver_step_size=0.1, ole_solver_second_order=False,
                 ole_log_head_stats=False, nuclear_norm_mode="exact", **kwargs):
        depth = kwargs.get("depth", 12)
        num_heads = kwargs.get("num_heads", 12)
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
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_small_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
