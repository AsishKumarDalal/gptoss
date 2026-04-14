# model_parts.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from config_model import ModelConfig


# ──────────────────────────────────────────────────────────────────────────────
# KV Cache
# ──────────────────────────────────────────────────────────────────────────────

class LayerKVCache:
    def __init__(
        self,
        batch_size:   int,
        num_kv_heads: int,
        max_seq_len:  int,
        head_dim:     int,
        device:       torch.device,
        dtype:        torch.dtype = torch.float32,
    ):
        self.max_seq_len  = max_seq_len
        self.seq_len: int = 0

        self.k_cache = torch.zeros(
            batch_size, num_kv_heads, max_seq_len, head_dim,
            device=device, dtype=dtype,
        )
        self.v_cache = torch.zeros(
            batch_size, num_kv_heads, max_seq_len, head_dim,
            device=device, dtype=dtype,
        )

    def update(
        self,
        new_k: torch.Tensor,
        new_v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        T_new = new_k.shape[2]
        end   = self.seq_len + T_new
        if end > self.max_seq_len:
            raise ValueError(
                f"KV cache overflow: position {end - 1} >= max_seq_len {self.max_seq_len}. "
                "Call reset() between requests or increase max_seq_len."
            )
        self.k_cache[:, :, self.seq_len:end] = new_k
        self.v_cache[:, :, self.seq_len:end] = new_v
        self.seq_len = end
        return self.k_cache[:, :, :end], self.v_cache[:, :, :end]

    def reset(self) -> None:
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.seq_len = 0


# ──────────────────────────────────────────────────────────────────────────────
# Grouped Query Attention + RoPE  (KV-cache compatible)
# ──────────────────────────────────────────────────────────────────────────────

class GroupedQueryAttentionWithRoPE(nn.Module):

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model % cfg.num_heads      == 0, "d_model must be divisible by num_heads"
        assert cfg.num_heads % cfg.num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"

        self.num_heads    = cfg.num_heads
        self.num_kv_heads = cfg.num_kv_heads
        self.num_groups   = cfg.num_heads // cfg.num_kv_heads
        self.head_dim     = cfg.d_model // cfg.num_heads
        self.d_out        = cfg.d_model
        self.use_rope     = cfg.use_rope
        self.use_attn_bias = cfg.use_attention_bias

        self.W_q      = nn.Linear(cfg.d_model, cfg.d_model,                       bias=cfg.qkv_bias)
        self.W_k      = nn.Linear(cfg.d_model, cfg.num_kv_heads * self.head_dim,  bias=cfg.qkv_bias)
        self.W_v      = nn.Linear(cfg.d_model, cfg.num_kv_heads * self.head_dim,  bias=cfg.qkv_bias)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.dropout  = nn.Dropout(cfg.dropout)

        if cfg.use_attention_bias:
            self.attention_bias = nn.Parameter(
                torch.zeros(1, cfg.num_heads, cfg.max_seq_len, cfg.max_seq_len)
            )

        if cfg.use_rope:
            # FIX: precompute for head_dim, NOT head_dim//2.
            # _precompute_rope returns cos/sin of shape (max_seq_len, head_dim).
            cos, sin = self._precompute_rope(cfg.max_seq_len, self.head_dim)
            self.register_buffer("cos_cached", cos)   # (max_seq_len, head_dim)
            self.register_buffer("sin_cached", sin)

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _precompute_rope(seq_len: int, dim: int, theta: float = 10_000.0):
        """
        Returns cos/sin tables of shape (seq_len, dim).
        inv_freq has dim//2 entries; we cat(freqs, freqs) to fill all dim slots
        so that the rotate-half trick works without any re-chunking.
        """
        assert dim % 2 == 0, "head_dim must be even for RoPE"
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))  # (dim//2,)
        t        = torch.arange(seq_len).float()                               # (seq_len,)
        freqs    = torch.outer(t, inv_freq)                                    # (seq_len, dim//2)
        emb      = torch.cat((freqs, freqs), dim=-1)                           # (seq_len, dim)
        return emb.cos(), emb.sin()

    @staticmethod
    def _apply_rotary_emb(
        x:   torch.Tensor,   # (b, H, T, D)
        cos: torch.Tensor,   # (1, 1, T, D)  — already sliced & reshaped by caller
        sin: torch.Tensor,   # (1, 1, T, D)
    ) -> torch.Tensor:
        """
        Rotate-half RoPE.  cos/sin are already (1,1,T,D) — no re-chunking needed.
        x is split into first-half / second-half along head_dim.
        """
        D = x.shape[-1]
        half = D // 2
        # FIX: split on head_dim directly — no .chunk() on cos/sin
        x1 = x[..., :half]    # (b, H, T, D//2)
        x2 = x[..., half:]    # (b, H, T, D//2)
        c  = cos[..., :half]   # (1, 1, T, D//2)
        s  = sin[..., :half]   # (1, 1, T, D//2)
        return torch.cat([x1 * c - x2 * s,
                          x2 * c + x1 * s], dim=-1)

    def _repeat_kv(self, kv: torch.Tensor) -> torch.Tensor:
        """Expand (b, Hkv, S, D) → (b, H, S, D)."""
        if self.num_groups == 1:
            return kv
        b, hkv, S, D = kv.shape
        kv = kv[:, :, None, :, :].expand(b, hkv, self.num_groups, S, D)
        return kv.reshape(b, hkv * self.num_groups, S, D)

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        x:        torch.Tensor,              # (b, T, d_model)
        kv_cache: Optional[LayerKVCache] = None,
    ) -> torch.Tensor:

        b, T, _ = x.shape

        q = self.W_q(x).view(b, T, self.num_heads,    self.head_dim).transpose(1, 2)
        k = self.W_k(x).view(b, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.W_v(x).view(b, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if self.use_rope:
            offset = kv_cache.seq_len if kv_cache is not None else 0

            # FIX: guard against offset + T exceeding precomputed table
            max_pos = self.cos_cached.shape[0]
            if offset + T > max_pos:
                raise ValueError(
                    f"RoPE table overflow: offset ({offset}) + T ({T}) = {offset + T} "
                    f"> max_seq_len ({max_pos}). Increase context_length in ModelConfig."
                )

            # Slice exact T positions, then reshape for broadcasting
            cos = self.cos_cached[offset:offset + T]            # (T, D)
            sin = self.sin_cached[offset:offset + T]            # (T, D)
            cos = cos.unsqueeze(0).unsqueeze(0)                  # (1, 1, T, D)
            sin = sin.unsqueeze(0).unsqueeze(0)                  # (1, 1, T, D)

            q = self._apply_rotary_emb(q, cos, sin)
            k = self._apply_rotary_emb(k, cos, sin)

        if kv_cache is not None:
            k_full, v_full = kv_cache.update(k, v)
        else:
            k_full, v_full = k, v

        S = k_full.shape[2]

        k_full = self._repeat_kv(k_full)
        v_full = self._repeat_kv(v_full)

        # (b, H, T, D) × (b, H, D, S) → (b, H, T, S)
        attn = torch.matmul(q, k_full.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if self.use_attn_bias:
            row_start = S - T
            attn = attn + self.attention_bias[:, :, row_start:row_start + T, :S]

        if kv_cache is None:
            causal = torch.triu(
                torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
            )
            attn = attn.masked_fill(causal, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        ctx = torch.matmul(attn, v_full)                         # (b, H, T, D)
        ctx = ctx.transpose(1, 2).contiguous().view(b, T, self.d_out)
        return self.out_proj(ctx)


# ──────────────────────────────────────────────────────────────────────────────
# Expert
# ──────────────────────────────────────────────────────────────────────────────

class Expert(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.fc1  = nn.Linear(cfg.d_model, cfg.d_ff)
        self.fc2  = nn.Linear(cfg.d_ff,    cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.drop(F.gelu(self.fc1(x))))


# ──────────────────────────────────────────────────────────────────────────────
# Router
# ──────────────────────────────────────────────────────────────────────────────

class TopKRouter(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.num_experts = cfg.num_experts
        self.top_k       = cfg.top_k
        self.W_r = nn.Linear(cfg.d_model, cfg.num_experts, bias=False)

    def forward(self, x: torch.Tensor):
        router_probs = F.softmax(self.W_r(x), dim=-1)            # (N, E)
        gate_weights, expert_idx = torch.topk(router_probs, self.top_k, dim=-1)
        gate_weights = gate_weights / gate_weights.sum(dim=-1, keepdim=True)
        return gate_weights, expert_idx, router_probs


# ──────────────────────────────────────────────────────────────────────────────
# Load-balance auxiliary loss
# ──────────────────────────────────────────────────────────────────────────────

def load_balance_loss(
    router_probs: torch.Tensor,
    expert_idx:   torch.Tensor,
    num_experts:  int,
) -> torch.Tensor:
    one_hot    = F.one_hot(expert_idx, num_classes=num_experts).float()
    token_mask = one_hot.sum(dim=1).clamp(max=1.0)
    f = token_mask.mean(dim=0)
    P = router_probs.mean(dim=0)
    return num_experts * (f * P).sum()


# ──────────────────────────────────────────────────────────────────────────────
# MoE FFN layer
# ──────────────────────────────────────────────────────────────────────────────

class MoELayer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg     = cfg
        self.router  = TopKRouter(cfg)
        self.experts = nn.ModuleList([Expert(cfg) for _ in range(cfg.num_experts)])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, T, d = x.shape
        N = b * T
        x_flat = x.view(N, d)

        gate_weights, expert_idx, router_probs = self.router(x_flat)

        aux_loss = self.cfg.aux_loss_coef * load_balance_loss(
            router_probs, expert_idx, self.cfg.num_experts
        )

        capacity = (
            max(int(self.cfg.capacity_factor * N / self.cfg.num_experts), 1)
            if (self.cfg.capacity_factor is not None and self.training)
            else N
        )

        out_flat = torch.zeros_like(x_flat)
        for expert_id, expert in enumerate(self.experts):
            token_mask, slot_idx = torch.where(expert_idx == expert_id)
            if token_mask.numel() == 0:
                continue
            if token_mask.numel() > capacity:
                token_mask = token_mask[:capacity]
                slot_idx   = slot_idx[:capacity]
            w          = gate_weights[token_mask, slot_idx]
            expert_out = expert(x_flat[token_mask])
            out_flat.index_add_(0, token_mask, expert_out * w.unsqueeze(-1))

        return out_flat.view(b, T, d), aux_loss


# ──────────────────────────────────────────────────────────────────────────────
# Transformer block
# ──────────────────────────────────────────────────────────────────────────────

class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = nn.RMSNorm(cfg.d_model)
        self.gqa   = GroupedQueryAttentionWithRoPE(cfg)
        self.norm2 = nn.RMSNorm(cfg.d_model)
        self.moe   = MoELayer(cfg)

    def forward(
        self,
        x:        torch.Tensor,
        kv_cache: Optional[LayerKVCache] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        residual = x
        x = self.gqa(self.norm1(x), kv_cache=kv_cache)
        x = residual + x

        residual = x
        moe_out, aux_loss = self.moe(self.norm2(x))
        x = residual + moe_out

        return x, aux_loss