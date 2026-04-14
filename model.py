# model.py

import torch
import torch.nn as nn
from typing import List, Optional, Tuple

from config_model import ModelConfig
from model_parts import Transformer, LayerKVCache


class LLM(nn.Module):
    """
    Decoder-only LLM with GQA, RoPE, and Mixture-of-Experts FFN.

    Supports three operating modes via `use_cache`:

    ┌─────────────────┬───────────────────────────────────────────────────┐
    │ Mode            │ How to call                                        │
    ├─────────────────┼───────────────────────────────────────────────────┤
    │ Training        │ logits, aux = model(ids, use_cache=False)          │
    │                 │   — no cache allocated, full causal mask           │
    ├─────────────────┼───────────────────────────────────────────────────┤
    │ Prefill         │ logits, aux = model(prompt_ids, use_cache=True)    │
    │                 │   — allocates cache on first call, full causal     │
    │                 │     mask, fills cache with prompt KVs              │
    ├─────────────────┼───────────────────────────────────────────────────┤
    │ Decode          │ logits, _ = model(next_tok, use_cache=True)        │
    │  (one token)    │   — appends one token to cache each step,          │
    │                 │     no mask needed (cache order = causality)       │
    └─────────────────┴───────────────────────────────────────────────────┘

    Call  model.reset_cache()  between independent generation requests.
    Call  model.build_cache(batch_size, device)  to pre-allocate the cache
    manually before generation (optional — auto-allocated on first use).

    Attributes
    ----------
    kv_caches : List[LayerKVCache] | None
        One LayerKVCache per transformer layer.  None when cache is inactive.

    Returns
    -------
        logits   : (b, T, vocabulary_size)
        aux_loss : scalar  — sum of MoE load-balance losses across all layers
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.embed    = nn.Embedding(cfg.vocabulary_size, cfg.d_model)
        self.drop     = nn.Dropout(cfg.dropout)
        self.layers   = nn.ModuleList(
            [Transformer(cfg) for _ in range(cfg.transformer_blocks)]
        )
        self.norm     = nn.RMSNorm(cfg.d_model)
        self.out_head = nn.Linear(cfg.d_model, cfg.vocabulary_size)

        # Cache — None until build_cache() is called or use_cache=True triggers
        # auto-allocation on the first forward pass.
        self.kv_caches: Optional[List[LayerKVCache]] = None

    # ── Cache lifecycle ────────────────────────────────────────────────────────

    def build_cache(
        self,
        batch_size: int,
        device:     torch.device,
        dtype:      torch.dtype = torch.float32,
    ) -> None:
        """
        Pre-allocate KV caches for all layers.

        Called automatically by forward() when use_cache=True and the cache
        has not yet been built.  You can call it explicitly beforehand to
        control dtype or to reset an existing cache with a new batch size.

        Parameters
        ----------
        batch_size : int
        device     : torch.device
        dtype      : torch.dtype   default float32; use bfloat16 to halve memory
        """
        head_dim = self.cfg.d_model // self.cfg.num_heads
        self.kv_caches = [
            LayerKVCache(
                batch_size   = batch_size,
                num_kv_heads = self.cfg.num_kv_heads,
                max_seq_len  = self.cfg.max_seq_len,
                head_dim     = head_dim,
                device       = device,
                dtype        = dtype,
            )
            for _ in range(self.cfg.transformer_blocks)
        ]

    def reset_cache(self) -> None:
        """
        Zero all cache buffers and reset fill pointers to 0.
        Call between independent generation requests with the same batch size.
        Does nothing if no cache has been built yet.
        """
        if self.kv_caches is not None:
            for cache in self.kv_caches:
                cache.reset()

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids: torch.Tensor,          # (b, T)
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        input_ids : (b, T)  — token ids
        use_cache : bool
            False → training / eval without cache (full causal mask, T tokens).
            True  → generation; cache is auto-allocated on first call if needed.

        Returns
        -------
        logits   : (b, T, vocabulary_size)
        aux_loss : scalar tensor — sum of per-layer MoE load-balance losses
        """
        b, T = input_ids.shape

        # ── Auto-allocate cache on first generation call ───────────────────
        if use_cache and self.kv_caches is None:
            self.build_cache(
                batch_size = b,
                device     = input_ids.device,
                dtype      = self.embed.weight.dtype,
            )

        # ── Embedding + dropout ───────────────────────────────────────────────
        x = self.drop(self.embed(input_ids))   # (b, T, d_model)

        # ── Transformer layers ────────────────────────────────────────────────
        total_aux_loss = torch.tensor(0.0, device=input_ids.device)

        for i, layer in enumerate(self.layers):
            cache = self.kv_caches[i] if use_cache else None
            x, aux_loss = layer(x, kv_cache=cache)
            total_aux_loss = total_aux_loss + aux_loss

        # ── Output projection ─────────────────────────────────────────────────
        x      = self.norm(x)
        logits = self.out_head(x)   # (b, T, vocabulary_size)

        return logits, total_aux_loss


# ──────────────────────────────────────────────────────────────────────────────
# Greedy / sampling generation helper
# ──────────────────────────────────────────────────────────────────────────────

@torch.inference_mode()
def generate(
    model:          LLM,
    prompt_ids:     torch.Tensor,       # (b, T_prompt)
    max_new_tokens: int,
    temperature:    float = 1.0,
    top_k:          int   = 0,          # 0 = disabled
    eos_token_id:   Optional[int] = None,
) -> torch.Tensor:
    """
    Autoregressive generation using KV cache.

    Steps
    -----
    1. Prefill  — run the full prompt through the model to populate the cache.
    2. Decode   — feed one token at a time, appending to the cache each step.

    Parameters
    ----------
    model          : LLM (will be set to eval mode internally)
    prompt_ids     : (b, T_prompt)
    max_new_tokens : int
    temperature    : float   > 1 = more random, < 1 = sharper
    top_k          : int     if > 0, restrict sampling to top-k logits
    eos_token_id   : int | None   stop early when all sequences emit EOS

    Returns
    -------
    generated : (b, T_prompt + max_new_tokens)  — prompt + new tokens
    """
    model.eval()
    model.reset_cache()

    device    = prompt_ids.device
    generated = prompt_ids.clone()         # (b, T_prompt)

    # ── Prefill ────────────────────────────────────────────────────────────
    # Process the whole prompt; keep only the last-position logits.
    logits, _ = model(prompt_ids, use_cache=True)   # (b, T_prompt, V)
    next_logits = logits[:, -1, :]                  # (b, V)

    # ── Decode loop ────────────────────────────────────────────────────────
    for _ in range(max_new_tokens):

        # Sample / greedy from last-position logits
        next_token = _sample(next_logits, temperature=temperature, top_k=top_k)  # (b, 1)
        generated  = torch.cat([generated, next_token], dim=1)

        # Early stopping
        if eos_token_id is not None and (next_token == eos_token_id).all():
            break

        # Single-step decode — append one token, read back logits
        logits, _   = model(next_token, use_cache=True)   # (b, 1, V)
        next_logits = logits[:, -1, :]                    # (b, V)

    return generated


def _sample(
    logits:      torch.Tensor,   # (b, V)
    temperature: float,
    top_k:       int,
) -> torch.Tensor:               # (b, 1)
    """Temperature + optional top-k sampling. Returns token ids (b, 1)."""
    if temperature != 1.0:
        logits = logits / temperature

    if top_k > 0:
        # Zero out all logits below the k-th largest
        values, _ = torch.topk(logits, top_k, dim=-1)
        threshold  = values[:, -1].unsqueeze(-1)
        logits     = logits.masked_fill(logits < threshold, float("-inf"))

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)   # (b, 1)