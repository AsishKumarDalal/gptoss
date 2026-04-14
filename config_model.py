from dataclasses import dataclass
from typing import Optional

try:
    import tiktoken
    enc = tiktoken.get_encoding("o200k_base")
    _VOCAB_SIZE = enc.n_vocab
except Exception:
    _VOCAB_SIZE = 200_019   # o200k_base vocab size fallback


@dataclass
class ModelConfig:
    # Shared
    d_model:            int   = 1024
    dropout:            float = 0.1
    context_length:     int   = 2048
    vocabulary_size:    int   = _VOCAB_SIZE

    # Attention
    num_heads:          int   = 32
    transformer_blocks: int   = 12
    num_kv_heads:       int   = 8
    qkv_bias:           bool  = False
    use_rope:           bool  = True
    use_attention_bias: bool  = True

    # MoE FFN
    d_ff:               int   = 2048
    num_experts:        int   = 32
    top_k:              int   = 2
    aux_loss_coef:      float = 1e-2
    capacity_factor:    Optional[float] = 1.25

    @property
    def max_seq_len(self) -> int:
        """Alias used by model_parts.py."""
        return self.context_length