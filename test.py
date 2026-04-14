import torch.nn as nn
import torch
from model import LLM
from config_model import ModelConfig
def model_summary(model: nn.Module, dtype=torch.float32) -> dict:
    """
    Print parameter count and estimated memory usage for a PyTorch model.

    Args:
        model  : any nn.Module
        dtype  : dtype used for training (default: float32)
                 e.g. torch.float16 / torch.bfloat16 for half-precision

    Returns:
        dict with keys: total_params, trainable_params,
                        frozen_params, memory_mb
    """
    bytes_per_param = {
        torch.float32: 4,
        torch.float16: 2,
        torch.bfloat16: 2,
        torch.int8: 1,
    }.get(dtype, 4)

    total      = sum(p.numel() for p in model.parameters())
    trainable  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen     = total - trainable

    param_mb   = (total * bytes_per_param) / (1024 ** 2)
    # Rough estimate: params + gradients + optimizer states (Adam ≈ 3× params)
    train_mb   = (trainable * bytes_per_param * 3) / (1024 ** 2)
    total_mb   = param_mb + train_mb

    def fmt(n):
        if n >= 1_000_000_000: return f"{n/1e9:.2f}B"
        if n >= 1_000_000:     return f"{n/1e6:.2f}M"
        if n >= 1_000:         return f"{n/1e3:.2f}K"
        return str(n)

    # ── Per-layer breakdown ─────────────────────────────────────────────────
    rows = []
    for name, module in model.named_modules():
        own = sum(p.numel() for p in module.parameters(recurse=False))
        if own > 0:
            rows.append((name or "(root)", module.__class__.__name__, own))

    col1 = max(len(r[0]) for r in rows) + 2 if rows else 10
    col2 = max(len(r[1]) for r in rows) + 2 if rows else 10

    print("\n" + "═" * 60)
    print(f" MODEL SUMMARY  [{dtype}]")
    print("═" * 60)
    print(f"  {'Layer':<{col1}} {'Type':<{col2}} {'Params':>10}")
    print("─" * 60)
    for name, cls, n in rows:
        print(f"  {name:<{col1}} {cls:<{col2}} {fmt(n):>10}")
    print("─" * 60)
    print(f"  {'Total params':<30} {fmt(total):>10}")
    print(f"  {'Trainable params':<30} {fmt(trainable):>10}")
    print(f"  {'Frozen params':<30} {fmt(frozen):>10}")
    print("─" * 60)
    print(f"  Param memory (weights only)    {param_mb:>8.2f} MB")
    print(f"  Est. training memory (+ Adam)  {total_mb:>8.2f} MB")
    print("═" * 60 + "\n")

    return {
        "total_params":     total,
        "trainable_params": trainable,
        "frozen_params":    frozen,
        "memory_mb":        total_mb,
    }


# ── Example usage ───────────────────────────────────────────────────────
if __name__ == "__main__":

    # Any model — swap in your own
    cfg=ModelConfig()
    model = LLM(cfg)

    stats = model_summary(model, dtype=torch.float32)

    # Also works with half-precision (e.g. ViT, DeiT training)
    # stats = model_summary(model, dtype=torch.float16)