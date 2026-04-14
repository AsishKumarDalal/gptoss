"""
predict.py
──────────
Interactive text-generation script for the trained MoE-LLM.

Usage
-----
    # One-shot from CLI
    python predict.py --prompt "Once upon a time"

    # Interactive REPL (leave --prompt blank)
    python predict.py

    # Custom checkpoint / sampling params
    python predict.py \\
        --ckpt  checkpoints/step_0010000.pt \\
        --prompt "The little dragon" \\
        --max_new_tokens 300 \\
        --temperature 0.8 \\
        --top_k 50

Options
-------
    --ckpt            path to .pt checkpoint  (default: checkpoints/latest.pt)
    --prompt          input text  (if omitted, enter interactive REPL)
    --max_new_tokens  tokens to generate (default 200)
    --temperature     sampling temperature  1.0 = neutral, <1 sharper, >1 random
    --top_k           restrict to top-k tokens (0 = disabled, i.e. full distribution)
    --device          cpu / cuda / mps  (auto-detected if omitted)
    --dtype           bfloat16 | float16 | float32
"""

import argparse
import sys
from pathlib import Path

import tiktoken
import torch

from config_model import ModelConfig
from model import LLM, generate


# ── helpers ───────────────────────────────────────────────────────────────────

def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate text with the MoE-LLM.")
    p.add_argument("--ckpt",           default="checkpoints/step_0000260.pt")
    p.add_argument("--prompt",         default="",     help="seed text; blank → REPL")
    p.add_argument("--max_new_tokens", type=int,   default=200)
    p.add_argument("--temperature",    type=float, default=0.9)
    p.add_argument("--top_k",          type=int,   default=50,
                   help="top-k sampling (0 = full distribution)")
    p.add_argument("--device",         default="",    help="auto-detect if blank")
    p.add_argument("--dtype",          default="bfloat16",
                   choices=["float32", "float16", "bfloat16"])
    return p.parse_args()


def detect_device(requested: str) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(ckpt_path: Path, device: torch.device, dtype: torch.dtype) -> LLM:
    if not ckpt_path.exists():
        sys.exit(
            f"Checkpoint not found: {ckpt_path}\n"
            "Run  python train.py  first, or pass a valid --ckpt path."
        )

    print(f"Loading checkpoint: {ckpt_path}  …", end=" ", flush=True)
    state = torch.load(ckpt_path, map_location=device)

    # Rebuild config from saved dict so we don't rely on hard-coded defaults
    cfg_dict = state.get("cfg", {})
    cfg      = ModelConfig(**{k: v for k, v in cfg_dict.items()
                               if k in ModelConfig.__dataclass_fields__})

    model = LLM(cfg).to(device=device, dtype=dtype)
    # Strip _orig_mod. prefix produced by torch.compile if present
    raw_sd = {k.replace("_orig_mod.", ""): v for k, v in state["model"].items()}
    model.load_state_dict(raw_sd, strict=True)
    model.eval()
    print("done.")
    print(f"  params : {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M")
    print(f"  step   : {state.get('step', '?')}")
    print(f"  val ℒ  : {state.get('val_loss', float('nan')):.4f}")
    return model, cfg


# ── generation wrapper ────────────────────────────────────────────────────────

def run_generate(
    model:          LLM,
    enc:            tiktoken.Encoding,
    prompt:         str,
    max_new_tokens: int,
    temperature:    float,
    top_k:          int,
    device:         torch.device,
) -> str:
    eos_id = enc.encode_single_token("<|endoftext|>")

    if not prompt.strip():
        # Empty prompt — start from EOS token (standard "unconditional" generation)
        ids = [eos_id]
    else:
        ids = enc.encode_ordinary(prompt)

    prompt_ids = torch.tensor([ids], dtype=torch.long, device=device)   # (1, T)

    with torch.no_grad():
        out_ids = generate(
            model          = model,
            prompt_ids     = prompt_ids,
            max_new_tokens = max_new_tokens,
            temperature    = temperature,
            top_k          = top_k,
            eos_token_id   = eos_id,
        )

    # Decode only the newly generated tokens
    new_ids = out_ids[0, len(ids):].tolist()

    # Stop at EOS if present
    if eos_id in new_ids:
        new_ids = new_ids[:new_ids.index(eos_id)]

    return enc.decode(new_ids)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args   = get_args()
    device = detect_device(args.device)
    dtype  = {"float32": torch.float32,
               "float16": torch.float16,
               "bfloat16": torch.bfloat16}[args.dtype]

    # On CPU or MPS, fall back to float32 silently
    if device.type != "cuda" and dtype != torch.float32:
        print(f"Note: {args.dtype} not supported on {device.type}; using float32.")
        dtype = torch.float32

    ckpt_path = Path(args.ckpt)
    model, cfg = load_model(ckpt_path, device, dtype)

    enc = tiktoken.get_encoding("o200k_base")

    print(f"\nSampling config: max_new_tokens={args.max_new_tokens} "
          f"temperature={args.temperature}  top_k={args.top_k}\n")

    # ── One-shot mode ─────────────────────────────────────────────────────────
    if args.prompt:
        print(f"Prompt: {args.prompt!r}\n")
        output = run_generate(
            model, enc, args.prompt,
            args.max_new_tokens, args.temperature, args.top_k, device,
        )
        print("─" * 60)
        print(args.prompt + output)
        print("─" * 60)
        return

    # ── Interactive REPL ──────────────────────────────────────────────────────
    print("Interactive mode — type a prompt and press Enter.")
    print("Commands:  :q  quit  |  :t <float>  set temperature  |  :k <int>  set top_k\n")

    temperature = args.temperature
    top_k       = args.top_k

    while True:
        try:
            prompt = input("Prompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not prompt:
            continue

        # Meta-commands
        if prompt.startswith(":q"):
            print("Bye!")
            break
        if prompt.startswith(":t "):
            try:
                temperature = float(prompt.split()[1])
                print(f"  temperature → {temperature}")
            except ValueError:
                print("  Usage:  :t <float>")
            continue
        if prompt.startswith(":k "):
            try:
                top_k = int(prompt.split()[1])
                print(f"  top_k → {top_k}")
            except ValueError:
                print("  Usage:  :k <int>")
            continue

        output = run_generate(
            model, enc, prompt,
            args.max_new_tokens, temperature, top_k, device,
        )
        print("─" * 60)
        print(prompt + output)
        print("─" * 60 + "\n")


if __name__ == "__main__":
    main()