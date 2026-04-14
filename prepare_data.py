"""
prepare_data.py
───────────────
Downloads TinyStories from HuggingFace, tokenises with tiktoken (o200k_base),
and writes binary shards to  data/train/  and  data/val/.

Each shard is a flat uint32 numpy array saved as  shard_NNNN.npy.
A tiny metadata file  data/{split}/meta.json  records total token count
and number of shards so the DataLoader never has to glob at runtime.

Usage
-----
    python prepare_data.py                      # defaults below
    python prepare_data.py --shard_size 5e7     # 50 M tokens per shard
    python prepare_data.py --val_ratio 0.005    # 0.5 % held out for val
    python prepare_data.py --num_proc 8         # more tokenisation workers

Output layout
-------------
    data/
        train/
            shard_0000.npy
            shard_0001.npy
            ...
            meta.json          {"num_shards": N, "total_tokens": T}
        val/
            shard_0000.npy
            meta.json
"""

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

# ── defaults ──────────────────────────────────────────────────────────────────
DEFAULT_SHARD_SIZE = int(1e8)   # 100 M tokens per shard  (~400 MB uint32)
DEFAULT_VAL_RATIO  = 0.01       # 1 % of stories → validation
DEFAULT_NUM_PROC   = max(1, os.cpu_count() // 2)
DATA_ROOT          = Path("data")
EOS_TOKEN          = "<|endoftext|>"


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tokenise TinyStories into binary shards.")
    p.add_argument("--shard_size", type=float, default=DEFAULT_SHARD_SIZE,
                   help="tokens per shard (default 1e8)")
    p.add_argument("--val_ratio",  type=float, default=DEFAULT_VAL_RATIO,
                   help="fraction of stories held out for validation")
    p.add_argument("--num_proc",   type=int,   default=DEFAULT_NUM_PROC,
                   help="tokenisation worker processes")
    return p.parse_args()


# ── tokenisation ──────────────────────────────────────────────────────────────

def make_tokenise_fn(enc: tiktoken.Encoding):
    eos_id = enc.encode_single_token(EOS_TOKEN)

    def tokenise(example: dict) -> dict:
        ids = enc.encode_ordinary(example["text"])
        ids.append(eos_id)            # story boundary
        return {"ids": ids, "len": len(ids)}

    return tokenise


# ── shard writer ──────────────────────────────────────────────────────────────

def write_shards(
    dataset,
    split_dir:  Path,
    shard_size: int,
) -> dict:
    """Stream tokens from `dataset` into fixed-size uint32 shards."""
    split_dir.mkdir(parents=True, exist_ok=True)

    total_tokens = 0
    shard_idx    = 0
    buf          = np.empty(shard_size, dtype=np.uint32)
    ptr          = 0

    def flush(buf, n, idx):
        out = split_dir / f"shard_{idx:04d}.npy"
        np.save(out, buf[:n])
        return out

    saved = []
    for example in tqdm(dataset, desc=f"  writing {split_dir.name}", unit="story"):
        ids = example["ids"]
        arr = np.array(ids, dtype=np.uint32)

        # might span multiple shards
        offset = 0
        while offset < len(arr):
            space   = shard_size - ptr
            chunk   = arr[offset : offset + space]
            buf[ptr : ptr + len(chunk)] = chunk
            ptr    += len(chunk)
            offset += len(chunk)

            if ptr == shard_size:
                saved.append(flush(buf, ptr, shard_idx))
                total_tokens += ptr
                shard_idx += 1
                ptr = 0

    # final partial shard
    if ptr > 0:
        saved.append(flush(buf, ptr, shard_idx))
        total_tokens += ptr
        shard_idx += 1

    meta = {"num_shards": shard_idx, "total_tokens": total_tokens}
    (split_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = get_args()
    shard_size = int(args.shard_size)

    print("Loading TinyStories from HuggingFace …")
    ds = load_dataset("roneneldan/TinyStories", split="train")

    # train / val split
    split = ds.train_test_split(test_size=args.val_ratio, seed=42)
    train_ds, val_ds = split["train"], split["test"]

    print(f"  train stories : {len(train_ds):,}")
    print(f"  val   stories : {len(val_ds):,}")

    enc = tiktoken.get_encoding("o200k_base")
    tokenise = make_tokenise_fn(enc)

    print("Tokenising (this takes a few minutes) …")
    train_tok = train_ds.map(
        tokenise, remove_columns=train_ds.column_names,
        num_proc=args.num_proc, desc="train tokenise"
    )
    val_tok = val_ds.map(
        tokenise, remove_columns=val_ds.column_names,
        num_proc=args.num_proc, desc="val tokenise"
    )

    print("Writing train shards …")
    train_meta = write_shards(train_tok, DATA_ROOT / "train", shard_size)
    print("Writing val shards …")
    val_meta   = write_shards(val_tok,   DATA_ROOT / "val",   shard_size)

    print("\n✓  Done!")
    print(f"   Train : {train_meta['total_tokens']:,} tokens  |  {train_meta['num_shards']} shards")
    print(f"   Val   : {val_meta['total_tokens']:,} tokens  |  {val_meta['num_shards']} shards")
    print(f"   Saved to  {DATA_ROOT.resolve()}/")


if __name__ == "__main__":
    main()