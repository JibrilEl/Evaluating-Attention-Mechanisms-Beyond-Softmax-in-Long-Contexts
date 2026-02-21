#!/usr/bin/env python3
"""
Train language models on text8 (char-level) for SSMax vs Softmax comparison.
Subcommands: task1 (learning curves), task2 (length generalization).
"""
import os
import math
import argparse
import zipfile
import urllib.request

import torch
import torch.nn as nn
from torch.nn import functional as F

TEXT8_URL = "http://mattmahoney.net/dc/text8.zip"
ZIP_PATH = "text8.zip"
TXT_PATH = "text8"


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def ensure_text8(data_dir=None):
    """Download and extract text8 if not present."""
    base = data_dir or os.path.dirname(os.path.abspath(__file__))
    txt_path = os.path.join(base, TXT_PATH)
    zip_path = os.path.join(base, ZIP_PATH)
    if os.path.exists(txt_path):
        return txt_path
    if not os.path.exists(zip_path):
        print("Downloading text8.zip ...")
        urllib.request.urlretrieve(TEXT8_URL, zip_path)
        print("Done.")
    print("Extracting text8 ...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(base)
    print("Extracted.")
    return txt_path


def load_text8(data_path, device):
    """Load text8, build char vocab, return train/val tensors and vocab size."""
    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    def encode(s):
        return [stoi[c] for c in s]

    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    return train_data, val_data, vocab_size


# ---------------------------------------------------------------------------
# Attention heads (shared)
# ---------------------------------------------------------------------------

class Head(nn.Module):
    """Vanilla self-attention head (softmax)."""

    def __init__(self, n_embd, head_size, block_size, dropout=0.0):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, _ = x.shape
        k = self.key(x)
        q = self.query(x)
        scores = (q @ k.transpose(-2, -1)) * (k.shape[-1] ** -0.5)
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        w = F.softmax(scores, dim=-1)
        w = self.dropout(w)
        v = self.value(x)
        return w @ v


class SSMaxHead(nn.Module):
    """SSMax head (learnable s). Scale q before matmul for stability."""

    def __init__(self, n_embd, head_size, block_size, dropout=0.0):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
        self.s_raw = nn.Parameter(torch.tensor(0.0))

    def s_value(self):
        return F.softplus(self.s_raw)

    def forward(self, x):
        B, T, _ = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        n = torch.arange(1, T + 1, device=x.device, dtype=x.dtype)
        logn = torch.log(n).view(1, T, 1)
        q_scaled = q * (self.s_value() * logn)
        scores = (q_scaled @ k.transpose(-2, -1)) * (k.shape[-1] ** -0.5)
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        w = F.softmax(scores, dim=-1)
        w = self.dropout(w)
        return w @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout=0.0, attn_mode="softmax"):
        super().__init__()
        head_size = n_embd // n_head
        if attn_mode == "softmax":
            mk = lambda: Head(n_embd, head_size, block_size, dropout)
        elif attn_mode == "ssmax":
            mk = lambda: SSMaxHead(n_embd, head_size, block_size, dropout)
        else:
            raise ValueError(attn_mode)
        self.heads = nn.ModuleList([mk() for _ in range(n_head)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout=0.0, attn_mode="softmax"):
        super().__init__()
        self.sa = MultiHeadAttention(n_embd, n_head, block_size, dropout, attn_mode)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


def sinusoidal_pos_emb(T, C, device):
    position = torch.arange(T, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, C, 2, device=device) * (-math.log(10000.0) / C))
    pe = torch.zeros(T, C, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


# ---------------------------------------------------------------------------
# Task1: fixed block_size, learned position embedding
# ---------------------------------------------------------------------------

class BigramLMEmbedPos(nn.Module):
    """LM with learned position embedding (fixed block_size). For task1."""

    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout=0.0, attn_mode="softmax"):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[
            Block(n_embd, n_head, block_size, dropout, attn_mode) for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))
        return logits, loss


# ---------------------------------------------------------------------------
# Task2: variable length, sinusoidal position
# ---------------------------------------------------------------------------

class BigramLMSinusoidal(nn.Module):
    """LM with sinusoidal position (supports any T up to block_size). For task2."""

    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout=0.0, attn_mode="softmax"):
        super().__init__()
        self.block_size = block_size
        self.n_embd = n_embd
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.blocks = nn.Sequential(*[
            Block(n_embd, n_head, block_size, dropout, attn_mode) for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = sinusoidal_pos_emb(T, self.n_embd, idx.device)
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))
        return logits, loss


# ---------------------------------------------------------------------------
# Task1 entry
# ---------------------------------------------------------------------------

def run_task1(args):
    device = get_device()
    print(f"Using device: {device}")
    torch.manual_seed(args.seed)

    data_path = ensure_text8(args.data_dir)
    train_data, val_data, vocab_size = load_text8(data_path, device)

    block_size = args.block_size
    n_embd = args.n_embd
    n_head = args.n_head
    n_layer = args.n_layer
    dropout = args.dropout
    batch_size = args.batch_size
    max_iters = args.max_iters
    eval_interval = args.eval_interval
    learning_rate = args.learning_rate
    eval_iters = args.eval_iters
    head_type = args.head_type  # "standard" or "ssmax"

    attn_mode = "softmax" if head_type == "standard" else "ssmax"

    def get_batch(split):
        data = train_data if split == "train" else val_data
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i : i + block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
        return x.to(device), y.to(device)

    @torch.no_grad()
    def estimate_loss(model):
        out = {}
        model.eval()
        for split in ["train", "val"]:
            losses = torch.zeros(eval_iters, device=device)
            for k in range(eval_iters):
                X, Y = get_batch(split)
                _, loss = model(X, Y)
                losses[k] = loss
            out[split] = losses.mean().item()
        model.train()
        return out

    model = BigramLMEmbedPos(
        vocab_size, n_embd, n_head, n_layer, block_size, dropout, attn_mode
    ).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.4f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    train_log = []

    for it in range(max_iters):
        if it % eval_interval == 0 or it == max_iters - 1:
            losses = estimate_loss(model)
            train_log.append({"step": it, "train": losses["train"], "val": losses["val"]})
            print(f"step {it}: train {losses['train']:.4f}, val {losses['val']:.4f}")

        xb, yb = get_batch("train")
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    if args.save_path:
        torch.save({
            "model_state_dict": model.state_dict(),
            "args": args,
            "train_log": train_log,
        }, args.save_path)
        print(f"Saved to {args.save_path}")


# ---------------------------------------------------------------------------
# Task2 entry
# ---------------------------------------------------------------------------

def run_task2(args):
    device = get_device()
    print(f"Using device: {device}")
    torch.manual_seed(args.seed)

    data_path = ensure_text8(args.data_dir)
    train_data, val_data, vocab_size = load_text8(data_path, device)

    block_size_train = args.block_size_train
    block_sizes_eval = args.block_sizes_eval
    max_block_size = max(block_sizes_eval)
    n_embd = args.n_embd
    n_head = args.n_head
    n_layer = args.n_layer
    dropout = args.dropout
    batch_size = args.batch_size
    max_iters = args.max_iters
    eval_interval = args.eval_interval
    learning_rate = args.learning_rate
    eval_iters = args.eval_iters
    head_type = args.head_type
    attn_mode = "softmax" if head_type == "standard" else "ssmax"

    def get_batch(split, bs):
        data = train_data if split == "train" else val_data
        ix = torch.randint(len(data) - bs - 1, (batch_size,))
        x = torch.stack([data[i : i + bs] for i in ix])
        y = torch.stack([data[i + 1 : i + bs + 1] for i in ix])
        return x.to(device), y.to(device)

    @torch.no_grad()
    def estimate_loss_for_block(model, split, bs):
        model.eval()
        losses = torch.zeros(eval_iters, device=device)
        for k in range(eval_iters):
            X, Y = get_batch(split, bs)
            _, loss = model(X, Y)
            losses[k] = loss
        model.train()
        return losses.mean().item()

    model = BigramLMSinusoidal(
        vocab_size, n_embd, n_head, n_layer, max_block_size, dropout, attn_mode
    ).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.4f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    train_log = []

    for it in range(max_iters):
        if it % eval_interval == 0 or it == max_iters - 1:
            val_trainlen = estimate_loss_for_block(model, "val", block_size_train)
            vals = {L: estimate_loss_for_block(model, "val", L) for L in block_sizes_eval}
            train_loss_one = estimate_loss_for_block(model, "train", block_size_train)
            train_log.append({
                "step": it,
                "train": train_loss_one,
                "val_trainlen": val_trainlen,
                **{f"val@{L}": vals[L] for L in block_sizes_eval},
            })
            msg = f"step {it}: train {train_loss_one:.4f} | val@{block_size_train} {val_trainlen:.4f}"
            for L in block_sizes_eval:
                msg += f" | val@{L} {vals[L]:.4f}"
            print(msg)

        xb, yb = get_batch("train", block_size_train)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    if args.save_path:
        torch.save({
            "model_state_dict": model.state_dict(),
            "args": args,
            "train_log": train_log,
        }, args.save_path)
        print(f"Saved to {args.save_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train on text8 (task1 or task2)")
    parser.add_argument("task", choices=["task1", "task2"], help="task1: learning curves, task2: length generalization")
    parser.add_argument("--data_dir", type=str, default=None, help="Directory for text8 (default: script dir)")
    parser.add_argument("--head_type", type=str, default="ssmax", choices=["standard", "ssmax"])
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--save_path", type=str, default=None)

    # task1 defaults
    parser.add_argument("--block_size", type=int, default=256, help="Context length (task1)")
    parser.add_argument("--max_iters", type=int, default=10000)
    parser.add_argument("--eval_interval", type=int, default=200)
    parser.add_argument("--eval_iters", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--n_embd", type=int, default=64)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--n_layer", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)

    # task2-only
    parser.add_argument("--block_size_train", type=int, default=256)
    parser.add_argument("--block_sizes_eval", type=int, nargs="+", default=[256, 512, 1024])

    args = parser.parse_args()
    if args.data_dir is None:
        args.data_dir = os.path.dirname(os.path.abspath(__file__))

    if args.task == "task1":
        run_task1(args)
    else:
        run_task2(args)


if __name__ == "__main__":
    main()
