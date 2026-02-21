#!/usr/bin/env python3
"""
Task 3: Needle-in-a-Haystack retrieval.
Train Softmax (a) and SSMax (b), evaluate on (context length × needle depth),
output only the side-by-side heatmap (Context Length × Needle Depth).
"""
import os
import math
import random
import json
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Vocabulary: 27 chars, markers k and q, 24 target classes
FULL_VOCAB = list("abcdefghijklmnopqrstuvwxyz ")
VOCAB_SIZE = len(FULL_VOCAB)
KEY_MARKER = "k"
QUERY_MARKER = "q"
TARGET_CHARS = [c for c in FULL_VOCAB if c not in [KEY_MARKER, QUERY_MARKER, " "]]
NOISE_CHARS = TARGET_CHARS.copy()


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def build_vocab():
    stoi = {ch: i for i, ch in enumerate(FULL_VOCAB)}
    itos = {i: ch for i, ch in enumerate(FULL_VOCAB)}
    def encode(s):
        return [stoi[c] for c in s]
    def decode(lst):
        return "".join(itos[i] for i in lst)
    return stoi, itos, encode, decode


def generate_noise(length):
    if length <= 0:
        return ""
    return "".join(random.choice(NOISE_CHARS) for _ in range(length))


def generate_retrieval_sample(seq_len, needle_depth):
    """Format: [prefix_noise] k [target] [middle_noise] q."""
    target_char = random.choice(TARGET_CHARS)
    min_total = 5
    if seq_len < min_total:
        seq_len = min_total
    min_prefix = 1
    max_prefix = max(min_prefix, seq_len - 4)
    prefix_len = int(min_prefix + needle_depth * (max_prefix - min_prefix))
    prefix_len = max(min_prefix, min(max_prefix, prefix_len))
    middle_len = max(1, seq_len - prefix_len - 3)
    prefix = generate_noise(prefix_len)
    middle = generate_noise(middle_len)
    input_seq = prefix + KEY_MARKER + target_char + middle + QUERY_MARKER
    if len(input_seq) < seq_len:
        input_seq = generate_noise(seq_len - len(input_seq)) + input_seq
    elif len(input_seq) > seq_len:
        input_seq = input_seq[-seq_len:]
    needle_pos = input_seq.find(KEY_MARKER)
    return input_seq, target_char, needle_pos


# ---------------------------------------------------------------------------
# Attention heads and blocks (task3: max_block_size for causal mask)
# ---------------------------------------------------------------------------

class Head(nn.Module):
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
        else:
            mk = lambda: SSMaxHead(n_embd, head_size, block_size, dropout)
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


class RetrievalTransformer(nn.Module):
    """Predict only at last position (after 'q')."""

    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout=0.0, attn_mode="softmax"):
        super().__init__()
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
        logits = self.lm_head(x[:, -1, :])
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def predict(self, idx):
        logits, _ = self.forward(idx)
        return logits.argmax(dim=-1)


# ---------------------------------------------------------------------------
# Data and evaluation
# ---------------------------------------------------------------------------

def get_retrieval_batch(batch_size, seq_len, needle_depth, device):
    stoi, _, encode, _ = build_vocab()
    X_list, Y_list = [], []
    for _ in range(batch_size):
        depth = needle_depth if needle_depth is not None else random.random()
        input_seq, target_char, _ = generate_retrieval_sample(seq_len, depth)
        X_list.append(encode(input_seq))
        Y_list.append(stoi[target_char])
    X = torch.tensor(X_list, dtype=torch.long, device=device)
    Y = torch.tensor(Y_list, dtype=torch.long, device=device)
    return X, Y


@torch.no_grad()
def evaluate_accuracy(model, seq_len, needle_depth, n_samples, batch_size, device):
    model.eval()
    correct, total = 0, 0
    bs = min(batch_size, n_samples)
    for _ in range(n_samples // bs):
        X, Y = get_retrieval_batch(bs, seq_len, needle_depth, device)
        preds = model.predict(X)
        correct += (preds == Y).sum().item()
        total += bs
    model.train()
    return correct / max(total, 1)


def train_one_experiment(experiment, args, device):
    """experiment: 'a' = softmax, 'b' = ssmax. Returns results dict and train_log."""
    stoi, itos, encode, decode = build_vocab()
    attn_mode = "softmax" if experiment == "a" else "ssmax"
    block_size_train = args.block_size_train
    max_block_size = args.max_block_size
    n_embd = args.n_embd
    n_head = args.n_head
    n_layer = args.n_layer
    dropout = args.dropout
    batch_size = args.batch_size
    max_iters = args.max_iters
    eval_interval = args.eval_interval
    learning_rate = args.learning_rate
    eval_lengths = args.eval_lengths
    eval_depths = args.eval_depths
    n_eval_samples = args.n_eval_samples

    model = RetrievalTransformer(
        VOCAB_SIZE, n_embd, n_head, n_layer, max_block_size, dropout, attn_mode
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    train_log = []

    for it in range(max_iters):
        if it % eval_interval == 0 or it == max_iters - 1:
            X_tr, Y_tr = get_retrieval_batch(batch_size, block_size_train, None, device)
            _, loss_tr = model(X_tr, Y_tr)
            acc_train = evaluate_accuracy(
                model, block_size_train, None, 320, batch_size, device
            )
            s_mean = 0.0
            if attn_mode == "ssmax":
                for block in model.blocks:
                    for h in block.sa.heads:
                        if hasattr(h, "s_value"):
                            s_mean += h.s_value().item()
                s_mean /= (n_layer * n_head)
            train_log.append({
                "step": it,
                "loss": loss_tr.item(),
                f"acc@{block_size_train}": acc_train,
                "s_mean": s_mean,
            })
            print(f"  [{experiment}] step {it}: loss={loss_tr.item():.4f} acc@{block_size_train}={acc_train:.2%} s_mean={s_mean:.4f}")

        X, Y = get_retrieval_batch(batch_size, block_size_train, None, device)
        _, loss = model(X, Y)
        if torch.isnan(loss):
            raise RuntimeError(f"NaN loss at step {it}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # Full grid evaluation
    results = {}
    for L in eval_lengths:
        results[L] = {}
        for depth in eval_depths:
            acc = evaluate_accuracy(model, L, depth, n_eval_samples, batch_size, device)
            results[L][depth] = acc
    return {
        "eval_lengths": eval_lengths,
        "eval_depths": eval_depths,
        "results": {str(L): {str(d): results[L][d] for d in eval_depths} for L in eval_lengths},
        "train_log": train_log,
    }


# ---------------------------------------------------------------------------
# Heatmap (only output figure)
# ---------------------------------------------------------------------------

def plot_heatmaps(results_a, results_b, output_path=None):
    """Side-by-side heatmap: Softmax | SSMax (Context Length × Needle Depth)."""
    eval_lengths = results_a["eval_lengths"]
    eval_depths = results_a["eval_depths"]
    acc_matrix_a = np.zeros((len(eval_lengths), len(eval_depths)))
    acc_matrix_b = np.zeros((len(eval_lengths), len(eval_depths)))
    for i, L in enumerate(eval_lengths):
        for j, d in enumerate(eval_depths):
            acc_matrix_a[i, j] = results_a["results"][str(L)][str(d)]
            acc_matrix_b[i, j] = results_b["results"][str(L)][str(d)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    im1 = ax1.imshow(acc_matrix_a, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax1.set_xticks(range(len(eval_depths)))
    ax1.set_xticklabels([f"{d:.2f}" for d in eval_depths])
    ax1.set_yticks(range(len(eval_lengths)))
    ax1.set_yticklabels([str(L) for L in eval_lengths])
    ax1.set_xlabel("Needle Depth")
    ax1.set_ylabel("Context Length")
    ax1.set_title("Softmax")
    for i in range(len(eval_lengths)):
        for j in range(len(eval_depths)):
            val = acc_matrix_a[i, j]
            color = "white" if val < 0.5 else "black"
            ax1.text(j, i, f"{val:.0%}", ha="center", va="center", color=color, fontsize=9)

    im2 = ax2.imshow(acc_matrix_b, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax2.set_xticks(range(len(eval_depths)))
    ax2.set_xticklabels([f"{d:.2f}" for d in eval_depths])
    ax2.set_yticks(range(len(eval_lengths)))
    ax2.set_yticklabels([str(L) for L in eval_lengths])
    ax2.set_xlabel("Needle Depth")
    ax2.set_ylabel("Context Length")
    ax2.set_title("SSMax")
    for i in range(len(eval_lengths)):
        for j in range(len(eval_depths)):
            val = acc_matrix_b[i, j]
            color = "white" if val < 0.5 else "black"
            ax2.text(j, i, f"{val:.0%}", ha="center", va="center", color=color, fontsize=9)

    fig.colorbar(im2, ax=[ax1, ax2], shrink=0.6, label="Accuracy")
    plt.suptitle("Task 3: Retrieval Accuracy — Softmax vs SSMax", fontsize=14)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved heatmap to {output_path}")
    plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Task 3: needle-in-haystack retrieval, output heatmap only")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--block_size_train", type=int, default=128)
    parser.add_argument("--max_block_size", type=int, default=1024)
    parser.add_argument("--max_iters", type=int, default=8000)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--n_embd", type=int, default=128)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--n_layer", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--eval_lengths", type=int, nargs="+", default=[128, 256, 512, 1024])
    parser.add_argument("--eval_depths", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--n_eval_samples", type=int, default=500)
    parser.add_argument("--results_dir", type=str, default=None, help="Where to save task3_results_*.json (default: cwd)")
    parser.add_argument("--output", type=str, default=None, help="Path to save heatmap PNG")
    parser.add_argument("--skip_train", action="store_true", help="Only plot heatmap from existing task3_results_*.json")
    args = parser.parse_args()
    if args.results_dir is None:
        args.results_dir = os.getcwd()

    device = get_device()
    print(f"Using device: {device}")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.skip_train:
        path_a = os.path.join(args.results_dir, "task3_results_a.json")
        path_b = os.path.join(args.results_dir, "task3_results_b.json")
        if not os.path.exists(path_a) or not os.path.exists(path_b):
            print("Run without --skip_train first to generate task3_results_a.json and task3_results_b.json")
            return
        with open(path_a) as f:
            results_a = json.load(f)
        with open(path_b) as f:
            results_b = json.load(f)
        plot_heatmaps(results_a, results_b, args.output)
        return

    # Train a (softmax) then b (ssmax)
    print("Training experiment a (Softmax)...")
    results_a = train_one_experiment("a", args, device)
    path_a = os.path.join(args.results_dir, "task3_results_a.json")
    with open(path_a, "w") as f:
        json.dump(results_a, f, indent=2)
    print(f"Results saved to {path_a}")

    print("Training experiment b (SSMax)...")
    results_b = train_one_experiment("b", args, device)
    path_b = os.path.join(args.results_dir, "task3_results_b.json")
    with open(path_b, "w") as f:
        json.dump(results_b, f, indent=2)
    print(f"Results saved to {path_b}")

    plot_heatmaps(results_a, results_b, args.output)


if __name__ == "__main__":
    main()
