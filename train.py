import os
import zipfile
import urllib.request
import argparse

import torch
from torch.nn import functional as F
from transformers import AutoTokenizer

from model import SmallLanguageModel, BigramLMEmbedPos, BigramLMSinusoidal

TEXT8_URL = "http://mattmahoney.net/dc/text8.zip"
ZIP_PATH = "text8.zip"
TXT_PATH = "text8"


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ---------------------------------------------------------------------------
# LM (Tiny Shakespeare / custom text)
# ---------------------------------------------------------------------------

def get_batch(train_data, val_data, split, batch_size, block_size, device):
    """Generate a batch of data"""
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model, train_data, val_data, batch_size, block_size, device, eval_iters=200):
    """Estimate loss on train and validation sets"""
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(train_data, val_data, split, batch_size, block_size, device)
            logits, loss, latent = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def train_lm(args):
    device = get_device()
    print(f"Using device: {device}")
    torch.manual_seed(args.seed)

    with open(args.data_path, 'r', encoding='utf-8') as f:
        text = f.read()

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    vocab_size = tokenizer.vocab_size

    def encode(text):
        return tokenizer.encode(text, add_special_tokens=True)

    def decode(ids):
        return tokenizer.decode(ids)

    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    model = SmallLanguageModel(
        vocab_size=vocab_size,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_layer=args.n_layer,
        block_size=args.block_size,
        dropout=args.dropout,
        head_type=args.head_type
    )
    model = model.to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    loss_list = []
    loss_list_val = []

    for iter in range(args.max_iters):
        if iter % args.eval_interval == 0 or iter == args.max_iters - 1:
            losses = estimate_loss(
                model, train_data, val_data,
                args.batch_size, args.block_size, device, args.eval_iters
            )
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            loss_list.append(losses['train'].item())
            loss_list_val.append(losses['val'].item())

        xb, yb = get_batch(train_data, val_data, 'train', args.batch_size, args.block_size, device)
        logits, loss, latent = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated = model.generate(context, max_new_tokens=args.gen_tokens)
    print("\nGenerated text:")
    print(decode(generated[0].tolist()))

    if args.save_path:
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'args': args,
            'train_losses': loss_list,
            'val_losses': loss_list_val
        }, args.save_path)
        print(f"\nModel saved to {args.save_path}")


# ---------------------------------------------------------------------------
# Text8 (char-level): helpers
# ---------------------------------------------------------------------------

def ensure_text8(data_dir=None):
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
    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    data = torch.tensor([[chars.index(c) for c in text]], dtype=torch.long).squeeze(0)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    return train_data.to(device), val_data.to(device), vocab_size


# ---------------------------------------------------------------------------
# Text8 task1 (learning curves)
# ---------------------------------------------------------------------------

def train_text8_task1(args):
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
    attn_mode = "softmax" if args.head_type == "standard" else "ssmax"

    def get_batch(split):
        data = train_data if split == "train" else val_data
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i:i + block_size] for i in ix])
        y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
        return x, y

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
# Text8 task2 (length generalization)
# ---------------------------------------------------------------------------

def train_text8_task2(args):
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
    attn_mode = "softmax" if args.head_type == "standard" else "ssmax"

    def get_batch(split, bs):
        data = train_data if split == "train" else val_data
        ix = torch.randint(len(data) - bs - 1, (batch_size,))
        x = torch.stack([data[i:i + bs] for i in ix])
        y = torch.stack([data[i + 1:i + bs + 1] for i in ix])
        return x, y

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
    parser = argparse.ArgumentParser(description="Train language models: lm (e.g. Shakespeare) or text8 (char-level)")
    subparsers = parser.add_subparsers(dest="command", help="Command")

    # lm: Tiny Shakespeare / custom text (GPT-2 tokenizer)
    lm_parser = subparsers.add_parser("lm", help="Train on custom text (e.g. Tiny Shakespeare) with GPT-2 tokenizer")
    lm_parser.add_argument("--data_path", type=str, default="input.txt", help="Path to training data")
    lm_parser.add_argument("--n_embd", type=int, default=64)
    lm_parser.add_argument("--n_head", type=int, default=2)
    lm_parser.add_argument("--n_layer", type=int, default=2)
    lm_parser.add_argument("--block_size", type=int, default=8192)
    lm_parser.add_argument("--dropout", type=float, default=0.0)
    lm_parser.add_argument("--head_type", type=str, default="softmax_free",
                           choices=["softmax_free", "standard", "ssmax", "logistic"])
    lm_parser.add_argument("--batch_size", type=int, default=16)
    lm_parser.add_argument("--max_iters", type=int, default=1000)
    lm_parser.add_argument("--learning_rate", type=float, default=3e-4)
    lm_parser.add_argument("--eval_interval", type=int, default=100)
    lm_parser.add_argument("--eval_iters", type=int, default=200)
    lm_parser.add_argument("--gen_tokens", type=int, default=1000)
    lm_parser.add_argument("--seed", type=int, default=1337)
    lm_parser.add_argument("--save_path", type=str, default=None)

    # text8: char-level task1 or task2
    text8_parser = subparsers.add_parser("text8", help="Train on text8 (char-level), task1 or task2")
    text8_parser.add_argument("task", choices=["task1", "task2"], help="task1: learning curves, task2: length generalization")
    text8_parser.add_argument("--data_dir", type=str, default=None)
    text8_parser.add_argument("--head_type", type=str, default="ssmax", choices=["standard", "ssmax"])
    text8_parser.add_argument("--seed", type=int, default=1337)
    text8_parser.add_argument("--save_path", type=str, default=None)
    text8_parser.add_argument("--block_size", type=int, default=256)
    text8_parser.add_argument("--max_iters", type=int, default=10000)
    text8_parser.add_argument("--eval_interval", type=int, default=200)
    text8_parser.add_argument("--eval_iters", type=int, default=50)
    text8_parser.add_argument("--batch_size", type=int, default=32)
    text8_parser.add_argument("--learning_rate", type=float, default=1e-3)
    text8_parser.add_argument("--n_embd", type=int, default=64)
    text8_parser.add_argument("--n_head", type=int, default=4)
    text8_parser.add_argument("--n_layer", type=int, default=4)
    text8_parser.add_argument("--dropout", type=float, default=0.0)
    text8_parser.add_argument("--block_size_train", type=int, default=256)
    text8_parser.add_argument("--block_sizes_eval", type=int, nargs="+", default=[256, 512, 1024])

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return
    if args.command == "lm":
        train_lm(args)
    elif args.command == "text8":
        if args.data_dir is None:
            args.data_dir = os.path.dirname(os.path.abspath(__file__))
        if args.task == "task1":
            train_text8_task1(args)
        else:
            train_text8_task2(args)


if __name__ == "__main__":
    main()
