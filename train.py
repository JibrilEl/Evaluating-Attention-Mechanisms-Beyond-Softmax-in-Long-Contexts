import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import AutoTokenizer
import argparse
from model import SmallLanguageModel


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


def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
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


def main():
    parser = argparse.ArgumentParser(description='Train a small language model')
    
    parser.add_argument('--data_path', type=str, default='input.txt',
                        help='Path to training data')
    
    parser.add_argument('--n_embd', type=int, default=64,
                        help='Embedding dimension')
    parser.add_argument('--n_head', type=int, default=2,
                        help='Number of attention heads')
    parser.add_argument('--n_layer', type=int, default=2,
                        help='Number of transformer layers')
    parser.add_argument('--block_size', type=int, default=8192,
                        help='Maximum context length')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout rate')
    parser.add_argument('--head_type', type=str, default='softmax_free',
                        choices=['softmax_free', 'standard', 'ssmax', 'logistic'],
                        help='Type of attention head to use')
    
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--max_iters', type=int, default=1000,
                        help='Maximum training iterations')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--eval_interval', type=int, default=100,
                        help='Evaluation interval')
    parser.add_argument('--eval_iters', type=int, default=200,
                        help='Number of iterations for evaluation')
    
    parser.add_argument('--gen_tokens', type=int, default=1000,
                        help='Number of tokens to generate after training')
    
    parser.add_argument('--seed', type=int, default=1337,
                        help='Random seed')
    parser.add_argument('--save_path', type=str, default=None,
                        help='Path to save trained model')
    
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
