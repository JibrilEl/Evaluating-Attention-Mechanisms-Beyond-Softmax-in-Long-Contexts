import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import argparse
from model import SmallVisionTransformer


def get_dataloaders(batch_size=32, data_dir='./data'):
    """Create CIFAR-10 train and validation dataloaders"""
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616)
        )
    ])
    
    train_set = datasets.CIFAR10(
        root=data_dir, 
        train=True, 
        download=True, 
        transform=transform
    )
    val_set = datasets.CIFAR10(
        root=data_dir, 
        train=False, 
        download=True, 
        transform=transform
    )
    
    train_loader = DataLoader(
        train_set, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_set, 
        batch_size=batch_size,
        num_workers=2,
        pin_memory=True
    )
    
    return train_loader, val_loader


@torch.no_grad()
def estimate_loss(model, train_loader, val_loader, device, eval_iters=200):
    """Estimate loss and accuracy on train and validation sets"""
    out = {}
    out_accs = {}
    model.eval()
    
    for split, loader in [('train', train_loader), ('val', val_loader)]:
        losses = []
        accuracies = []
        
        data_iter = iter(loader)
        for k in range(min(eval_iters, len(loader))):
            try:
                X, Y = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                X, Y = next(data_iter)
            
            X, Y = X.to(device), Y.to(device)
            logits, loss = model(X, Y)
            
            predicted_class = torch.argmax(logits, dim=1)
            accuracy = (Y == predicted_class).float().mean()
            
            accuracies.append(accuracy.item())
            losses.append(loss.item())
        
        out[split] = sum(losses) / len(losses)
        out_accs[split] = sum(accuracies) / len(accuracies)
    
    model.train()
    return out, out_accs


def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    torch.manual_seed(args.seed)
    
    train_loader, val_loader = get_dataloaders(
        batch_size=args.batch_size,
        data_dir=args.data_dir
    )
    
    model = SmallVisionTransformer(
        img_size=args.img_size,
        patch_size=args.patch_size,
        in_channels=3,
        num_classes=10,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_layer=args.n_layer,
        dropout=args.dropout,
        head_type=args.head_type
    )
    model = model.to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    loss_list = []
    loss_list_val = []
    acc_list = []
    acc_list_val = []
    
    train_iter = iter(train_loader)
    
    for iteration in range(args.max_iters):
        if iteration % args.eval_interval == 0 or iteration == args.max_iters - 1:
            losses, accs = estimate_loss(
                model, train_loader, val_loader, device, args.eval_iters
            )
            print(f"step {iteration}: "
                  f"train loss {losses['train']:.4f}, train acc {accs['train']:.4f}, "
                  f"val loss {losses['val']:.4f}, val acc {accs['val']:.4f}")
            
            loss_list.append(losses['train'])
            loss_list_val.append(losses['val'])
            acc_list.append(accs['train'])
            acc_list_val.append(accs['val'])

        xb, yb = next(train_iter)
        
        xb, yb = xb.to(device), yb.to(device)
        
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    
    if args.save_path:
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'args': args,
            'train_losses': loss_list,
            'val_losses': loss_list_val,
            'train_accs': acc_list,
            'val_accs': acc_list_val
        }, args.save_path)
        print(f"\nModel saved to {args.save_path}")


def main():
    parser = argparse.ArgumentParser(description='Train a Vision Transformer on CIFAR-10')
    
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Directory for CIFAR-10 data')
    parser.add_argument('--img_size', type=int, default=32,
                        help='Image size (CIFAR-10 is 32x32)')
    parser.add_argument('--patch_size', type=int, default=4,
                        help='Patch size for vision transformer')
    
    parser.add_argument('--n_embd', type=int, default=64,
                        help='Embedding dimension')
    parser.add_argument('--n_head', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--n_layer', type=int, default=3,
                        help='Number of transformer layers')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate')
    parser.add_argument('--head_type', type=str, default='standard',
                        choices=['standard', 'logistic', 'softmax_free'],
                        help='Type of attention head to use')
    
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--max_iters', type=int, default=5000,
                        help='Maximum training iterations')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--eval_interval', type=int, default=100,
                        help='Evaluation interval')
    parser.add_argument('--eval_iters', type=int, default=200,
                        help='Number of iterations for evaluation')
    
    parser.add_argument('--seed', type=int, default=1337,
                        help='Random seed')
    parser.add_argument('--save_path', type=str, default=None,
                        help='Path to save trained model')
    
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
