import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from transformers import AutoTokenizer
from torchvision import datasets, transforms
import numpy as np
from model import SmallLanguageModel, SmallVisionTransformer


def _ensure_output_dir(path):
    if path and os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)


def plot_cifar10_attention(model, data_loader, device, layer=0, head=0, 
                           num_samples=4, output_path='cifar10_attention.png'):
    """Plot attention weights for CIFAR-10 images"""
    
    model.eval()
    
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
    
    with torch.no_grad():
        
        images, labels = next(iter(data_loader))
        images = images[:num_samples].to(device)
        labels = labels[:num_samples]
        
        # Forward pass with attention hooks
        logits, loss, attentions = model(images, return_hook=True)
        predictions = torch.argmax(logits, dim=1)
        
        # attentions[layer]: (B, num_heads, T, T)
        attn_map = attentions[layer][:, head, 0, 1:].cpu()  # (B, num_patches)
        
        # Get number of patches per side
        num_patches = int(np.sqrt(attn_map.shape[1]))
        
        fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 3, 6))
        
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
        std = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)
        images_denorm = images.cpu() * std + mean
        images_denorm = torch.clamp(images_denorm, 0, 1)
        
        for i in range(num_samples):
            axes[0, i].imshow(images_denorm[i].permute(1, 2, 0))
            axes[0, i].set_title(f'True: {classes[labels[i]]}\nPred: {classes[predictions[i]]}')
            axes[0, i].axis('off')
            
            # Plot attention map
            attn_reshaped = attn_map[i].reshape(num_patches, num_patches)
            im = axes[1, i].imshow(attn_reshaped, cmap='viridis', interpolation='nearest')
            axes[1, i].set_title(f'Attention (L{layer}, H{head})')
            axes[1, i].axis('off')
            plt.colorbar(im, ax=axes[1, i], fraction=0.046)
        
        plt.tight_layout()
        _ensure_output_dir(output_path)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"CIFAR-10 attention visualization saved to {output_path}")
        plt.close()


def plot_cifar10_training_curves(checkpoint_path, output_path='cifar10_training.png'):
    """Plot training curves for CIFAR-10"""
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    train_losses = checkpoint.get('train_losses', [])
    val_losses = checkpoint.get('val_losses', [])
    train_accs = checkpoint.get('train_accs', [])
    val_accs = checkpoint.get('val_accs', [])
    
    if not train_losses:
        print("No training history found in checkpoint")
        return
    
    epochs = range(len(train_losses))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    ax1.plot(epochs, train_losses, marker='o', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_losses, marker='s', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Evaluation Steps')
    ax1.set_ylabel('Cross Entropy Loss')
    ax1.set_title('CIFAR-10 Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    if train_accs and val_accs:
        ax2.plot(epochs, train_accs, marker='o', label='Training Accuracy', linewidth=2)
        ax2.plot(epochs, val_accs, marker='s', label='Validation Accuracy', linewidth=2)
        ax2.set_xlabel('Evaluation Steps')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('CIFAR-10 Training Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    _ensure_output_dir(output_path)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"CIFAR-10 training curves saved to {output_path}")
    plt.close()


def plot_attention_heatmap(model, data, tokenizer, device, layer=0, head=0, output_path='attention_heatmap.png'):
    """Plot attention weights as a heatmap"""
    
    model.eval()
    with torch.no_grad():
        sentence = data[:1]  # Take first sequence from batch
        B, T = sentence.shape
        
        tok_emb = model.token_embedding_table(sentence)
        pos_emb = model.position_embedding_table(torch.arange(T, device=device))
        embedded = tok_emb + pos_emb
        
        output, hook = model.blocks[layer].sa.heads[head](embedded, return_hook=True)
        
        # Convert tokens to readable format
        tokens = tokenizer.convert_ids_to_tokens(sentence[0])
        replaced = [t.replace("Ġ", " ") for t in tokens]
        
        # Plot heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(hook.detach().cpu()[0], cmap='viridis', square=True)
        plt.title(f'Attention Weights - Layer {layer}, Head {head}')
        plt.xlabel('Key Position')
        plt.ylabel('Query Position')
        
        if len(replaced) <= 50:
            plt.xticks(range(len(replaced)), replaced, rotation=90, fontsize=8)
            plt.yticks(range(len(replaced)), replaced, fontsize=8)
        
        plt.tight_layout()
        _ensure_output_dir(output_path)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Attention heatmap saved to {output_path}")
        plt.close()


def plot_benchmark_results(results_path='benchmark_results.json', output_path='benchmark_plot.png'):
    """Plot benchmark results from JSON file"""
    import json
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    context_lengths = results['context_lengths']
    
    plt.figure(figsize=(10, 6))
    plt.title("CPU Inference Speed of a Single Head\n(Batch Size = 2)")
    plt.xlabel("Context Length")
    plt.ylabel("Average Inference Time (seconds)")
    
    plt.plot(context_lengths, results['vanilla'], marker='o', label="Softmax", linewidth=2)
    plt.plot(context_lengths, results['logistic'], marker='s', label="Element-wise Sigmoid", linewidth=2)
    plt.plot(context_lengths, results['ssmax'], marker='^', label="SSMax", linewidth=2)
    plt.plot(context_lengths, results['sima'], marker='d', label="SimA", linewidth=2)
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    _ensure_output_dir(output_path)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Benchmark plot saved to {output_path}")
    plt.close()


def plot_training_curves(checkpoint_path, output_path='training_curves.png'):
    """Plot training and validation loss curves"""
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    train_losses = checkpoint.get('train_losses', [])
    val_losses = checkpoint.get('val_losses', [])
    
    if not train_losses or not val_losses:
        print("No training history found in checkpoint")
        return
    
    epochs = range(len(train_losses))
    
    plt.figure(figsize=(10, 6))
    plt.xlabel("Evaluation Steps")
    plt.ylabel("Cross Entropy Loss")
    plt.plot(epochs, train_losses, marker='o', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, marker='s', label='Validation Loss', linewidth=2)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    _ensure_output_dir(output_path)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Training curves saved to {output_path}")
    plt.close()


def plot_text8_task1_curves(checkpoint_path, output_path='text8_task1_curves.png', checkpoint_b=None):
    """
    Plot Task 1 learning curves from train_text8.py checkpoints.
    Uses train_log produced by train_text8.py task1 (list of {step, train, val}).
    - Single run: one checkpoint -> train and val loss vs step (like notebook single plot).
    - Comparison: two checkpoints -> validation loss only, Softmax vs SSMax (like notebook comparison).
    """
    def load_log(path):
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        log = ckpt.get('train_log', [])
        if not log:
            raise ValueError(f"No 'train_log' in {path}. Run train_text8.py task1 with --save_path first.")
        return log, getattr(ckpt.get('args'), 'head_type', None)

    if checkpoint_b is None:
        # Single run: train + val
        log, head_type = load_log(checkpoint_path)
        steps = [d['step'] for d in log]
        train_loss = [d['train'] for d in log]
        val_loss = [d['val'] for d in log]
        label = "SSMax" if head_type == 'ssmax' else "Softmax"
        plt.figure(figsize=(7, 5))
        plt.plot(steps, train_loss, label='train', linewidth=2)
        plt.plot(steps, val_loss, label='val', linewidth=2)
        plt.xlabel('step')
        plt.ylabel('loss')
        plt.title(f'text8 pretraining — {label}')
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        # Comparison: validation only, Softmax vs SSMax
        log_a, head_a = load_log(checkpoint_path)
        log_b, head_b = load_log(checkpoint_b)
        steps_a = [d['step'] for d in log_a]
        val_a = [d['val'] for d in log_a]
        steps_b = [d['step'] for d in log_b]
        val_b = [d['val'] for d in log_b]
        label_a = 'Softmax' if (head_a == 'standard' or head_a is None) else 'SSMax'
        label_b = 'Softmax' if (head_b == 'standard' or head_b is None) else 'SSMax'
        plt.figure(figsize=(7, 5))
        plt.plot(steps_a, val_a, label=label_a, linewidth=2)
        plt.plot(steps_b, val_b, label=label_b, linewidth=2)
        plt.xlabel('Training step')
        plt.ylabel('Validation loss')
        plt.title('text8 pretraining — Softmax vs SSMax')
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    _ensure_output_dir(output_path)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Text8 Task1 curves saved to {output_path}")
    plt.close()


def _task2_log_lengths(log):
    """Infer eval lengths from Task2 train_log keys (e.g. val@256, val@512, val@1024)."""
    if not log:
        return []
    keys = [k for k in log[0].keys() if isinstance(k, str) and k.startswith('val@')]
    return sorted([int(k.split('@')[1]) for k in keys])


def plot_text8_task2_curves(
    checkpoint_path,
    output_path='text8_task2_curves.png',
    output_final=None,
    checkpoint_b=None,
):
    """
    Plot Task 2 length generalization from train_text8.py task2 checkpoints.
    Uses train_log with step, train, val_trainlen, val@256, val@512, val@1024.
    Produces two figures (like the last two plots of the Task 2 notebook):
    1. Validation loss vs training step for each eval length (256, 512, 1024), Softmax vs SSMax if two checkpoints.
    2. Final length generalization: validation loss vs context length at last step (Softmax vs SSMax if two checkpoints).
    """
    def load_log(path):
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        log = ckpt.get('train_log', [])
        if not log:
            raise ValueError(f"No 'train_log' in {path}. Run train_text8.py task2 with --save_path first.")
        lengths = _task2_log_lengths(log)
        if not lengths:
            raise ValueError(f"Task2 train_log must contain val@L keys. Got keys: {list(log[0].keys())}")
        return log, lengths, getattr(ckpt.get('args'), 'head_type', None)

    if output_final is None and output_path:
        base = output_path.rsplit('.', 1)
        output_final = f"{base[0]}_final_gen.{base[1]}" if len(base) == 2 else f"{output_path}_final_gen.png"

    if checkpoint_b is None:
        # Single run
        log, lengths, head_type = load_log(checkpoint_path)
        steps = [d['step'] for d in log]
        label_run = 'SSMax' if head_type == 'ssmax' else 'Softmax'

        # Plot 1: val loss vs step for each length
        plt.figure(figsize=(8, 5))
        for L in lengths:
            plt.plot(steps, [d[f'val@{L}'] for d in log], label=f'val@{L}', linewidth=2)
        plt.xlabel('training step')
        plt.ylabel('validation loss')
        plt.title(f'Task 2 — Length generalization ({label_run})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        _ensure_output_dir(output_path)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Text8 Task2 curves saved to {output_path}")
        plt.close()

        # Plot 2: final length generalization (one curve)
        last = log[-1]
        vals = [last[f'val@{L}'] for L in lengths]
        plt.figure(figsize=(6, 4))
        plt.plot(lengths, vals, marker='o', label=label_run, linewidth=2)
        plt.xlabel('context length (eval)')
        plt.ylabel('validation loss')
        plt.title(f'Task 2 — Final length generalization ({label_run})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        _ensure_output_dir(output_final)
        plt.savefig(output_final, dpi=150, bbox_inches='tight')
        print(f"Text8 Task2 final length gen saved to {output_final}")
        plt.close()
        return

    # Two checkpoints: Softmax vs SSMax
    log_a, lengths, head_a = load_log(checkpoint_path)
    log_b, _, head_b = load_log(checkpoint_b)
    label_a = 'Softmax' if (head_a == 'standard' or head_a is None) else 'SSMax'
    label_b = 'Softmax' if (head_b == 'standard' or head_b is None) else 'SSMax'
    steps_a = [d['step'] for d in log_a]
    steps_b = [d['step'] for d in log_b]

    # Plot 1: Validation loss vs step for each length — Softmax vs SSMax (6 curves)
    plt.figure(figsize=(8, 5))
    for L in lengths:
        plt.plot(steps_a, [d[f'val@{L}'] for d in log_a], label=f'{label_a} val@{L}', linewidth=2)
        plt.plot(steps_b, [d[f'val@{L}'] for d in log_b], label=f'{label_b} val@{L}', linewidth=2)
    plt.xlabel('training step')
    plt.ylabel('validation loss')
    plt.title('Task 2 — Length generalization: Softmax vs SSMax')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    _ensure_output_dir(output_path)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Text8 Task2 curves saved to {output_path}")
    plt.close()

    # Plot 2: Final length generalization — two curves
    last_a = log_a[-1]
    last_b = log_b[-1]
    vals_a = [last_a[f'val@{L}'] for L in lengths]
    vals_b = [last_b[f'val@{L}'] for L in lengths]
    plt.figure(figsize=(6, 4))
    plt.plot(lengths, vals_a, marker='o', label=label_a, linewidth=2)
    plt.plot(lengths, vals_b, marker='s', label=label_b, linewidth=2)
    plt.xlabel('context length (eval)')
    plt.ylabel('validation loss')
    plt.title('Task 2 — Final length generalization')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    _ensure_output_dir(output_final)
    plt.savefig(output_final, dpi=150, bbox_inches='tight')
    print(f"Text8 Task2 final length gen saved to {output_final}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize attention weights and benchmarks')
    
    subparsers = parser.add_subparsers(dest='command', help='Visualization command')
    
    # Attention heatmap command (for language models)
    heatmap_parser = subparsers.add_parser('heatmap', help='Plot attention heatmap for language model')
    heatmap_parser.add_argument('--checkpoint', type=str, required=True,
                                help='Path to model checkpoint')
    heatmap_parser.add_argument('--data_path', type=str, default='input.txt',
                                help='Path to data file')
    heatmap_parser.add_argument('--layer', type=int, default=0,
                                help='Layer index to visualize')
    heatmap_parser.add_argument('--head', type=int, default=0,
                                help='Head index to visualize')
    heatmap_parser.add_argument('--output', type=str, default='plots/attention_heatmap.png',
                                help='Output path for heatmap')
    
    # CIFAR-10 attention visualization
    cifar_attn_parser = subparsers.add_parser('cifar10_attention', 
                                               help='Plot attention for CIFAR-10 images')
    cifar_attn_parser.add_argument('--checkpoint', type=str, required=True,
                                    help='Path to model checkpoint')
    cifar_attn_parser.add_argument('--data_dir', type=str, default='./data',
                                    help='CIFAR-10 data directory')
    cifar_attn_parser.add_argument('--layer', type=int, default=0,
                                    help='Layer index to visualize')
    cifar_attn_parser.add_argument('--head', type=int, default=0,
                                    help='Head index to visualize')
    cifar_attn_parser.add_argument('--num_samples', type=int, default=4,
                                    help='Number of samples to visualize')
    cifar_attn_parser.add_argument('--output', type=str, default='plots/cifar10_attention.png',
                                    help='Output path for visualization')
    
    # CIFAR-10 training curves
    cifar_curves_parser = subparsers.add_parser('cifar10_curves', 
                                                 help='Plot CIFAR-10 training curves')
    cifar_curves_parser.add_argument('--checkpoint', type=str, required=True,
                                     help='Path to model checkpoint')
    cifar_curves_parser.add_argument('--output', type=str, default='plots/cifar10_training.png',
                                     help='Output path for training curves')
    
    # Benchmark plot command
    benchmark_parser = subparsers.add_parser('benchmark', help='Plot benchmark results')
    benchmark_parser.add_argument('--results', type=str, default='experiment_results/benchmark_results.json',
                                  help='Path to benchmark results JSON')
    benchmark_parser.add_argument('--output', type=str, default='plots/benchmark_plot.png',
                                  help='Output path for benchmark plot')
    
    # Training curves command (for language models)
    curves_parser = subparsers.add_parser('curves', help='Plot language model training curves')
    curves_parser.add_argument('--checkpoint', type=str, required=True,
                               help='Path to model checkpoint')
    curves_parser.add_argument('--output', type=str, default='plots/training_curves.png',
                               help='Output path for training curves')
    
    # Text8 Task1 curves (from train_text8.py task1 --save_path)
    text8_parser = subparsers.add_parser('text8_curves', help='Plot text8 Task1 learning curves (train/val from checkpoint)')
    text8_parser.add_argument('--checkpoint', type=str, required=True,
                              help='Path to train_text8.py task1 checkpoint (.pt)')
    text8_parser.add_argument('--checkpoint_b', type=str, default=None,
                              help='Optional second checkpoint for Softmax vs SSMax comparison (validation loss only)')
    text8_parser.add_argument('--output', type=str, default='plots/text8_task1_curves.png',
                              help='Output path for plot')
    
    # Text8 Task2 curves (from train_text8.py task2 --save_path): curves over training + final length gen
    text8_task2_parser = subparsers.add_parser('text8_task2_curves', help='Plot text8 Task2 length generalization (val loss vs step + final length gen)')
    text8_task2_parser.add_argument('--checkpoint', type=str, required=True,
                                    help='Path to train_text8.py task2 checkpoint (.pt)')
    text8_task2_parser.add_argument('--checkpoint_b', type=str, default=None,
                                    help='Optional second checkpoint for Softmax vs SSMax comparison')
    text8_task2_parser.add_argument('--output', type=str, default='plots/text8_task2_curves.png',
                                    help='Output path for curves-over-training plot')
    text8_task2_parser.add_argument('--output_final', type=str, default=None,
                                    help='Output path for final length generalization plot (default: <output>_final_gen.png)')
    
    args = parser.parse_args()
    
    if args.command == 'heatmap':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model_args = checkpoint['args']
        
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        vocab_size = tokenizer.vocab_size
        
        with open(args.data_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        def encode(text):
            return tokenizer.encode(text, add_special_tokens=True)
        
        data = torch.tensor(encode(text), dtype=torch.long)
        n = int(0.9 * len(data))
        val_data = data[n:]
        
        block_size = model_args.block_size
        batch_size = 1
        ix = torch.randint(len(val_data) - block_size, (batch_size,))
        val_batch = torch.stack([val_data[i:i + block_size] for i in ix])
        val_batch = val_batch.to(device)
        
        model = SmallLanguageModel(
            vocab_size=vocab_size,
            n_embd=model_args.n_embd,
            n_head=model_args.n_head,
            n_layer=model_args.n_layer,
            block_size=model_args.block_size,
            dropout=model_args.dropout,
            head_type=model_args.head_type
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        
        plot_attention_heatmap(
            model, val_batch, tokenizer, device,
            layer=args.layer, head=args.head, output_path=args.output
        )
    
    elif args.command == 'cifar10_attention':

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model_args = checkpoint['args']
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616)
            )
        ])
        
        val_set = datasets.CIFAR10(
            root=args.data_dir,
            train=False,
            download=True,
            transform=transform
        )
        
        from torch.utils.data import DataLoader
        val_loader = DataLoader(val_set, batch_size=args.num_samples, shuffle=True)
        
        model = SmallVisionTransformer(
            img_size=model_args.img_size,
            patch_size=model_args.patch_size,
            in_channels=3,
            num_classes=10,
            n_embd=model_args.n_embd,
            n_head=model_args.n_head,
            n_layer=model_args.n_layer,
            dropout=model_args.dropout,
            head_type=model_args.head_type
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        
        plot_cifar10_attention(
            model, val_loader, device,
            layer=args.layer, head=args.head,
            num_samples=args.num_samples, output_path=args.output
        )
    
    elif args.command == 'cifar10_curves':
        plot_cifar10_training_curves(args.checkpoint, args.output)
    
    elif args.command == 'benchmark':
        plot_benchmark_results(args.results, args.output)
    
    elif args.command == 'curves':
        plot_training_curves(args.checkpoint, args.output)
    
    elif args.command == 'text8_curves':
        plot_text8_task1_curves(
            args.checkpoint,
            output_path=args.output,
            checkpoint_b=getattr(args, 'checkpoint_b', None),
        )
    
    elif args.command == 'text8_task2_curves':
        plot_text8_task2_curves(
            args.checkpoint,
            output_path=args.output,
            output_final=getattr(args, 'output_final', None),
            checkpoint_b=getattr(args, 'checkpoint_b', None),
        )
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()