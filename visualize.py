import torch
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from transformers import AutoTokenizer
from model import SmallLanguageModel


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
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Benchmark plot saved to {output_path}")
    plt.close()


def plot_training_curves(checkpoint_path, output_path='training_curves.png'):
    """Plot training and validation loss curves"""
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
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
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Training curves saved to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize attention weights and benchmarks')
    
    subparsers = parser.add_subparsers(dest='command', help='Visualization command')
    
    heatmap_parser = subparsers.add_parser('heatmap', help='Plot attention heatmap')
    heatmap_parser.add_argument('--checkpoint', type=str, required=True,
                                help='Path to model checkpoint')
    heatmap_parser.add_argument('--data_path', type=str, default='input.txt',
                                help='Path to data file')
    heatmap_parser.add_argument('--layer', type=int, default=0,
                                help='Layer index to visualize')
    heatmap_parser.add_argument('--head', type=int, default=0,
                                help='Head index to visualize')
    heatmap_parser.add_argument('--output', type=str, default='attention_heatmap.png',
                                help='Output path for heatmap')
    
    benchmark_parser = subparsers.add_parser('benchmark', help='Plot benchmark results')
    benchmark_parser.add_argument('--results', type=str, default='benchmark_results.json',
                                  help='Path to benchmark results JSON')
    benchmark_parser.add_argument('--output', type=str, default='benchmark_plot.png',
                                  help='Output path for benchmark plot')
    
    curves_parser = subparsers.add_parser('curves', help='Plot training curves')
    curves_parser.add_argument('--checkpoint', type=str, required=True,
                               help='Path to model checkpoint')
    curves_parser.add_argument('--output', type=str, default='training_curves.png',
                               help='Output path for training curves')
    
    args = parser.parse_args()
    
    if args.command == 'heatmap':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        checkpoint = torch.load(args.checkpoint, map_location=device)
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
    
    elif args.command == 'benchmark':
        plot_benchmark_results(args.results, args.output)
    
    elif args.command == 'curves':
        plot_training_curves(args.checkpoint, args.output)
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
