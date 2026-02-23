# Evaluating attention mechanisms beyond softmax in long contexts

This repository contains all the implementations for the poster that was presented during the CentraleSupélec deep learning course poster session (see the full poster below !).

## Implemented Attention Heads

- **Standard Softmax** (`Head`): Vanilla self attention head
- **SSMax** (`SSMaxHead`): Scalable softmax for better long context accuracy (see [1])
- **Logistic** (`LogisticHead`): Element-wise sigmoid activation instead of softmax
- **SimA** (`SoftmaxFreeHead`): Softmax-free attention with normalized queries and keys (see [2])


## Usage

### 1. Training a Model

`train.py` has two commands: **lm** (custom text with GPT-2 tokenizer) and **text8** (char-level on text8, task1 or task2).

**LM (e.g. Tiny Shakespeare):** train with the default softmax-free (SimA) attention:

```bash
python train.py lm --data_path input.txt --save_path model.pt
```

Train with different attention mechanisms:

```bash
# Standard softmax attention
python train.py lm --data_path input.txt --head_type standard --save_path model_softmax.pt

# SSMax attention
python train.py lm --data_path input.txt --head_type ssmax --save_path model_ssmax.pt

# Logistic (sigmoid) attention
python train.py lm --data_path input.txt --head_type logistic --save_path model_logistic.pt
```


### 2. Benchmarking Performance

Run benchmarks to compare inference speed of different attention heads:

```bash
python benchmark.py --output benchmark_results.json
```

Customize benchmark settings:

```bash
python benchmark.py \
    --context_lengths 64 128 256 512 1024 2048 4096 8192 \
    --batch_size 2 \
    --n_runs 50 \
    --output benchmark_results.json
```

### 3. Visualization

#### Plot Attention Heatmaps

Visualize attention weights from a trained model:

```bash
python visualize.py heatmap \
    --checkpoint model.pt \
    --data_path input.txt \
    --layer 0 \
    --head 0 \
    --output attention_heatmap.png
```


#### Plot CIFAR-10 Attention Maps

Visualize attention patterns on CIFAR-10 images:

```bash
python visualize.py cifar10_attention \
    --checkpoint cifar10_model.pt \
    --layer 0 \
    --head 0 \
    --num_samples 4 \
    --output cifar10_attention.png
```

#### Plot Benchmark Results

Generate a plot from benchmark results:

```bash
python visualize.py benchmark \
    --results benchmark_results.json \
    --output benchmark_plot.png
```

#### Plot Training Curves

Visualize training and validation loss:

```bash
python visualize.py curves \
    --checkpoint model.pt \
    --output training_curves.png
```

### 4. Text8 language model (Tasks 1 & 2: Softmax vs SSMax)

Train char-level language models on **text8** for learning curves (task1) or length generalization (task2). The script downloads text8 automatically if missing.

```bash
# Task 1: learning curves (fixed context 256, position embedding)
python train.py text8 task1 --head_type ssmax --max_iters 10000 --save_path ckpt_task1_ssmax.pt

# Task 2: length generalization (train at 256, eval at 256/512/1024, sinusoidal positions)
python train.py text8 task2 --head_type ssmax --max_iters 3000 --save_path ckpt_task2_ssmax.pt
```

Use `--head_type standard` for Softmax or `--head_type ssmax` for SSMax. Optional: `--data_dir` to set the directory for text8, `--save_path` to save the checkpoint and training log. Checkpoints produced by `train.py text8 task1` / `task2` are compatible with the visualization commands below.

To **replot Task 1 learning curves** from saved checkpoints (like in the notebook):

```bash
# Single run: train + val loss
python visualize.py text8_curves --checkpoint ckpt_task1_ssmax.pt --output task1_curves.png

# Comparison: Softmax vs SSMax (validation loss only)
python visualize.py text8_curves --checkpoint ckpt_task1_softmax.pt --checkpoint_b ckpt_task1_ssmax.pt --output task1_comparison.png
```

To **replot Task 2 length generalization** (validation loss vs step for 256/512/1024 + final length gen, like the last two plots of the Task 2 notebook):

```bash
# Single run: two figures (curves over training + final length gen)
python visualize.py text8_task2_curves --checkpoint ckpt_task2_ssmax.pt --output task2_curves.png

# Softmax vs SSMax: same two figures with both runs
python visualize.py text8_task2_curves --checkpoint ckpt_task2_softmax.pt --checkpoint_b ckpt_task2_ssmax.pt --output task2_curves.png --output_final task2_final_gen.png
```

### 5. Task 3: Needle-in-a-Haystack retrieval

Train Softmax (a) and SSMax (b) on synthetic retrieval data, then evaluate on a grid (context length × needle depth). **Output:** side-by-side heatmap (Softmax | SSMax) only.

```bash
# Run both experiments and plot heatmap (saves task3_results_a.json, task3_results_b.json)
python run_task3_retrieval.py --output task3_heatmap.png

# Plot heatmap from existing results only (no training)
python run_task3_retrieval.py --skip_train --output task3_heatmap.png
```

Options: `--results_dir` for JSON output directory, `--max_iters`, `--eval_lengths`, `--eval_depths`, etc.

## Training Arguments language model

| Argument | Description | Default |
|----------|-------------|---------|
| `--data_path` | Path to training data | `input.txt` |
| `--n_embd` | Embedding dimension | 64 |
| `--n_head` | Number of attention heads | 2 |
| `--n_layer` | Number of transformer layers | 2 |
| `--block_size` | Maximum context length | 8192 |
| `--dropout` | Dropout rate | 0.0 |
| `--head_type` | Attention mechanism (`softmax_free`, `standard`, `ssmax`, `logistic`) | `softmax_free` |
| `--batch_size` | Batch size | 16 |
| `--max_iters` | Training iterations | 1000 |
| `--learning_rate` | Learning rate | 3e-4 |
| `--eval_interval` | Evaluation frequency | 100 |
| `--gen_tokens` | Tokens to generate after training | 1000 |
| `--seed` | Random seed | 1337 |
| `--save_path` | Path to save model | None |


### CIFAR-10 Vision Transformer (train_cifar10.py)

| Argument | Description | Default |
|----------|-------------|---------|
| `--data_dir` | Directory for CIFAR-10 data | `./data` |
| `--img_size` | Image size | 32 |
| `--patch_size` | Patch size for vision transformer | 4 |
| `--n_embd` | Embedding dimension | 64 |
| `--n_head` | Number of attention heads | 8 |
| `--n_layer` | Number of transformer layers | 3 |
| `--dropout` | Dropout rate | 0.2 |
| `--head_type` | Attention mechanism (`standard`, `logistic`, `softmax_free`) | `standard` |
| `--batch_size` | Batch size | 32 |
| `--max_iters` | Training iterations | 5000 |
| `--learning_rate` | Learning rate | 3e-4 |
| `--eval_interval` | Evaluation frequency | 100 |
| `--eval_iters` | Iterations for evaluation | 200 |
| `--seed` | Random seed | 1337 |
| `--save_path` | Path to save model | None |

## Data

The training script expects a text file (e.g., `input.txt`) containing the training corpus. 

You can download the tiny shakespeare dataset here :
```bash
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

## Poster

[![Poster preview](assets/poster_preview.png)](assets/poster.pdf)

**[Download the poster (PDF)](assets/poster.pdf)**


## References

1. Nakanishi, K. M. (2025). **Scalable-softmax is superior for attention**. arXiv preprint arXiv:2501.19399.

2. Koohpayegani, S. A., & Pirsiavash, H. (2024). **Sima: Simple softmax-free attention for vision transformers**. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (pp. 2607-2617).