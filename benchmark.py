import os
import torch
import numpy as np
import argparse
import json
from model import Head, LogisticHead, SoftmaxFreeHead, SSMaxHead


def benchmark_heads(context_lengths, n_embd=64, batch_size=2, n_runs=20, warmup_runs=20):
    """Benchmark different attention head implementations"""
    
    results = {
        'context_lengths': context_lengths,
        'vanilla': [],
        'logistic': [],
        'sima': [],
        'ssmax': []
    }
    
    for context_length in context_lengths:
        print(f"\nBenchmarking context length: {context_length}")
        block_size = context_length
        head_size = 64
        
        # Initialize heads
        vanilla_head = Head(head_size, block_size, n_embd).eval()
        logistic_head = LogisticHead(head_size, block_size, n_embd).eval()
        sima_head = SoftmaxFreeHead(head_size, block_size, n_embd).eval()
        ssmax_head = SSMaxHead(head_size, block_size, n_embd).eval()
        
        vanilla_times, logistic_times, sima_times, ssmax_times = [], [], [], []
        
        dummy_input = torch.zeros(size=(batch_size, context_length, n_embd))
        
        with torch.no_grad():
            
            print("  Benchmarking standard softmax head...")
            for i in range(warmup_runs):
                output = vanilla_head(dummy_input)
            
            for i in range(n_runs):
                with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU]
                ) as p:
                    output = vanilla_head(dummy_input)
                
                events = p.events()
                cpu_time_us = (
                    max(evt.time_range.end for evt in events)
                    - min(evt.time_range.start for evt in events)
                )
                vanilla_times.append(cpu_time_us)
            
            print("  Benchmarking logistic (sigmoid) head...")
            for i in range(warmup_runs):
                output = logistic_head(dummy_input)
            
            for i in range(n_runs):
                with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU]
                ) as p:
                    output = logistic_head(dummy_input)
                
                events = p.events()
                cpu_time_us = (
                    max(evt.time_range.end for evt in events)
                    - min(evt.time_range.start for evt in events)
                )
                logistic_times.append(cpu_time_us)
            
            print("  Benchmarking SimA (softmax-free) head...")
            for i in range(warmup_runs):
                output = sima_head(dummy_input)
            
            for i in range(n_runs):
                with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU]
                ) as p:
                    output = sima_head(dummy_input)
                
                events = p.events()
                cpu_time_us = (
                    max(evt.time_range.end for evt in events)
                    - min(evt.time_range.start for evt in events)
                )
                sima_times.append(cpu_time_us)
            
            print("  Benchmarking SSMax head...")
            for i in range(warmup_runs):
                output = ssmax_head(dummy_input)
            
            for i in range(n_runs):
                with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU]
                ) as p:
                    output = ssmax_head(dummy_input)
                
                events = p.events()
                cpu_time_us = (
                    max(evt.time_range.end for evt in events)
                    - min(evt.time_range.start for evt in events)
                )
                ssmax_times.append(cpu_time_us)
        
        vanilla_times = np.array(vanilla_times) / 1e6
        logistic_times = np.array(logistic_times) / 1e6
        sima_times = np.array(sima_times) / 1e6
        ssmax_times = np.array(ssmax_times) / 1e6
        
        results['vanilla'].append(float(vanilla_times.mean()))
        results['logistic'].append(float(logistic_times.mean()))
        results['sima'].append(float(sima_times.mean()))
        results['ssmax'].append(float(ssmax_times.mean()))
        
        print(f"  Softmax: {vanilla_times.mean():.4f}s")
        print(f"  Logistic: {logistic_times.mean():.4f}s")
        print(f"  SimA: {sima_times.mean():.4f}s")
        print(f"  SSMax: {ssmax_times.mean():.4f}s")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Benchmark attention head implementations')
    
    parser.add_argument('--context_lengths', type=int, nargs='+',
                        default=[64, 128, 256, 512, 1024, 2048, 4096, 8192],
                        help='Context lengths to benchmark')
    parser.add_argument('--n_embd', type=int, default=64,
                        help='Embedding dimension')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size for benchmarking')
    parser.add_argument('--n_runs', type=int, default=20,
                        help='Number of benchmark runs per configuration')
    parser.add_argument('--warmup_runs', type=int, default=20,
                        help='Number of warmup runs before benchmarking')
    parser.add_argument('--output', type=str, default='experiment_results/benchmark_results.json',
                        help='Output file for benchmark results')
    
    args = parser.parse_args()
    
    print("Starting benchmark...")
    print(f"Context lengths: {args.context_lengths}")
    print(f"Batch size: {args.batch_size}")
    print(f"Embedding dimension: {args.n_embd}")
    print(f"Runs per config: {args.n_runs}")
    
    results = benchmark_heads(
        context_lengths=args.context_lengths,
        n_embd=args.n_embd,
        batch_size=args.batch_size,
        n_runs=args.n_runs,
        warmup_runs=args.warmup_runs
    )
    
    # Save results
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nBenchmark results saved to {args.output}")


if __name__ == '__main__':
    main()
