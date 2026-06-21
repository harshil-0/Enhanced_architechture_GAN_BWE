import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
from utils.config import load_config
from datasets.dataset import get_dataloader
from models.generator import Generator
from evaluation.metrics import (
    calculate_sisdr, calculate_lsd, calculate_pesq,
    calculate_stoi, count_parameters, benchmark_inference
)

def main():
    parser = argparse.ArgumentParser(description="Evaluate HybridGAN-BWE Speech Bandwidth Extension Model")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config YAML file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to generator model checkpoint (.pth)")
    parser.add_argument("--output_report", type=str, default="outputs/evaluation_report.md", help="Path to save markdown report")
    parser.add_argument("--num_samples", type=int, default=None, help="Limit evaluation to first N samples from test loader")
    parser.add_argument("--max_len", type=int, default=32768, help="Max length in samples to evaluate to avoid CUDA OOM (default: 32768)")
    args = parser.parse_args()

    # 1. Load config
    config = load_config(args.config)
    target_sr = config.audio.target_sr
    
    # 2. Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for evaluation: {device}")

    # 3. Load dataset in evaluation mode (full length audio files)
    test_manifest = config.dataset.test_manifest
    if not os.path.exists(test_manifest):
        raise FileNotFoundError(f"Test manifest not found: {test_manifest}. Please run train.py first to generate datasets.")
        
    print(f"Loading test dataset from manifest: {test_manifest}")
    test_loader = get_dataloader(test_manifest, config, is_validation=False, is_evaluation=True)
    print(f"Test dataset loaded. Number of files: {len(test_loader)}")

    # 4. Load Generator
    print("Loading Generator model...")
    generator = Generator(config)
    
    # Load state dict
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if "generator_state" in state:
        generator.load_state_dict(state["generator_state"])
    else:
        generator.load_state_dict(state)
        
    generator = generator.to(device)
    generator.eval()
    print("Checkpoint loaded successfully.")

    # Model complexity
    param_count = count_parameters(generator)
    print(f"Generator parameter count: {param_count:,}")

    # 5. Evaluation Loop
    sisdr_scores = []
    lsd_scores = []
    pesq_scores = []
    stoi_scores = []
    latencies = []
    rtfs = []
    
    print("Evaluating test dataset...")
    count = 0
    with torch.no_grad():
        for x_nb, y_real in tqdm(test_loader):
            if args.num_samples is not None and count >= args.num_samples:
                break
                
            # Crop to max_len if specified to prevent CUDA OOM
            if args.max_len is not None and x_nb.shape[-1] > args.max_len:
                start = (x_nb.shape[-1] - args.max_len) // 2
                x_nb = x_nb[..., start:start + args.max_len]
                y_real = y_real[..., start:start + args.max_len]
                
            # Batch size is 1 for evaluation mode
            x_nb = x_nb.to(device)
            y_real = y_real.to(device)
            
            # Benchmark inference latency and RTF on this sample
            latency, rtf, _ = benchmark_inference(generator, x_nb, target_sr, num_runs=5)
            latencies.append(latency)
            rtfs.append(rtf)
            
            # Run forward pass
            y_fake = generator(x_nb)
            
            # Compute quality metrics
            sisdr = calculate_sisdr(y_fake, y_real)
            lsd = calculate_lsd(y_fake, y_real)
            pesq_score = calculate_pesq(y_fake, y_real, target_sr)
            stoi_score = calculate_stoi(y_fake, y_real, target_sr)
            
            sisdr_scores.append(sisdr)
            lsd_scores.append(lsd)
            if not np.isnan(pesq_score):
                pesq_scores.append(pesq_score)
            if not np.isnan(stoi_score):
                stoi_scores.append(stoi_score)
                
            count += 1
                
    # 6. Compute Averages
    mean_sisdr = np.mean(sisdr_scores)
    mean_lsd = np.mean(lsd_scores)
    mean_pesq = np.mean(pesq_scores) if pesq_scores else float('nan')
    mean_stoi = np.mean(stoi_scores) if stoi_scores else float('nan')
    mean_latency = np.mean(latencies)
    mean_rtf = np.mean(rtfs)

    print("\n--- Evaluation Results ---")
    print(f"SI-SDR (dB): {mean_sisdr:.4f}")
    print(f"LSD (dB):    {mean_lsd:.4f}")
    print(f"PESQ (WB):   {mean_pesq:.4f}" if not np.isnan(mean_pesq) else "PESQ (WB):   Not Available")
    print(f"STOI:        {mean_stoi:.4f}" if not np.isnan(mean_stoi) else "STOI:        Not Available")
    print(f"Latency:     {mean_latency:.2f} ms")
    print(f"RTF:         {mean_rtf:.4f} (Real-time factor)")
    
    # 7. Write Markdown Report
    os.makedirs(os.path.dirname(args.output_report), exist_ok=True)
    
    report = f"""# HybridGAN-BWE Evaluation Report

## Model Summary
* **Checkpoint Path:** `{args.checkpoint}`
* **Parameter Count:** `{param_count:,}`
* **Reconstruction Branch:** `{"Waveform-only (Baseline)" if not config.generator.use_spectral_branch else "Hybrid (Waveform + Spectral)"}`
* **Target Sample Rate:** `{target_sr} Hz`
* **Source/Codec Degradation:** `{config.audio.degradation_type}`
* **Evaluated Samples:** `{count}`

## Perceptual and Objective Metrics
| Metric | Value | Range / Target |
| :--- | :--- | :--- |
| **SI-SDR** | `{mean_sisdr:.4f} dB` | Higher is better (speech restoration) |
| **LSD** | `{mean_lsd:.4f} dB` | Lower is better (spectral envelope distance) |
| **PESQ (WB)** | `{"{:.4f}".format(mean_pesq) if not np.isnan(mean_pesq) else "N/A"}` | 1.0 to 4.5 (perceptual quality) |
| **STOI** | `{"{:.4f}".format(mean_stoi) if not np.isnan(mean_stoi) else "N/A"}` | 0.0 to 1.0 (intelligibility) |

## Inference Speed and Latency Benchmarks
* **Average Inference Latency:** `{mean_latency:.2f} ms` (measured on `{device.type.upper()}`)
* **Real-Time Factor (RTF):** `{mean_rtf:.4f}`
* **Real-time inference status:** `{"Capable (RTF < 1.0)" if mean_rtf < 1.0 else "Not Capable (RTF >= 1.0)"}`

*Report generated on: 2026-06-18*
"""
    with open(args.output_report, "w") as f:
        f.write(report)
    print(f"\nMarkdown evaluation report saved to: {args.output_report}")

if __name__ == "__main__":
    main()
