import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
from utils.config import load_config
from datasets.dataset import get_dataloader
from models.generator import Generator
from evaluation.metrics import calculate_sisdr, calculate_lsd, calculate_pesq, calculate_stoi
from mlops.tracker import MLflowTracker

def main():
    parser = argparse.ArgumentParser(description="Evaluate HybridGAN-BWE and Log Results to MLflow")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config YAML file")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/lightweight/best_model.pth", help="Path to model checkpoint")
    parser.add_argument("--output_report", type=str, default="outputs/mlflow_evaluation_report.md", help="Output markdown report path")
    parser.add_argument("--num_samples", type=int, default=200, help="Number of test samples to evaluate")
    parser.add_argument("--experiment_name", type=str, default="HybridGAN-BWE", help="MLflow experiment name")
    parser.add_argument("--run_name", type=str, default="Test-Evaluation", help="MLflow run name")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    args = parser.parse_args()

    config = load_config(args.config)
    target_sr = config.audio.target_sr
    test_manifest = config.dataset.test_manifest

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[MLOps Evaluation] Loading checkpoint: {args.checkpoint}")
    generator = Generator(config)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    generator.load_state_dict(state["generator_state"] if "generator_state" in state else state)
    generator = generator.to(device).eval()

    param_count = sum(p.numel() for p in generator.parameters())
    test_loader = get_dataloader(test_manifest, config, is_validation=True)

    sisdr_scores, lsd_scores, pesq_scores, stoi_scores = [], [], [], []
    latencies, rtfs = [], []

    count = 0
    with torch.no_grad():
        for x_nb, y_real in tqdm(test_loader, desc="Evaluating"):
            if args.num_samples is not None and count >= args.num_samples:
                break

            x_nb = x_nb.to(device)
            y_real = y_real.to(device)

            # Measure Latency
            start_event = torch.cuda.Event(enable_timing=True) if device.type == 'cuda' else None
            end_event = torch.cuda.Event(enable_timing=True) if device.type == 'cuda' else None

            if device.type == 'cuda':
                start_event.record()
                y_fake = generator(x_nb)
                end_event.record()
                torch.cuda.synchronize()
                latency_ms = start_event.elapsed_time(end_event)
            else:
                import time
                t0 = time.time()
                y_fake = generator(x_nb)
                latency_ms = (time.time() - t0) * 1000.0

            audio_duration_sec = y_real.shape[-1] / target_sr
            rtf = (latency_ms / 1000.0) / audio_duration_sec

            latencies.append(latency_ms)
            rtfs.append(rtf)

            # Compute quality metrics
            sisdr = calculate_sisdr(y_fake, y_real)
            lsd = calculate_lsd(y_fake, y_real, target_sr)
            pesq_score = calculate_pesq(y_fake, y_real, target_sr)
            stoi_score = calculate_stoi(y_fake, y_real, target_sr)

            sisdr_scores.append(sisdr)
            lsd_scores.append(lsd)
            if not np.isnan(pesq_score):
                pesq_scores.append(pesq_score)
            if not np.isnan(stoi_score):
                stoi_scores.append(stoi_score)

            count += 1

    mean_sisdr = float(np.mean(sisdr_scores))
    mean_lsd = float(np.mean(lsd_scores))
    mean_pesq = float(np.mean(pesq_scores)) if pesq_scores else float('nan')
    mean_stoi = float(np.mean(stoi_scores)) if stoi_scores else float('nan')
    mean_latency = float(np.mean(latencies))
    mean_rtf = float(np.mean(rtfs))

    print("\n--- Evaluation Results ---")
    print(f"SI-SDR (dB): {mean_sisdr:.4f}")
    print(f"LSD (dB):    {mean_lsd:.4f}")
    print(f"PESQ (WB):   {mean_pesq:.4f}")
    print(f"STOI:        {mean_stoi:.4f}")
    print(f"Latency:     {mean_latency:.2f} ms")
    print(f"RTF:         {mean_rtf:.4f}")

    # Generate Markdown Report
    os.makedirs(os.path.dirname(args.output_report), exist_ok=True)
    report = f"""# HybridGAN-BWE MLflow Evaluation Report

## Model Summary
* **Checkpoint Path:** `{args.checkpoint}`
* **Parameter Count:** `{param_count:,}`
* **Evaluated Samples:** `{count}`

## Perceptual and Objective Metrics
| Metric | Value | Target Range |
| :--- | :---: | :--- |
| **SI-SDR** | `{mean_sisdr:.4f} dB` | Higher is better |
| **LSD** | `{mean_lsd:.4f} dB` | Lower is better |
| **PESQ (WB)** | `{mean_pesq:.4f}` | 1.0 to 4.5 (higher is better) |
| **STOI** | `{mean_stoi:.4f}` | 0.0 to 1.0 (higher is better) |

## Inference Latency & Speed
* **Average Inference Latency:** `{mean_latency:.2f} ms`
* **Real-Time Factor (RTF):** `{mean_rtf:.4f}`
* **Real-Time Status:** `{"Capable (RTF < 1.0)" if mean_rtf < 1.0 else "Non-Real-Time"}`
"""
    with open(args.output_report, "w", encoding="utf-8") as f:
        f.write(report)

    # Log to MLflow
    tracker = MLflowTracker(experiment_name=args.experiment_name)
    tracker.start_run(run_name=args.run_name, tags={"stage": "evaluation"})
    try:
        tracker.log_test_metrics({
            "test_sisdr_db": mean_sisdr,
            "test_lsd_db": mean_lsd,
            "test_pesq_wb": mean_pesq,
            "test_stoi": mean_stoi,
            "test_latency_ms": mean_latency,
            "test_rtf": mean_rtf
        })
        tracker.log_artifact(args.output_report, artifact_dir="reports")
        tracker.log_artifact(args.checkpoint, artifact_dir="checkpoints")
    finally:
        tracker.end_run()

if __name__ == "__main__":
    main()
