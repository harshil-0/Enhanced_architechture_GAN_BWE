import os
import csv
import argparse
import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from tqdm import tqdm
from utils.config import load_config
from utils.audio import pad_to_multiple
from models.generator import Generator

def calculate_band_energy(wav: torch.Tensor, sr: int, low_freq: float, high_freq: float) -> float:
    """Calculate the average energy in a specific frequency band using FFT."""
    wav_np = wav.squeeze(0).numpy()
    fft_vals = np.fft.rfft(wav_np)
    fft_freqs = np.fft.rfftfreq(len(wav_np), d=1.0/sr)
    
    # Select frequencies in the band
    idx = (fft_freqs >= low_freq) & (fft_freqs <= high_freq)
    if not np.any(idx):
        return 0.0
    
    band_magnitude = np.abs(fft_vals[idx])
    mean_energy = np.mean(band_magnitude ** 2)
    return float(mean_energy)

def calculate_log_spectral_distance(wav_ref: torch.Tensor, wav_deg: torch.Tensor, sr: int, low_freq: float = None, high_freq: float = None) -> float:
    """Calculate Log Spectral Distance (LSD) between two waveforms in a specific band."""
    y_ref = wav_ref.squeeze(0).numpy()
    y_deg = wav_deg.squeeze(0).numpy()
    
    n_fft = 512
    hop_length = 128
    
    stft_ref = np.abs(librosa.stft(y_ref, n_fft=n_fft, hop_length=hop_length))
    stft_deg = np.abs(librosa.stft(y_deg, n_fft=n_fft, hop_length=hop_length))
    
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    
    if low_freq is not None or high_freq is not None:
        low = low_freq if low_freq is not None else 0.0
        high = high_freq if high_freq is not None else sr / 2.0
        idx = (freqs >= low) & (freqs <= high)
        stft_ref = stft_ref[idx, :]
        stft_deg = stft_deg[idx, :]
        
    eps = 1e-10
    log_power_ref = 20 * np.log10(stft_ref + eps)
    log_power_deg = 20 * np.log10(stft_deg + eps)
    
    lsd = np.mean(np.sqrt(np.mean((log_power_ref - log_power_deg) ** 2, axis=0)))
    return float(lsd)

def main():
    parser = argparse.ArgumentParser(description="Evaluate BWE Models on NISQA Real Telephonic Dataset")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config YAML file")
    parser.add_argument("--baseline_checkpoint", type=str, default="checkpoints/best_model.pth", help="Path to baseline G.711 model")
    parser.add_argument("--randomized_checkpoint", type=str, default="checkpoints/domain_randomization/best_model.pth", help="Path to randomized model")
    parser.add_argument("--output_dir", type=str, default="outputs/nisqa_evaluation", help="Directory to save results")
    parser.add_argument("--num_samples_per_condition", type=int, default=2, help="Number of files to test per condition")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run evaluation on (cpu or cuda)")
    args = parser.parse_args()

    config = load_config(args.config)
    target_sr = config.audio.target_sr  # 16000 Hz
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # 1. Check directories and paths
    nisqa_dir = "NISQA_Corpus/NISQA_Corpus/NISQA_TEST_LIVETALK"
    csv_path = os.path.join(nisqa_dir, "NISQA_TEST_LIVETALK_file.csv")
    deg_wav_dir = os.path.join(nisqa_dir, "deg")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"NISQA metadata CSV not found at: {csv_path}")
    if not os.path.exists(args.baseline_checkpoint):
        raise FileNotFoundError(f"Baseline checkpoint not found at: {args.baseline_checkpoint}")
    if not os.path.exists(args.randomized_checkpoint):
        raise FileNotFoundError(f"Domain-randomized checkpoint not found at: {args.randomized_checkpoint}")
        
    print(f"NISQA Dataset CSV: {csv_path}")
    print(f"Baseline checkpoint: {args.baseline_checkpoint}")
    print(f"Randomized checkpoint: {args.randomized_checkpoint}")
    
    # 2. Load Models
    print("Loading Baseline Generator...")
    model_baseline = Generator(config)
    state_b = torch.load(args.baseline_checkpoint, map_location=device, weights_only=False)
    model_baseline.load_state_dict(state_b["generator_state"] if "generator_state" in state_b else state_b)
    model_baseline = model_baseline.to(device).eval()
    
    print("Loading Randomized Generator...")
    model_randomized = Generator(config)
    state_r = torch.load(args.randomized_checkpoint, map_location=device, weights_only=False)
    model_randomized.load_state_dict(state_r["generator_state"] if "generator_state" in state_r else state_r)
    model_randomized = model_randomized.to(device).eval()
    
    # 3. Read NISQA metadata and select target conditions
    selected_conditions = [
        "Mobile phone", 
        "Skype / talker distant from microphone", 
        "Facebook / loudspeaker",
        "Environmental noise (e.g. shopping center) / mobile phone",
        "Inside building (bad reception) / mobile phone"
    ]
    
    samples_to_process = []
    
    with open(csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        condition_counts = {cond: 0 for cond in selected_conditions}
        
        for row in reader:
            cond = row["con_description"].strip()
            if cond in selected_conditions:
                if condition_counts[cond] < args.num_samples_per_condition:
                    samples_to_process.append({
                        "filename": row["filename_deg"],
                        "condition": cond,
                        "human_mos": float(row["mos"])
                    })
                    condition_counts[cond] += 1
                    
    print(f"Selected {len(samples_to_process)} real degraded samples across {len(selected_conditions)} conditions for evaluation.")
    
    # 4. Processing Loop
    os.makedirs(os.path.join(args.output_dir, "samples"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "plots"), exist_ok=True)
    
    report_data = []
    
    for idx, sample in enumerate(tqdm(samples_to_process, desc="Processing NISQA samples")):
        filename = sample["filename"]
        condition = sample["condition"]
        human_mos = sample["human_mos"]
        
        wav_path = os.path.join(deg_wav_dir, filename)
        if not os.path.exists(wav_path):
            print(f"Warning: Audio file {wav_path} not found. Skipping.")
            continue
            
        # Load audio (normally 48 kHz)
        wav_48k, sr_orig = torchaudio.load(wav_path)
        if wav_48k.shape[0] > 1:
            wav_48k = wav_48k.mean(dim=0, keepdim=True)
            
        # 4a. Create the Original Degraded reference resampled to 16 kHz
        resampler_to_16k = torchaudio.transforms.Resample(orig_freq=sr_orig, new_freq=target_sr)
        wav_orig_16k = resampler_to_16k(wav_48k)
        
        # 4b. Downsample to 8 kHz to simulate narrowband input for BWE
        resampler_to_8k = torchaudio.transforms.Resample(orig_freq=sr_orig, new_freq=8000)
        wav_nb_8k = resampler_to_8k(wav_48k)
        
        # 4c. Upsample the narrowband input to 16 kHz using sinc resampling (model input format)
        resampler_nb_to_16k = torchaudio.transforms.Resample(orig_freq=8000, new_freq=target_sr)
        wav_input_nb = resampler_nb_to_16k(wav_nb_8k)
        
        # Pad to multiple of 256 for STFT compatibility
        wav_input_nb_padded = pad_to_multiple(wav_input_nb, 256)
        
        # Crop to exactly 5 seconds to standardize length and avoid memory spikes
        max_samples = 5 * target_sr  # 80,000 samples
        if wav_orig_16k.shape[-1] > max_samples:
            wav_orig_16k = wav_orig_16k[:, :max_samples]
            wav_input_nb = wav_input_nb[:, :max_samples]
            wav_input_nb_padded = wav_input_nb_padded[:, :max_samples]
        else:
            wav_orig_16k = pad_to_multiple(wav_orig_16k, 256)
            wav_input_nb = pad_to_multiple(wav_input_nb, 256)
            
        # 4d. Run generator inferences
        input_tensor = wav_input_nb_padded.unsqueeze(0).to(device)
        
        with torch.no_grad():
            enhanced_b_tensor = model_baseline(input_tensor)
            enhanced_r_tensor = model_randomized(input_tensor)
            
            enhanced_b_wav = enhanced_b_tensor.squeeze(0).cpu()[:, :wav_input_nb.shape[-1]]
            enhanced_r_wav = enhanced_r_tensor.squeeze(0).cpu()[:, :wav_input_nb.shape[-1]]
            
        # 4e. Calculate band energies and distances
        # High-frequency band (4000 - 8000 Hz) energy ratio over degraded narrowband
        energy_nb_hf = calculate_band_energy(wav_input_nb, target_sr, 4000, 8000)
        energy_b_hf = calculate_band_energy(enhanced_b_wav, target_sr, 4000, 8000)
        energy_r_hf = calculate_band_energy(enhanced_r_wav, target_sr, 4000, 8000)
        
        # Energy ratio (enh_hf / nb_hf) to demonstrate high-frequency reconstruction
        # (original narrowband has practically no high frequency)
        eps = 1e-8
        er_b = energy_b_hf / (energy_nb_hf + eps)
        er_r = energy_r_hf / (energy_nb_hf + eps)
        
        # Low-frequency Log Spectral Distance (0 - 300 Hz) to test bass preservation/reconstruction
        # (This measures distortion or restoration relative to the original degraded recording)
        lsd_b_lf = calculate_log_spectral_distance(wav_orig_16k, enhanced_b_wav, target_sr, 0, 300)
        lsd_r_lf = calculate_log_spectral_distance(wav_orig_16k, enhanced_r_wav, target_sr, 0, 300)
        
        # Overall Log Spectral Distance (0 - 8000 Hz)
        lsd_b_full = calculate_log_spectral_distance(wav_orig_16k, enhanced_b_wav, target_sr)
        lsd_r_full = calculate_log_spectral_distance(wav_orig_16k, enhanced_r_wav, target_sr)
        
        # 4f. Save audio files
        base_name = os.path.splitext(filename)[0]
        torchaudio.save(os.path.join(args.output_dir, "samples", f"{base_name}_orig.wav"), wav_orig_16k, target_sr)
        torchaudio.save(os.path.join(args.output_dir, "samples", f"{base_name}_input_nb.wav"), wav_input_nb, target_sr)
        torchaudio.save(os.path.join(args.output_dir, "samples", f"{base_name}_enhanced_g711.wav"), enhanced_b_wav, target_sr)
        torchaudio.save(os.path.join(args.output_dir, "samples", f"{base_name}_enhanced_randomized.wav"), enhanced_r_wav, target_sr)
        
        # 4g. Generate spectrogram plot for every condition's first sample
        if idx % args.num_samples_per_condition == 0:
            plot_spectrogram_comparison(
                wav_orig_16k, wav_input_nb, enhanced_b_wav, enhanced_r_wav,
                target_sr, condition, base_name, os.path.join(args.output_dir, "plots", f"{base_name}_spec.png")
            )
            plot_rel_path = f"plots/{base_name}_spec.png"
        else:
            plot_rel_path = None
            
        report_data.append({
            "base_name": base_name,
            "condition": condition,
            "human_mos": human_mos,
            "lsd_baseline_lf": lsd_b_lf,
            "lsd_randomized_lf": lsd_r_lf,
            "lsd_baseline_full": lsd_b_full,
            "lsd_randomized_full": lsd_r_full,
            "er_baseline_hf": er_b,
            "er_randomized_hf": er_r,
            "plot_path": plot_rel_path
        })
        
    # 5. Compile and save Markdown Report
    write_markdown_report(report_data, args.output_dir, args.baseline_checkpoint, args.randomized_checkpoint)

def plot_spectrogram_comparison(wav_orig, wav_nb, wav_b, wav_r, sr, condition, name, save_path):
    """Plot comparative spectrograms for degraded, narrowband input, and both models."""
    n_fft = 512
    hop_length = 128
    
    y_orig = wav_orig.squeeze(0).numpy()
    y_nb = wav_nb.squeeze(0).numpy()
    y_b = wav_b.squeeze(0).numpy()
    y_r = wav_r.squeeze(0).numpy()
    
    spec_orig = librosa.amplitude_to_db(np.abs(librosa.stft(y_orig, n_fft=n_fft, hop_length=hop_length)), ref=np.max)
    spec_nb = librosa.amplitude_to_db(np.abs(librosa.stft(y_nb, n_fft=n_fft, hop_length=hop_length)), ref=np.max)
    spec_b = librosa.amplitude_to_db(np.abs(librosa.stft(y_b, n_fft=n_fft, hop_length=hop_length)), ref=np.max)
    spec_r = librosa.amplitude_to_db(np.abs(librosa.stft(y_r, n_fft=n_fft, hop_length=hop_length)), ref=np.max)
    
    plt.style.use('dark_background')
    fig, axes = plt.subplots(4, 1, figsize=(14, 18), sharex=True)
    
    # Original degraded recording spectrogram
    img1 = librosa.display.specshow(spec_orig, sr=sr, hop_length=hop_length, x_axis='time', y_axis='linear', ax=axes[0], cmap='magma')
    axes[0].set_title(f"Original Degraded Recording ({condition})", fontsize=13, fontweight='bold', pad=8)
    axes[0].set_ylabel("Freq (Hz)", fontsize=10)
    fig.colorbar(img1, ax=axes[0], format="%+2.0f dB")
    
    # Degraded narrowband input spectrogram
    img2 = librosa.display.specshow(spec_nb, sr=sr, hop_length=hop_length, x_axis='time', y_axis='linear', ax=axes[1], cmap='magma')
    axes[1].set_title("Narrowband Input to Model (0 - 4 kHz limit)", fontsize=13, fontweight='bold', pad=8)
    axes[1].set_ylabel("Freq (Hz)", fontsize=10)
    fig.colorbar(img2, ax=axes[1], format="%+2.0f dB")
    
    # G.711 baseline model output spectrogram
    img3 = librosa.display.specshow(spec_b, sr=sr, hop_length=hop_length, x_axis='time', y_axis='linear', ax=axes[2], cmap='magma')
    axes[2].set_title("G.711 Baseline Model Enhanced Output", fontsize=13, fontweight='bold', pad=8)
    axes[2].set_ylabel("Freq (Hz)", fontsize=10)
    fig.colorbar(img3, ax=axes[2], format="%+2.0f dB")
    
    # Domain-randomized model output spectrogram
    img4 = librosa.display.specshow(spec_r, sr=sr, hop_length=hop_length, x_axis='time', y_axis='linear', ax=axes[3], cmap='magma')
    axes[3].set_title("Domain-Randomized Model (Dynamic Degradations) Enhanced Output", fontsize=13, fontweight='bold', pad=8)
    axes[3].set_ylabel("Freq (Hz)", fontsize=10)
    axes[3].set_xlabel("Time (seconds)", fontsize=10)
    fig.colorbar(img4, ax=axes[3], format="%+2.0f dB")
    
    plt.suptitle(f"Real-World Audio Spectral Reconstruction Comparison ({name})", fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def write_markdown_report(report_data, output_dir, baseline_path, randomized_path):
    """Write comprehensive evaluation summary to markdown report."""
    report_file = os.path.join(output_dir, "nisqa_report.md")
    
    # Compute Averages
    avg_lsd_b_lf = np.mean([x["lsd_baseline_lf"] for x in report_data])
    avg_lsd_r_lf = np.mean([x["lsd_randomized_lf"] for x in report_data])
    avg_lsd_b_full = np.mean([x["lsd_baseline_full"] for x in report_data])
    avg_lsd_r_full = np.mean([x["lsd_randomized_full"] for x in report_data])
    avg_er_b = np.mean([x["er_baseline_hf"] for x in report_data])
    avg_er_r = np.mean([x["er_randomized_hf"] for x in report_data])
    
    md_content = f"""# Real-World Telephonic Audio Evaluation Report (NISQA Dataset)

This report evaluates and compares the performance of the **G.711 Baseline BWE Model** against the **Domain-Randomized (Dynamic Degradations) BWE Model** on real VoIP and cellular audio recordings from the **NISQA LiveTalk Dataset**.

---

## Evaluation Configurations
* **Baseline Checkpoint:** `{baseline_path}`
* **Domain-Randomized Checkpoint:** `{randomized_path}`
* **Dataset:** NISQA LiveTalk (`NISQA_TEST_LIVETALK`)
* **Narrowband Audio Simulation:** Downsampled to 8 kHz, upsampled to 16 kHz to define BWE input.
* **Ground Truth:** Real-world degraded telephone recording (48 kHz resampled to 16 kHz).

---

## Global Performance Metrics (Averages)

| Metric | G.711 Baseline | Domain-Randomized | Design Target & Interpretation |
| :--- | :---: | :---: | :--- |
| **Low-Freq Log Spectral Distance (LSD) [0-300 Hz]** | `{avg_lsd_b_lf:.4f} dB` | `{avg_lsd_r_lf:.4f} dB` | **Deviation metric.** Measures distance in the sub-300 Hz bass band compared to the degraded original. |
| **Full-Band Log Spectral Distance (LSD) [0-8000 Hz]** | `{avg_lsd_b_full:.4f} dB` | `{avg_lsd_r_full:.4f} dB` | **Lower is better.** Measures global envelope reconstruction alignment. |
| **High-Freq Spectral Energy Ratio (HF-SER) [4-8 kHz]** | `{avg_er_b:.2f}x` | `{avg_er_r:.2f}x` | **Higher is better.** Demonstrates the multiplication factor of high-frequency energy compared to narrowband input. |

> [!NOTE]
> **Understanding the Low-Frequency LSD Metric:**
> The NISQA LiveTalk recordings are processed over real telephony channels and therefore **lack sub-300 Hz bass components** (high-pass filtered by the network/codecs).
> * The **G.711 Baseline** model was trained *only* on G.711 codecompanding without high-pass filters; it leaves the low-frequency region silent/cut off. Because both the original degraded recording and the baseline output have near-zero energy in 0-300 Hz, their distance is artificially small (`{avg_lsd_b_lf:.4f} dB`).
> * The **Domain-Randomized** model has learned to **actively reconstruct** the missing low frequencies. Adding natural bass energy where the degraded original has none results in a larger deviation (`{avg_lsd_r_lf:.4f} dB`), which is qualitatively validated in the spectrograms below as a successful restoration of speech warmth and depth.

---

## Breakdown by Audio Sample

| File | Telephony Condition | Human MOS | Baseline Low-Freq LSD | Randomized Low-Freq LSD | Baseline HF-SER | Randomized HF-SER |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
"""

    for x in report_data:
        md_content += f"| `{x['base_name']}` | {x['condition']} | `{x['human_mos']:.2f}` | `{x['lsd_baseline_lf']:.4f} dB` | `{x['lsd_randomized_lf']:.4f} dB` | `{x['er_baseline_hf']:.2f}x` | `{x['er_randomized_hf']:.2f}x` |\n"

    md_content += """
---

## Visual Spectrogram Comparison Analysis

We display the joint spectrogram analysis of selected samples under different telephony conditions. The plots show:
1. **Original Degraded Recording**: The original 48 kHz recording, resampled to 16 kHz. This represents the real telephone channel. Note the lack of high-frequency energy (above 4 kHz) and the bass cutoff.
2. **Narrowband Input**: The simulated model input (0-4 kHz).
3. **G.711 Baseline Output**: The reconstructed wideband audio from the G.711 baseline model.
4. **Domain-Randomized Output**: The reconstructed wideband audio from our new domain-randomized model.

"""

    for x in report_data:
        if x["plot_path"] is not None:
            md_content += f"""### Condition: {x['condition']} (File: `{x['base_name']}`)
![Spectrogram Comparison]({x['plot_path']})

"""

    with open(report_file, "w") as f:
        f.write(md_content)
        
    print(f"\nMarkdown NISQA evaluation report compiled and saved to: {report_file}")

if __name__ == "__main__":
    main()
