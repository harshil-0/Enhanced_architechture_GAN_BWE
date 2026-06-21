import os
import json
import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from utils.config import load_config
from utils.degradation import DegradationPipeline
from utils.audio import pad_to_multiple
from models.generator import Generator

def main():
    print("==================================================")
    print("HybridGAN-BWE: Graphical Analysis & Audio Sample Generator")
    print("==================================================")
    
    # 1. Load config
    config = load_config("configs/config.yaml")
    target_sr = config.audio.target_sr
    
    # 2. Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 3. Load test manifest and select a sample
    manifest_path = config.dataset.test_manifest
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Test manifest '{manifest_path}' not found. Please run training or manifest generation first.")
        
    with open(manifest_path, "r") as f:
        test_entries = json.load(f)
        
    if not test_entries:
        raise ValueError("Test manifest is empty.")
        
    # Select a speaker sample (e.g. the first entry)
    sample_entry = test_entries[0]
    filepath = sample_entry["filepath"]
    print(f"Selected evaluation sample: {filepath}")
    
    # 4. Load audio and resample to 16 kHz if necessary
    wav, sr = torchaudio.load(filepath)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
        
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        wav = resampler(wav)
        
    # Crop to exactly 3 seconds for clear visualization scale (3 * 16000 = 48000 samples)
    max_samples = 48000
    if wav.shape[-1] > max_samples:
        wav = wav[:, :max_samples]
    else:
        wav = pad_to_multiple(wav, 256)
        
    # 5. Load Generator checkpoint
    checkpoint_path = "checkpoints/best_model.pth"
    print(f"Loading Generator from: {checkpoint_path}")
    generator = Generator(config)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "generator_state" in state:
        generator.load_state_dict(state["generator_state"])
    else:
        generator.load_state_dict(state)
    generator = generator.to(device)
    generator.eval()
    
    # 6. Apply degradation pipeline (degraded narrowband resampled back to 16 kHz)
    degrader = DegradationPipeline(config.audio)
    degraded_wav = degrader(wav)
    
    # Pad to multiple of 256 for STFT compatibility
    degraded_wav_padded = pad_to_multiple(degraded_wav, 256)
    
    # 7. Run generator enhancement
    with torch.no_grad():
        input_tensor = degraded_wav_padded.unsqueeze(0).to(device) # shape (1, 1, samples)
        enhanced_tensor = generator(input_tensor)
        enhanced_wav = enhanced_tensor.squeeze(0).cpu() # shape (1, samples)
        
    # Align lengths for plotting
    min_len = min(wav.shape[-1], degraded_wav.shape[-1], enhanced_wav.shape[-1])
    wav = wav[:, :min_len]
    degraded_wav = degraded_wav[:, :min_len]
    enhanced_wav = enhanced_wav[:, :min_len]
    
    # 8. Save audio outputs
    out_dir = "outputs/samples"
    os.makedirs(out_dir, exist_ok=True)
    
    real_path = os.path.join(out_dir, "original_real.wav")
    degraded_path = os.path.join(out_dir, "degraded_nb.wav")
    enhanced_path = os.path.join(out_dir, "enhanced_gen.wav")
    
    torchaudio.save(real_path, wav, target_sr)
    torchaudio.save(degraded_path, degraded_wav, target_sr)
    torchaudio.save(enhanced_path, enhanced_wav, target_sr)
    
    print("\nSaved audio files:")
    print(f"  - Real Wideband GT: {real_path}")
    print(f"  - Degraded Narrowband: {degraded_path}")
    print(f"  - Enhanced Generated: {enhanced_path}")
    
    # 9. Plot comparative spectrograms using librosa
    print("\nPlotting comparative spectrograms...")
    n_fft = 512
    hop_length = 128
    
    y_real = wav.squeeze(0).numpy()
    y_deg = degraded_wav.squeeze(0).numpy()
    y_enh = enhanced_wav.squeeze(0).numpy()
    
    spec_real = librosa.amplitude_to_db(np.abs(librosa.stft(y_real, n_fft=n_fft, hop_length=hop_length)), ref=np.max)
    spec_deg = librosa.amplitude_to_db(np.abs(librosa.stft(y_deg, n_fft=n_fft, hop_length=hop_length)), ref=np.max)
    spec_enh = librosa.amplitude_to_db(np.abs(librosa.stft(y_enh, n_fft=n_fft, hop_length=hop_length)), ref=np.max)
    
    plt.style.use('dark_background')
    fig, axes = plt.subplots(3, 1, figsize=(12, 14), sharex=True)
    
    # Real Spectrogram
    img1 = librosa.display.specshow(spec_real, sr=target_sr, hop_length=hop_length, x_axis='time', y_axis='linear', ax=axes[0], cmap='magma')
    axes[0].set_title("Original Real Wideband Ground Truth (0 - 8 kHz)", fontsize=14, fontweight='bold', pad=10)
    axes[0].set_ylabel("Frequency (Hz)", fontsize=11)
    fig.colorbar(img1, ax=axes[0], format="%+2.0f dB")
    
    # Degraded Spectrogram
    img2 = librosa.display.specshow(spec_deg, sr=target_sr, hop_length=hop_length, x_axis='time', y_axis='linear', ax=axes[1], cmap='magma')
    axes[1].set_title("Narrowband Degraded Telephone Input (0 - 4 kHz cut-off)", fontsize=14, fontweight='bold', pad=10)
    axes[1].set_ylabel("Frequency (Hz)", fontsize=11)
    fig.colorbar(img2, ax=axes[1], format="%+2.0f dB")
    
    # Enhanced Spectrogram
    img3 = librosa.display.specshow(spec_enh, sr=target_sr, hop_length=hop_length, x_axis='time', y_axis='linear', ax=axes[2], cmap='magma')
    axes[2].set_title("Enhanced Generated Wideband Output (Reconstructed 4 - 8 kHz)", fontsize=14, fontweight='bold', pad=10)
    axes[2].set_ylabel("Frequency (Hz)", fontsize=11)
    axes[2].set_xlabel("Time (seconds)", fontsize=11)
    fig.colorbar(img3, ax=axes[2], format="%+2.0f dB")
    
    plt.suptitle("HybridGAN-BWE Spectral Analysis Comparison", fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    img_path = os.path.join(out_dir, "spectral_comparison.png")
    plt.savefig(img_path, dpi=300)
    plt.close()
    
    print(f"Spectral analysis plot saved to: {img_path}")
    print("\nGeneration and analysis successfully completed!")

if __name__ == "__main__":
    main()
