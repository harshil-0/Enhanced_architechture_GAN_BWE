import os
import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import gradio as gr
from utils.config import load_config
from utils.audio import pad_to_multiple
from utils.degradation import DegradationPipeline
from models.generator import Generator

# Cache loaded models to avoid loading weights on every run
cached_models = {}

def get_model(checkpoint_type: str):
    """Load model from cache or load weight files lazily."""
    if checkpoint_type in cached_models:
        return cached_models[checkpoint_type]
        
    config = load_config("configs/config.yaml")
    
    if checkpoint_type == "Baseline G.711 Model":
        checkpoint_path = "checkpoints/best_model.pth"
    else:
        checkpoint_path = "checkpoints/domain_randomization/best_model.pth"
        
    if not os.path.exists(checkpoint_path):
        # Fallback to whatever is available in checkpoints directory if domain_randomization path doesn't exist yet
        alt_path = "checkpoints/best_model.pth"
        if os.path.exists(alt_path):
            checkpoint_path = alt_path
        else:
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
        
    # Force loading on CPU to ensure zero VRAM impact on active training
    device = torch.device("cpu")
    
    print(f"Loading {checkpoint_type} from {checkpoint_path}...")
    model = Generator(config)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state["generator_state"] if "generator_state" in state else state)
    model = model.to(device).eval()
    
    cached_models[checkpoint_type] = (model, config)
    return model, config

def process_end_to_end(audio_path: str, audio_path_text: str, degradation_type: str, model_type: str):
    """Upload clean audio -> Degrade/Downsample -> Reconstruct."""
    path_to_use = audio_path if audio_path else audio_path_text
    if not path_to_use or not os.path.exists(path_to_use):
        return None, None, None, None
        
    try:
        model, config = get_model(model_type)
    except Exception as e:
        return None, None, None, f"Error loading checkpoint: {str(e)}"
        
    target_sr = config.audio.target_sr  # 16000
    
    # 1. Load original audio
    wav_orig, sr_orig = torchaudio.load(path_to_use)
    if wav_orig.shape[0] > 1:
        wav_orig = wav_orig.mean(dim=0, keepdim=True)
        
    # Resample original to 16 kHz to act as ground truth
    if sr_orig != target_sr:
        resampler_to_16k = torchaudio.transforms.Resample(orig_freq=sr_orig, new_freq=target_sr)
        wav_orig = resampler_to_16k(wav_orig)
        
    # Crop to max 8 seconds to prevent processing delays
    max_len = 8 * target_sr
    if wav_orig.shape[-1] > max_len:
        wav_orig = wav_orig[..., :max_len]
    else:
        wav_orig = pad_to_multiple(wav_orig, 256)
        
    # 2. Simulate Degradation
    config_audio = config.audio
    config_audio.degradation_type = degradation_type.lower().replace("-", "_").replace(" ", "_")
    if "resampling" in config_audio.degradation_type:
        config_audio.degradation_type = "none"
        
    degrader = DegradationPipeline(config_audio)
    wav_degraded = degrader(wav_orig)
    
    # Pad input for model compatibility
    wav_degraded_padded = pad_to_multiple(wav_degraded, 256)
    
    # 3. Model Inference
    input_tensor = wav_degraded_padded.unsqueeze(0)  # shape (1, 1, samples)
    with torch.no_grad():
        enhanced_tensor = model(input_tensor)
        enhanced_wav = enhanced_tensor.squeeze(0).cpu()[:, :wav_degraded.shape[-1]]
        
    # 4. Save files for playing
    os.makedirs("outputs/gradio_temp_sim", exist_ok=True)
    orig_save_path = "outputs/gradio_temp_sim/t1_original.wav"
    deg_save_path = "outputs/gradio_temp_sim/t1_degraded.wav"
    enh_save_path = "outputs/gradio_temp_sim/t1_enhanced.wav"
    
    torchaudio.save(orig_save_path, wav_orig, target_sr)
    torchaudio.save(deg_save_path, wav_degraded, target_sr)
    torchaudio.save(enh_save_path, enhanced_wav, target_sr)
    
    # 5. Plot Spectrogram Comparison
    y_orig = wav_orig.squeeze(0).numpy()
    y_deg = wav_degraded.squeeze(0).numpy()
    y_enh = enhanced_wav.squeeze(0).numpy()
    
    n_fft = 512
    hop_length = 128
    
    spec_orig = librosa.amplitude_to_db(np.abs(librosa.stft(y_orig, n_fft=n_fft, hop_length=hop_length)), ref=np.max)
    spec_deg = librosa.amplitude_to_db(np.abs(librosa.stft(y_deg, n_fft=n_fft, hop_length=hop_length)), ref=np.max)
    spec_enh = librosa.amplitude_to_db(np.abs(librosa.stft(y_enh, n_fft=n_fft, hop_length=hop_length)), ref=np.max)
    
    plt.style.use('dark_background')
    fig, axes = plt.subplots(3, 1, figsize=(11, 11), sharex=True)
    
    img1 = librosa.display.specshow(spec_orig, sr=target_sr, hop_length=hop_length, x_axis='time', y_axis='linear', ax=axes[0], cmap='magma')
    axes[0].set_title("Original High-Bandwidth Speech (0 - 8 kHz)", fontsize=11, fontweight='bold', pad=6)
    axes[0].set_ylabel("Freq (Hz)", fontsize=8)
    fig.colorbar(img1, ax=axes[0], format="%+2.0f dB")
    
    img2 = librosa.display.specshow(spec_deg, sr=target_sr, hop_length=hop_length, x_axis='time', y_axis='linear', ax=axes[1], cmap='magma')
    axes[1].set_title(f"Simulated Degraded/Downsampled Narrowband Input ({degradation_type})", fontsize=11, fontweight='bold', pad=6)
    axes[1].set_ylabel("Freq (Hz)", fontsize=8)
    fig.colorbar(img2, ax=axes[1], format="%+2.0f dB")
    
    img3 = librosa.display.specshow(spec_enh, sr=target_sr, hop_length=hop_length, x_axis='time', y_axis='linear', ax=axes[2], cmap='magma')
    axes[2].set_title(f"Enhanced Reconstructed Wideband Speech ({model_type})", fontsize=11, fontweight='bold', pad=6)
    axes[2].set_ylabel("Freq (Hz)", fontsize=8)
    axes[2].set_xlabel("Time (seconds)", fontsize=8)
    fig.colorbar(img3, ax=axes[2], format="%+2.0f dB")
    
    plt.tight_layout()
    plot_path = "outputs/gradio_temp_sim/t1_spec_comparison.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    
    return orig_save_path, deg_save_path, enh_save_path, plot_path

custom_css = """
body { background-color: #0b0f19; color: #f3f4f6; }
.gradio-container { background: #0b0f19 !important; border: none !important; }
.primary-button { background-color: #f43f5e !important; color: white !important; font-weight: bold !important; border-radius: 8px !important; }
.primary-button:hover { background-color: #e11d48 !important; }
"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="rose", neutral_hue="slate"), css=custom_css) as demo:
    gr.HTML(
        """
        <div style="text-align: center; padding: 25px 0 10px 0;">
            <h1 style="color: #f43f5e; font-size: 2.2rem; font-weight: 800; margin-bottom: 5px; font-family: 'Outfit', sans-serif;">
                📱 BWE App 1: End-to-End Simulation Pipeline
            </h1>
            <p style="color: #9ca3af; font-size: 1.1rem; margin-bottom: 20px; font-family: 'Inter', sans-serif;">
                Upload High-Bandwidth Speech ➔ Downsample & Degrade ➔ Reconstruct Wideband Signal
            </p>
        </div>
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown(
                """
                ### Inputs
                Upload a high-quality audio file (e.g. 16 kHz or 48 kHz). Select the simulation parameters and model checkpoint to evaluate.
                """
            )
            t1_input_file = gr.Audio(label="Upload Ground Truth Wideband Audio", type="filepath")
            t1_input_path = gr.Textbox(label="Or Enter Absolute File Path on Server (Alternative to upload)", placeholder="e.g. C:\\path\\to\\audio.wav")
            t1_degrad_type = gr.Dropdown(
                choices=["G.711 Mu-law", "G.711 A-law", "GSM Sim", "Pure Resampling", "Dynamic"],
                value="Dynamic",
                label="Select Telephone degradation type"
            )
            t1_model_type = gr.Dropdown(
                choices=["Domain-Randomized Model", "Baseline G.711 Model"],
                value="Domain-Randomized Model",
                label="Select BWE Model Checkpoint"
            )
            t1_submit = gr.Button("Downsample & Reconstruct", variant="primary", elem_classes="primary-button")
            
        with gr.Column(scale=2):
            gr.Markdown("### Results")
            with gr.Row():
                t1_play_orig = gr.Audio(label="1. Original Ground Truth Wideband", type="filepath", interactive=False)
                t1_play_deg = gr.Audio(label="2. Simulated Narrowband Degraded", type="filepath", interactive=False)
                t1_play_enh = gr.Audio(label="3. Reconstructed Wideband Output", type="filepath", interactive=False)
            
            t1_plot = gr.Image(label="Spectrogram Comparison (Original vs Degraded vs Enhanced)", type="filepath")
    
    t1_submit.click(
        fn=process_end_to_end,
        inputs=[t1_input_file, t1_input_path, t1_degrad_type, t1_model_type],
        outputs=[t1_play_orig, t1_play_deg, t1_play_enh, t1_plot]
    )
    
    gr.HTML(
        """
        <div style="text-align: center; padding: 20px 0; color: #6b7280; font-size: 0.9rem; font-family: 'Inter', sans-serif; border-top: 1px solid #1f2937; margin-top: 30px;">
            HybridGAN-BWE Neural Speech band extension platform • App 1: Simulation Pipeline
        </div>
        """
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7861, share=False)
