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
    elif checkpoint_type == "Ours - Lightweight Model":
        checkpoint_path = "checkpoints/lightweight/best_model.pth"
    else:
        checkpoint_path = "checkpoints/domain_randomization/best_model.pth"
        
    if not os.path.exists(checkpoint_path):
        # Fallback to whatever is available in checkpoints directory if domain_randomization path doesn't exist yet
        alt_path = "checkpoints/best_model.pth"
        if os.path.exists(alt_path):
            checkpoint_path = alt_path
        else:
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
        
    # Use GPU if available, fallback to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading {checkpoint_type} from {checkpoint_path}...")
    model = Generator(config)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state["generator_state"] if "generator_state" in state else state)
    model = model.to(device).eval()
    
    cached_models[checkpoint_type] = (model, config)
    return model, config

def process_upsampling(audio_path: str, audio_path_text: str, model_type: str):
    """Upload existing degraded/narrowband audio -> Upsample."""
    path_to_use = audio_path if audio_path else audio_path_text
    if not path_to_use or not os.path.exists(path_to_use):
        return None, None, None
        
    try:
        model, config = get_model(model_type)
    except Exception as e:
        return None, None, f"Error loading checkpoint: {str(e)}"
        
    target_sr = config.audio.target_sr  # 16000
    
    # 1. Load audio (e.g. 8 kHz telephone call)
    wav, sr_orig = torchaudio.load(path_to_use)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
        
    # Resample to target sample rate (16 kHz)
    resampler = torchaudio.transforms.Resample(orig_freq=sr_orig, new_freq=target_sr)
    wav_16k = resampler(wav)
    
    # Crop to max 8 seconds to prevent delays
    max_len = 8 * target_sr
    if wav_16k.shape[-1] > max_len:
        wav_16k = wav_16k[..., :max_len]
    else:
        wav_16k = pad_to_multiple(wav_16k, 256)
        
    # 2. Model Inference
    input_tensor = wav_16k.unsqueeze(0).to(device)  # shape (1, 1, samples)
    with torch.no_grad():
        enhanced_tensor = model(input_tensor)
        enhanced_wav = enhanced_tensor.squeeze(0).cpu()[:, :wav_16k.shape[-1]]
        
    # 3. Save files
    os.makedirs("outputs/gradio_temp_upsample", exist_ok=True)
    input_save_path = "outputs/gradio_temp_upsample/t2_input.wav"
    enh_save_path = "outputs/gradio_temp_upsample/t2_enhanced.wav"
    
    torchaudio.save(input_save_path, wav_16k, target_sr)
    torchaudio.save(enh_save_path, enhanced_wav, target_sr)
    
    # 4. Plot Spectrogram Comparison
    y_in = wav_16k.squeeze(0).numpy()
    y_enh = enhanced_wav.squeeze(0).numpy()
    
    n_fft = 512
    hop_length = 128
    
    spec_in = librosa.amplitude_to_db(np.abs(librosa.stft(y_in, n_fft=n_fft, hop_length=hop_length)), ref=np.max)
    spec_enh = librosa.amplitude_to_db(np.abs(librosa.stft(y_enh, n_fft=n_fft, hop_length=hop_length)), ref=np.max)
    
    plt.style.use('dark_background')
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True)
    
    img1 = librosa.display.specshow(spec_in, sr=target_sr, hop_length=hop_length, x_axis='time', y_axis='linear', ax=axes[0], cmap='magma')
    axes[0].set_title("Uploaded Narrowband Input Audio (0 - 4 kHz limit)", fontsize=11, fontweight='bold', pad=6)
    axes[0].set_ylabel("Freq (Hz)", fontsize=8)
    fig.colorbar(img1, ax=axes[0], format="%+2.0f dB")
    
    img2 = librosa.display.specshow(spec_enh, sr=target_sr, hop_length=hop_length, x_axis='time', y_axis='linear', ax=axes[1], cmap='magma')
    axes[1].set_title(f"Enhanced Generated Wideband Audio ({model_type})", fontsize=11, fontweight='bold', pad=6)
    axes[1].set_ylabel("Freq (Hz)", fontsize=8)
    axes[1].set_xlabel("Time (seconds)", fontsize=8)
    fig.colorbar(img2, ax=axes[1], format="%+2.0f dB")
    
    plt.tight_layout()
    plot_path = "outputs/gradio_temp_upsample/t2_spec_comparison.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    
    return input_save_path, enh_save_path, plot_path

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
                📞 BWE App 2: Direct Speech Restoration & Upsampling
            </h1>
            <p style="color: #9ca3af; font-size: 1.1rem; margin-bottom: 20px; font-family: 'Inter', sans-serif;">
                Upload Narrowband/Telephone Audio ➔ Perform Neural Bandwidth Extension (Upsampling)
            </p>
        </div>
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown(
                """
                ### Inputs
                Upload an existing band-limited telephone/cellular audio recording. Select the model checkpoint to apply.
                """
            )
            t2_input_file = gr.Audio(label="Upload Narrowband/Telephone Audio File", type="filepath")
            t2_input_path = gr.Textbox(label="Or Enter Absolute File Path on Server (Alternative to upload)", placeholder="e.g. C:\\path\\to\\audio.wav")
            t2_model_type = gr.Dropdown(
                choices=["Ours - Lightweight Model", "Domain-Randomized Model", "Baseline G.711 Model"],
                value="Ours - Lightweight Model",
                label="Select BWE Model Checkpoint"
            )
            t2_submit = gr.Button("Perform Bandwidth Extension", variant="primary", elem_classes="primary-button")
            
        with gr.Column(scale=2):
            gr.Markdown("### Results")
            with gr.Row():
                t2_play_in = gr.Audio(label="1. Uploaded Narrowband Input", type="filepath", interactive=False)
                t2_play_enh = gr.Audio(label="2. Enhanced Wideband Speech Output", type="filepath", interactive=False)
                
            t2_plot = gr.Image(label="Spectrogram Comparison (Narrowband vs Enhanced)", type="filepath")
            
    t2_submit.click(
        fn=process_upsampling,
        inputs=[t2_input_file, t2_input_path, t2_model_type],
        outputs=[t2_play_in, t2_play_enh, t2_plot]
    )
    
    gr.HTML(
        """
        <div style="text-align: center; padding: 20px 0; color: #6b7280; font-size: 0.9rem; font-family: 'Inter', sans-serif; border-top: 1px solid #1f2937; margin-top: 30px;">
            HybridGAN-BWE Neural Speech band extension platform • App 2: Direct Upsampling
        </div>
        """
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7862, share=False)
