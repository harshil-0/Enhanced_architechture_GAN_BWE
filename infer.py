import os
import argparse
import torch
import torchaudio
from tqdm import tqdm
from utils.config import load_config
from models.generator import Generator

def main():
    parser = argparse.ArgumentParser(description="Inference and Export Script for HybridGAN-BWE")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config YAML file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to generator model checkpoint (.pth)")
    parser.add_argument("--input", type=str, default=None, help="Path to a single WAV file or folder of WAV files")
    parser.add_argument("--output_dir", type=str, default="outputs/inference_results", help="Directory to save enhanced wideband outputs")
    parser.add_argument("--export_onnx", type=str, default=None, help="If set, export the generator model to this path (ONNX format)")
    parser.add_argument("--export_torchscript", type=str, default=None, help="If set, export the generator model to this path (TorchScript format)")
    args = parser.parse_args()

    # 1. Load config
    config = load_config(args.config)
    target_sr = config.audio.target_sr
    
    # 2. Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 3. Load Generator
    print("Loading Generator model...")
    generator = Generator(config)
    
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if "generator_state" in state:
        generator.load_state_dict(state["generator_state"])
    else:
        generator.load_state_dict(state)
        
    generator = generator.to(device)
    generator.eval()
    print("Checkpoint loaded successfully.")

    # 4. Handle JIT/ONNX Export
    dummy_input = torch.randn(1, 1, config.audio.segment_length, device=device)
    
    if args.export_torchscript:
        print(f"Exporting model to TorchScript: {args.export_torchscript}...")
        try:
            # We trace the model using the dummy input representation
            traced_model = torch.jit.trace(generator, dummy_input)
            os.makedirs(os.path.dirname(os.path.abspath(args.export_torchscript)), exist_ok=True)
            traced_model.save(args.export_torchscript)
            print(f"TorchScript model saved successfully at: {args.export_torchscript}")
        except Exception as e:
            print(f"Error exporting TorchScript: {e}")

    if args.export_onnx:
        print(f"Exporting model to ONNX: {args.export_onnx}...")
        try:
            os.makedirs(os.path.dirname(os.path.abspath(args.export_onnx)), exist_ok=True)
            torch.onnx.export(
                generator,
                dummy_input,
                args.export_onnx,
                input_names=["input_audio"],
                output_names=["enhanced_audio"],
                dynamic_axes={
                    "input_audio": {2: "num_samples"},
                    "enhanced_audio": {2: "num_samples"}
                },
                opset_version=17
            )
            print(f"ONNX model saved successfully at: {args.export_onnx}")
        except Exception as e:
            print(f"Error exporting ONNX: {e}")

    # 5. Handle Audio Inference
    if args.input is None:
        if not args.export_onnx and not args.export_torchscript:
            print("Warning: No input audio or export target provided. Exiting.")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check if input is directory or file
    if os.path.isdir(args.input):
        audio_files = [
            os.path.join(args.input, f) for f in os.listdir(args.input)
            if f.lower().endswith((".wav", ".flac", ".mp3"))
        ]
        print(f"Found {len(audio_files)} audio files in folder '{args.input}'")
    else:
        audio_files = [args.input]
        print(f"Processing single file: {args.input}")

    # Run inference
    with torch.no_grad():
        for filepath in tqdm(audio_files, desc="Enhancing audio"):
            # Load audio
            wav, sr = torchaudio.load(filepath)
            
            # Convert to mono if multi-channel
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
                
            # If input is at high rate, we downsample it to NB first to simulate degradation
            # If it's already at NB, we upsample to target_sr to feed the model
            if sr == target_sr:
                # Ground truth wideband: we simulate NB first
                from utils.degradation import DegradationPipeline
                degrader = DegradationPipeline(config.audio)
                input_wav = degrader(wav)
            else:
                # NB input: we resample to target_sr
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
                input_wav = resampler(wav)
                
            # Pad to multiple of 256 for STFT compatibility
            from utils.audio import pad_to_multiple
            input_wav = pad_to_multiple(input_wav, 256)
            
            # Add batch dimension: shape (1, 1, samples)
            input_tensor = input_wav.unsqueeze(0).to(device)
            
            # Forward pass
            enhanced_tensor = generator(input_tensor)
            
            # Post-processing
            enhanced_wav = enhanced_tensor.squeeze(0).cpu()
            
            # Save enhanced wideband output
            base_name = os.path.basename(filepath)
            out_path = os.path.join(args.output_dir, f"enhanced_{base_name}")
            torchaudio.save(out_path, enhanced_wav, target_sr)

    print(f"Inference complete. Enhanced audio files saved in: {args.output_dir}")

if __name__ == "__main__":
    main()
