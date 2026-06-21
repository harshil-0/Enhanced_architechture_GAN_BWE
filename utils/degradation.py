import math
import torch
import torchaudio
import scipy.signal
import numpy as np
from typing import Union, Any

def alaw_encode(x: torch.Tensor, A: float = 87.6) -> torch.Tensor:
    """Apply A-law compression to a waveform.
    
    Args:
        x: Input tensor with values in [-1.0, 1.0].
        A: A-law parameter (default 87.6).
        
    Returns:
        Encoded tensor in [-1.0, 1.0].
    """
    sgn = torch.sign(x)
    abs_x = torch.abs(x)
    
    mask = abs_x < (1.0 / A)
    
    encoded = torch.empty_like(x)
    encoded[mask] = (A * abs_x[mask]) / (1.0 + math.log(A))
    encoded[~mask] = (1.0 + torch.log(A * abs_x[~mask])) / (1.0 + math.log(A))
    
    # Quantize to 8-bit resolution (256 levels uniform quantization in compressed domain)
    quantized = torch.round(encoded * 127.0) / 127.0
    return sgn * quantized


def alaw_decode(y: torch.Tensor, A: float = 87.6) -> torch.Tensor:
    """Apply A-law expansion (decoding) to a compressed waveform.
    
    Args:
        y: Compressed tensor with values in [-1.0, 1.0].
        A: A-law parameter (default 87.6).
        
    Returns:
        Decoded tensor.
    """
    sgn = torch.sign(y)
    abs_y = torch.abs(y)
    
    threshold = 1.0 / (1.0 + math.log(A))
    mask = abs_y < threshold
    
    decoded = torch.empty_like(y)
    decoded[mask] = (abs_y[mask] * (1.0 + math.log(A))) / A
    decoded[~mask] = torch.exp(abs_y[~mask] * (1.0 + math.log(A)) - 1.0) / A
    
    return sgn * decoded


def apply_g711_mu(wav: torch.Tensor) -> torch.Tensor:
    """Simulate G.711 Mu-law codec.
    
    Args:
        wav: Waveform tensor.
        
    Returns:
        Degraded waveform tensor.
    """
    # torchaudio expectations: values in [-1.0, 1.0]
    orig_max = torch.max(torch.abs(wav))
    if orig_max > 0:
        wav = wav / orig_max
        
    quantized = torchaudio.functional.mu_law_encoding(wav, quantization_channels=256)
    decoded = torchaudio.functional.mu_law_decoding(quantized, quantization_channels=256)
    
    if orig_max > 0:
        decoded = decoded * orig_max
    return decoded


def apply_g711_a(wav: torch.Tensor) -> torch.Tensor:
    """Simulate G.711 A-law codec.
    
    Args:
        wav: Waveform tensor.
        
    Returns:
        Degraded waveform tensor.
    """
    orig_max = torch.max(torch.abs(wav))
    if orig_max > 0:
        wav = wav / orig_max
        
    encoded = alaw_encode(wav)
    decoded = alaw_decode(encoded)
    
    if orig_max > 0:
        decoded = decoded * orig_max
    return decoded


def apply_gsm_sim(wav: torch.Tensor, sr: int) -> torch.Tensor:
    """Simulate GSM-style speech codec degradation:
    - Bandpass filtering (300 Hz - 3400 Hz)
    - 13-bit quantization.
    
    Args:
        wav: Waveform tensor.
        sr: Sample rate of the waveform.
        
    Returns:
        Degraded waveform tensor.
    """
    # Convert to numpy for filtering
    wav_np = wav.cpu().numpy()
    
    # Design Bandpass filter (Butterworth)
    # GSM uses 300 to 3400 Hz bandpass
    nyq = 0.5 * sr
    low = 300.0 / nyq
    high = 3400.0 / nyq
    
    # Guard against invalid frequencies
    if high >= 1.0:
        high = 0.99
        
    b, a = scipy.signal.butter(4, [low, high], btype='band')
    
    # Filter along the last dimension
    filtered_np = scipy.signal.filtfilt(b, a, wav_np, axis=-1)
    filtered = torch.from_numpy(filtered_np.copy()).to(device=wav.device, dtype=torch.float32)
    
    # 13-bit PCM uniform quantization
    orig_max = torch.max(torch.abs(filtered))
    if orig_max > 0:
        norm_filtered = filtered / orig_max
        # Quantize to 13-bit (range -4096 to 4095)
        levels = 2**12
        quantized = torch.round(norm_filtered * levels) / levels
        filtered = quantized * orig_max
        
    return filtered


def apply_random_highpass(wav: torch.Tensor, sr: int) -> torch.Tensor:
    """Apply a random Butterworth high-pass filter (cutoff between 50 and 300 Hz) to simulate telephone microphone cut-off.
    
    Args:
        wav: Waveform tensor of shape (batch, 1, samples) or similar.
        sr: Sample rate.
    """
    wav_np = wav.cpu().numpy()
    nyq = 0.5 * sr
    
    # Pick a random cutoff frequency between 50 Hz and 300 Hz
    cutoff = np.random.uniform(50.0, 300.0)
    low = cutoff / nyq
    
    b, a = scipy.signal.butter(4, low, btype='high')
    
    # Filter along the last dimension
    filtered_np = scipy.signal.filtfilt(b, a, wav_np, axis=-1)
    return torch.from_numpy(filtered_np.copy()).to(device=wav.device, dtype=torch.float32)


def add_random_noise(wav: torch.Tensor) -> torch.Tensor:
    """Add a small amount of random white Gaussian noise to simulate line static/noise.
    
    Args:
        wav: Waveform tensor.
    """
    snr_db = np.random.uniform(25.0, 50.0)
    sig_power = torch.mean(wav ** 2)
    if sig_power > 0:
        noise_power = sig_power / (10 ** (snr_db / 10.0))
        noise = torch.randn_like(wav) * torch.sqrt(noise_power)
        return wav + noise
    return wav


class DegradationPipeline:
    """Class to apply configured degradation chain to audio tensors."""
    def __init__(self, config_audio: Any):
        self.input_sr = config_audio.input_sr
        self.target_sr = config_audio.target_sr
        self.degradation_type = config_audio.degradation_type
        
        # Setup resamplers if sample rates differ
        if self.target_sr != self.input_sr:
            self.downsampler = torchaudio.transforms.Resample(
                orig_freq=self.target_sr,
                new_freq=self.input_sr
            )
            self.upsampler = torchaudio.transforms.Resample(
                orig_freq=self.input_sr,
                new_freq=self.target_sr
            )
        else:
            self.downsampler = None
            self.upsampler = None
            
    def __call__(self, wav: torch.Tensor) -> torch.Tensor:
        """Apply degradation pipeline.
        
        Args:
            wav: High-frequency/Wideband Ground Truth waveform, shape (batch, 1, samples) or (batch, samples).
            
        Returns:
            Narrowband/Degraded waveform resampled back to target_sr.
        """
        # Step 1: Resample to telephone rate (usually 8 kHz)
        if self.downsampler is not None:
            nb_wav = self.downsampler(wav)
        else:
            nb_wav = wav.clone()
            
        # Step 2: Apply codec degradation at telephone rate
        if self.degradation_type == "g711_mu":
            nb_wav = apply_g711_mu(nb_wav)
        elif self.degradation_type == "g711_a":
            nb_wav = apply_g711_a(nb_wav)
        elif self.degradation_type == "gsm_sim":
            nb_wav = apply_gsm_sim(nb_wav, self.input_sr)
        elif self.degradation_type == "none":
            pass # Just resampling
        elif self.degradation_type == "dynamic":
            # 1. Randomly choose a codec
            p = np.random.rand()
            chosen_codec = "none"
            if p < 0.3:
                chosen_codec = "g711_mu"
                nb_wav = apply_g711_mu(nb_wav)
            elif p < 0.6:
                chosen_codec = "g711_a"
                nb_wav = apply_g711_a(nb_wav)
            elif p < 0.9:
                chosen_codec = "gsm_sim"
                nb_wav = apply_gsm_sim(nb_wav, self.input_sr)
            else:
                chosen_codec = "none" # pure resampling
                
            # 2. Randomly apply high-pass filtering (only for G.711 / pure resampling)
            if chosen_codec != "gsm_sim" and np.random.rand() < 0.8:
                nb_wav = apply_random_highpass(nb_wav, self.input_sr)
                
            # 3. Randomly add line noise
            if np.random.rand() < 0.7:
                nb_wav = add_random_noise(nb_wav)
        else:
            raise ValueError(f"Unknown degradation type: {self.degradation_type}")
            
        # Step 3: Resample back to target_sr
        if self.upsampler is not None:
            degraded_wav = self.upsampler(nb_wav)
        else:
            degraded_wav = nb_wav
            
        # Ensure sizes match exactly (resampling can introduce off-by-one sample count)
        if degraded_wav.shape[-1] != wav.shape[-1]:
            diff = wav.shape[-1] - degraded_wav.shape[-1]
            if diff > 0:
                degraded_wav = torch.nn.functional.pad(degraded_wav, (0, diff), mode='replicate')
            elif diff < 0:
                degraded_wav = degraded_wav[..., :wav.shape[-1]]
                
        return degraded_wav
