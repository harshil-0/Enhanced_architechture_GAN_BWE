import torch
import torch.nn as nn
import torch.nn.functional as F

def wav_to_spec(
    wav: torch.Tensor,
    fft_size: int,
    hop_length: int,
    win_length: int,
    window: torch.Tensor = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert waveform to magnitude and phase spectrograms.
    
    Args:
        wav: Time-domain signal tensor of shape (batch, 1, samples) or (batch, samples).
        fft_size: FFT window size.
        hop_length: Hop length between frames.
        win_length: Analysis window length.
        window: Pre-created window tensor (optional).
        
    Returns:
        magnitude: Magnitude spectrogram tensor.
        phase: Phase spectrogram tensor.
    """
    if wav.dim() == 3:
        wav = wav.squeeze(1) # (batch, samples)
        
    if window is None:
        window = torch.hann_window(win_length, device=wav.device)
        
    # STFT returns a complex tensor
    stft = torch.stft(
        wav,
        n_fft=fft_size,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        pad_mode='reflect',
        normalized=False,
        onesided=True,
        return_complex=True
    )
    
    magnitude = torch.abs(stft)
    phase = torch.angle(stft)
    return magnitude, phase


def spec_to_wav(
    magnitude: torch.Tensor,
    phase: torch.Tensor,
    fft_size: int,
    hop_length: int,
    win_length: int,
    window: torch.Tensor = None
) -> torch.Tensor:
    """Convert magnitude and phase spectrograms back to time-domain waveform.
    
    Args:
        magnitude: Magnitude spectrogram tensor.
        phase: Phase spectrogram tensor.
        fft_size: FFT window size.
        hop_length: Hop length between frames.
        win_length: Analysis window length.
        window: Pre-created window tensor (optional).
        
    Returns:
        wav: Reconstructed time-domain waveform of shape (batch, samples).
    """
    if window is None:
        window = torch.hann_window(win_length, device=magnitude.device)
        
    # Reconstruct complex spectrogram
    stft = torch.polar(magnitude, phase)
    
    wav = torch.istft(
        stft,
        n_fft=fft_size,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        normalized=False,
        onesided=True
    )
    
    return wav.unsqueeze(1) # (batch, 1, samples)


def pad_to_multiple(wav: torch.Tensor, multiple: int) -> torch.Tensor:
    """Pad waveform time-dimension to be a multiple of a given divisor.
    
    Args:
        wav: Tensor of shape (batch, channels, samples) or (batch, samples).
        multiple: The integer the length should be a multiple of.
        
    Returns:
        Padded tensor.
    """
    length = wav.shape[-1]
    remainder = length % multiple
    if remainder == 0:
        return wav
    
    pad_len = multiple - remainder
    pad_left = pad_len // 2
    pad_right = pad_len - pad_left
    
    # F.pad expects padding list in reverse order of dimensions (last dim padding: left, right)
    return F.pad(wav, (pad_left, pad_right), mode='reflect')
