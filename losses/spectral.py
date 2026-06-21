import torch
import torch.nn as nn
import torchaudio
from typing import List, Tuple, Any

class STFTLoss(nn.Module):
    """Single-resolution STFT loss (Spectral Convergence + Log Magnitude)."""
    def __init__(self, fft_size: int, hop_length: int, win_length: int):
        super().__init__()
        self.fft_size = fft_size
        self.hop_length = hop_length
        self.win_length = win_length
        self.register_buffer("window", torch.hann_window(win_length))

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            y_pred: Reconstructed waveform of shape (batch, 1, samples).
            y_true: Ground-truth waveform of shape (batch, 1, samples).
            
        Returns:
            Tuple of (spectral convergence loss, log magnitude loss).
        """
        # Ensure window is on same device
        self.window = self.window.to(y_pred.device)
        
        # Squeeze channel dim
        p_wav = y_pred.squeeze(1)
        t_wav = y_true.squeeze(1)
        
        # Compute STFT
        p_stft = torch.stft(
            p_wav, self.fft_size, self.hop_length, self.win_length,
            window=self.window, center=True, pad_mode='reflect',
            normalized=False, onesided=True, return_complex=True
        )
        t_stft = torch.stft(
            t_wav, self.fft_size, self.hop_length, self.win_length,
            window=self.window, center=True, pad_mode='reflect',
            normalized=False, onesided=True, return_complex=True
        )
        
        p_mag = torch.sqrt(p_stft.real**2 + p_stft.imag**2 + 1e-12)
        t_mag = torch.sqrt(t_stft.real**2 + t_stft.imag**2 + 1e-12)
        
        # 1. Spectral Convergence Loss
        # SC = || |STFT_true| - |STFT_pred| ||_F / || |STFT_true| ||_F
        # Stabilized Frobenius norm computation using sum of squares + eps
        sc_numerator = torch.sqrt(torch.sum((t_mag - p_mag)**2, dim=(-2, -1)) + 1e-12)
        sc_denominator = torch.sqrt(torch.sum(t_mag**2, dim=(-2, -1)) + 1e-12)
        
        sc_loss = torch.mean(sc_numerator / sc_denominator)
        
        # 2. Log Magnitude Loss
        # LogMag = L1( log(|STFT_true| + eps) - log(|STFT_pred| + eps) )
        log_p_mag = torch.log(p_mag + 1e-7)
        log_t_mag = torch.log(t_mag + 1e-7)
        log_mag_loss = torch.mean(torch.abs(log_t_mag - log_p_mag))
        
        return sc_loss, log_mag_loss


class MultiResolutionSTFTLoss(nn.Module):
    """Multi-resolution STFT loss summing across multiple window scales."""
    def __init__(
        self,
        fft_sizes: List[int] = [512, 1024, 2048],
        hop_lengths: List[int] = [50, 120, 240],
        win_lengths: List[int] = [240, 600, 1200]
    ):
        super().__init__()
        assert len(fft_sizes) == len(hop_lengths) == len(win_lengths)
        self.losses = nn.ModuleList([
            STFTLoss(f, h, w) for f, h, w in zip(fft_sizes, hop_lengths, win_lengths)
        ])

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sc_loss_total = 0.0
        log_mag_loss_total = 0.0
        for loss_fn in self.losses:
            sc_loss, log_mag_loss = loss_fn(y_pred, y_true)
            sc_loss_total = sc_loss_total + sc_loss
            log_mag_loss_total = log_mag_loss_total + log_mag_loss
            
        # Average across resolutions
        num_resolutions = len(self.losses)
        return sc_loss_total / num_resolutions, log_mag_loss_total / num_resolutions


class PhaseConsistencyLoss(nn.Module):
    """Loss for matching STFT phase angles using differentiable cosine and sine difference."""
    def __init__(self, fft_size: int = 512, hop_length: int = 128, win_length: int = 512):
        super().__init__()
        self.fft_size = fft_size
        self.hop_length = hop_length
        self.win_length = win_length
        self.register_buffer("window", torch.hann_window(win_length))

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        self.window = self.window.to(y_pred.device)
        
        p_stft = torch.stft(
            y_pred.squeeze(1), self.fft_size, self.hop_length, self.win_length,
            window=self.window, center=True, pad_mode='reflect',
            normalized=False, onesided=True, return_complex=True
        )
        t_stft = torch.stft(
            y_true.squeeze(1), self.fft_size, self.hop_length, self.win_length,
            window=self.window, center=True, pad_mode='reflect',
            normalized=False, onesided=True, return_complex=True
        )
        
        # Avoid torch.angle which has unstable derivatives at zero
        p_mag = torch.sqrt(p_stft.real**2 + p_stft.imag**2 + 1e-12)
        t_mag = torch.sqrt(t_stft.real**2 + t_stft.imag**2 + 1e-12)
        
        p_cos = p_stft.real / p_mag
        p_sin = p_stft.imag / p_mag
        
        t_cos = t_stft.real / t_mag
        t_sin = t_stft.imag / t_mag
        
        # Cosine/sine difference without computing the angle directly
        cos_diff = torch.mean(torch.abs(t_cos - p_cos))
        sin_diff = torch.mean(torch.abs(t_sin - p_sin))
        
        return cos_diff + sin_diff


class MelSpectrogramLoss(nn.Module):
    """L1 Loss calculated on Mel Spectrogram representations."""
    def __init__(
        self,
        sample_rate: int,
        fft_size: int = 1024,
        hop_length: int = 128,
        win_length: int = 1024,
        num_mels: int = 80
    ):
        super().__init__()
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=fft_size,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=num_mels,
            power=1.0,
            normalized=False,
            center=True,
            pad_mode="reflect",
            onesided=True
        )

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # Move mel transform to the target device
        self.mel_transform = self.mel_transform.to(y_pred.device)
        
        # Squeeze channel dimensions for torchaudio transform
        p_mel = self.mel_transform(y_pred.squeeze(1))
        t_mel = self.mel_transform(y_true.squeeze(1))
        
        # Log scale Mel Spectrograms
        p_mel_log = torch.log(torch.clamp(p_mel, min=1e-7))
        t_mel_log = torch.log(torch.clamp(t_mel, min=1e-7))
        
        return torch.mean(torch.abs(t_mel_log - p_mel_log))
