import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm, spectral_norm
from typing import List, Tuple, Dict, Any
from utils.audio import wav_to_spec

# ==============================================================================
# 1. Multi-Period Discriminator (MPD)
# ==============================================================================

class DiscriminatorP(nn.Module):
    """Sub-discriminator for a specific period of the waveform."""
    def __init__(self, period: int, base_channels: int = 32):
        super().__init__()
        self.period = period
        
        self.convs = nn.ModuleList([
            weight_norm(nn.Conv2d(1, base_channels, (5, 1), (3, 1), padding=(2, 0))),
            weight_norm(nn.Conv2d(base_channels, base_channels * 2, (5, 1), (3, 1), padding=(2, 0))),
            weight_norm(nn.Conv2d(base_channels * 2, base_channels * 4, (5, 1), (3, 1), padding=(2, 0))),
            weight_norm(nn.Conv2d(base_channels * 4, base_channels * 8, (5, 1), (3, 1), padding=(2, 0))),
            weight_norm(nn.Conv2d(base_channels * 8, base_channels * 8, (5, 1), (1, 1), padding=(2, 0))),
        ])
        self.conv_post = weight_norm(nn.Conv2d(base_channels * 8, 1, (3, 1), 1, padding=(1, 0)))
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: Waveform tensor of shape (batch, 1, samples).
            
        Returns:
            Tuple of (validity score, list of intermediate feature maps).
        """
        fmap = []
        # Pad 1D waveform to be divisible by period
        batch, channels, samples = x.shape
        if samples % self.period != 0:
            n_pad = self.period - (samples % self.period)
            x = nn.functional.pad(x, (0, n_pad), "reflect")
            samples = samples + n_pad
            
        # Reshape to 2D grid: (batch, 1, height, width) where width is the period
        x = x.view(batch, channels, samples // self.period, self.period)
        
        for layer in self.convs:
            x = layer(x)
            x = self.activation(x)
            fmap.append(x)
            
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1)
        
        return x, fmap


class MultiPeriodDiscriminator(nn.Module):
    """Collection of DiscriminatorP units looking at different periodicities."""
    def __init__(self, periods: List[int], base_channels: int = 32):
        super().__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorP(p, base_channels) for p in periods
        ])

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        scores = []
        fmaps = []
        for disc in self.discriminators:
            score, fmap = disc(x)
            scores.append(score)
            fmaps.append(fmap)
        return scores, fmaps


# ==============================================================================
# 2. Multi-Scale Discriminator (MSD)
# ==============================================================================

class DiscriminatorS(nn.Module):
    """Sub-discriminator for a specific scale of the waveform."""
    def __init__(self, use_spectral_norm: bool = False, base_channels: int = 32):
        super().__init__()
        norm_fn = spectral_norm if use_spectral_norm else weight_norm
        
        g2 = 4 if (base_channels % 4 == 0) else 1
        g3 = 16 if ((base_channels * 2) % 16 == 0) else 1
        g4 = 64 if ((base_channels * 4) % 64 == 0) else 1
        g5 = 128 if ((base_channels * 8) % 128 == 0) else 1

        self.convs = nn.ModuleList([
            norm_fn(nn.Conv1d(1, base_channels, 15, 1, padding=7)),
            norm_fn(nn.Conv1d(base_channels, base_channels * 2, 41, 4, groups=g2, padding=20)),
            norm_fn(nn.Conv1d(base_channels * 2, base_channels * 4, 41, 4, groups=g3, padding=20)),
            norm_fn(nn.Conv1d(base_channels * 4, base_channels * 8, 41, 4, groups=g4, padding=20)),
            norm_fn(nn.Conv1d(base_channels * 8, base_channels * 8, 41, 4, groups=g5, padding=20)),
            norm_fn(nn.Conv1d(base_channels * 8, base_channels * 8, 5, 1, padding=2)),
        ])
        self.conv_post = norm_fn(nn.Conv1d(base_channels * 8, 1, 3, 1, padding=1))
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        fmap = []
        for layer in self.convs:
            x = layer(x)
            x = self.activation(x)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1)
        return x, fmap


class MultiScaleDiscriminator(nn.Module):
    """Collection of DiscriminatorS units looking at different resolution scales."""
    def __init__(self, scales: List[int], base_channels: int = 32):
        super().__init__()
        self.scales = scales
        self.discriminators = nn.ModuleList()
        for idx, scale in enumerate(scales):
            # HiFi-GAN uses spectral norm on the first discriminator (scale 1) and weight norm on others
            use_sn = (idx == 0)
            self.discriminators.append(DiscriminatorS(use_spectral_norm=use_sn, base_channels=base_channels))
            
    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        scores = []
        fmaps = []
        for idx, disc in enumerate(self.discriminators):
            scale = self.scales[idx]
            if scale > 1:
                # Average pooling downsampling
                x_scaled = nn.functional.avg_pool1d(x, kernel_size=scale, stride=scale, padding=0)
            else:
                x_scaled = x
            score, fmap = disc(x_scaled)
            scores.append(score)
            fmaps.append(fmap)
        return scores, fmaps


# ==============================================================================
# 3. Spectral Discriminators (Magnitude and Phase)
# ==============================================================================

class Spectrogram2DDiscriminator(nn.Module):
    """Discriminator that operates on 2D Spectrogram inputs (Magnitude or Real/Imag complex features)."""
    def __init__(self, in_channels: int, base_channels: int = 32):
        super().__init__()
        self.convs = nn.ModuleList([
            weight_norm(nn.Conv2d(in_channels, base_channels, kernel_size=(5, 5), stride=(2, 2), padding=2)),
            weight_norm(nn.Conv2d(base_channels, base_channels * 2, kernel_size=(5, 5), stride=(2, 2), padding=2)),
            weight_norm(nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=(5, 5), stride=(2, 2), padding=2)),
            weight_norm(nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=(5, 5), stride=(2, 2), padding=2)),
        ])
        self.conv_post = weight_norm(nn.Conv2d(base_channels * 8, 1, kernel_size=(3, 3), stride=1, padding=1))
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: 2D Spectrogram features of shape (batch, in_channels, freq_bins, time_frames).
            
        Returns:
            Tuple of (validity score, list of intermediate feature maps).
        """
        fmap = []
        for layer in self.convs:
            x = layer(x)
            x = self.activation(x)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1)
        return x, fmap


# ==============================================================================
# 4. Multi-Discriminator Group Wrapper
# ==============================================================================

class DiscriminatorGroup(nn.Module):
    """Unified wrapper grouping Waveform (MPD, MSD) and Spectral discriminators."""
    def __init__(self, config: Any):
        super().__init__()
        self.config = config
        
        # 1. MPD
        self.mpd = MultiPeriodDiscriminator(
            periods=config.discriminator.mpd.periods,
            base_channels=config.discriminator.mpd.channels
        )
        
        # 2. MSD
        self.msd = MultiScaleDiscriminator(
            scales=config.discriminator.msd.scales,
            base_channels=config.discriminator.msd.channels
        )
        
        # 3. Spectral Magnitude & Phase discriminators
        self.use_magnitude = config.discriminator.spectral.use_magnitude
        self.use_phase = config.discriminator.spectral.use_phase
        
        if self.use_magnitude or self.use_phase:
            self.fft_sizes = config.discriminator.spectral.fft_sizes
            self.hop_lengths = config.discriminator.spectral.hop_lengths
            self.win_lengths = config.discriminator.spectral.win_lengths
            self.spectral_channels = config.discriminator.spectral.channels
            
            # We instantiate a Spectrogram2DDiscriminator for each STFT resolution
            if self.use_magnitude:
                self.mag_discs = nn.ModuleList([
                    Spectrogram2DDiscriminator(in_channels=1, base_channels=self.spectral_channels)
                    for _ in self.fft_sizes
                ])
                
            if self.use_phase:
                # Phase discriminator uses complex STFT input: Real & Imag parts concatenated (2 channels)
                self.phase_discs = nn.ModuleList([
                    Spectrogram2DDiscriminator(in_channels=2, base_channels=self.spectral_channels)
                    for _ in self.fft_sizes
                ])

    def forward(
        self,
        y: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        """Runs the waveform through all sub-discriminators.
        
        Args:
            y: Audio waveform tensor of shape (batch, 1, samples).
            
        Returns:
            Tuple of (scores list, feature maps list of lists).
        """
        scores = []
        fmaps = []
        
        # 1. Run MPD
        mpd_scores, mpd_fmaps = self.mpd(y)
        scores.extend(mpd_scores)
        fmaps.extend(mpd_fmaps)
        
        # 2. Run MSD
        msd_scores, msd_fmaps = self.msd(y)
        scores.extend(msd_scores)
        fmaps.extend(msd_fmaps)
        
        # 3. Run Spectral Discriminators
        if self.use_magnitude or self.use_phase:
            for idx, fft_size in enumerate(self.fft_sizes):
                hop = self.hop_lengths[idx]
                win = self.win_lengths[idx]
                
                # Compute STFT
                # Output complex tensor: shape (batch, freq_bins, time_frames)
                stft_complex = torch.stft(
                    y.squeeze(1),
                    n_fft=fft_size,
                    hop_length=hop,
                    win_length=win,
                    window=torch.hann_window(win, device=y.device),
                    center=True,
                    pad_mode='reflect',
                    normalized=False,
                    onesided=True,
                    return_complex=True
                )
                
                if self.use_magnitude:
                    # Magnitude: shape (batch, 1, freq, time)
                    mag = torch.sqrt(stft_complex.real**2 + stft_complex.imag**2 + 1e-12).unsqueeze(1)
                    mag_score, mag_fmap = self.mag_discs[idx](mag)
                    scores.append(mag_score)
                    fmaps.append(mag_fmap)
                    
                if self.use_phase:
                    # Phase: shape (batch, 2, freq, time) by stacking real and imag parts
                    real_imag = torch.stack([stft_complex.real, stft_complex.imag], dim=1)
                    phase_score, phase_fmap = self.phase_discs[idx](real_imag)
                    scores.append(phase_score)
                    fmaps.append(phase_fmap)
                    
        return scores, fmaps
