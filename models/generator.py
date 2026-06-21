import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from typing import Any, Tuple
from modules.waveform_branch import WaveformEncoder, WaveformDecoder

class Generator(nn.Module):
    """Hybrid GAN-BWE Generator with modular waveform and spectral branches."""
    def __init__(self, config: Any):
        super().__init__()
        self.config = config
        self.use_waveform = config.generator.use_waveform_branch
        self.use_spectral = config.generator.use_spectral_branch
        self.use_attention = config.generator.use_cross_attention
        self.gradient_checkpointing = config.train.gradient_checkpointing

        if not self.use_waveform and not self.use_spectral:
            raise ValueError("At least one of the branches (waveform or spectral) must be enabled.")

        # 1. Waveform Branch
        if self.use_waveform:
            self.waveform_encoder = WaveformEncoder(config.generator.waveform_encoder)
            latent_channels = self.waveform_encoder.out_channels
        else:
            latent_channels = 0

        # 2. Spectral Branch (Stub for baseline phase; implemented in Phase 8)
        if self.use_spectral:
            # We import dynamically to keep modules independent and prevent circular dependencies
            try:
                from modules.spectral_branch import SpectralEncoder
                self.spectral_encoder = SpectralEncoder(config.generator.spectral_encoder)
                spec_channels = self.spectral_encoder.out_channels
            except ImportError:
                # Fallback mock for early testing if spectral_branch is not yet created
                self.spectral_encoder = None
                spec_channels = 128
        else:
            spec_channels = 0

        # 3. Fusion Branch
        if self.use_waveform and self.use_spectral:
            if self.use_attention:
                try:
                    from modules.attention import CrossAttentionFusion
                    self.fusion = CrossAttentionFusion(
                        config.generator.attention, 
                        wave_channels=latent_channels, 
                        spec_channels=spec_channels
                    )
                    self.latent_dim = self.fusion.out_channels
                except ImportError:
                    # Fallback simple concat if attention module not yet created
                    self.fusion = None
                    self.latent_dim = latent_channels + spec_channels
            else:
                self.latent_dim = latent_channels + spec_channels
        elif self.use_waveform:
            self.latent_dim = latent_channels
        else:
            self.latent_dim = spec_channels

        # 4. Decoder
        self.decoder = WaveformDecoder(config.generator.decoder, in_channels=self.latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Degraded/Narrowband input waveform of shape (batch, 1, samples).
            
        Returns:
            Wideband reconstructed waveform of shape (batch, 1, samples).
        """
        feat_wave = None
        feat_spec = None

        # Process Waveform branch
        if self.use_waveform:
            if self.gradient_checkpointing and self.training:
                feat_wave = checkpoint.checkpoint(self.waveform_encoder, x, use_reentrant=False)
            else:
                feat_wave = self.waveform_encoder(x)

        # Process Spectral branch
        if self.use_spectral:
            from utils.audio import wav_to_spec
            # Compute STFT on input waveform
            mag, phase = wav_to_spec(
                x,
                fft_size=self.config.generator.spectral_encoder.fft_size,
                hop_length=self.config.generator.spectral_encoder.hop_length,
                win_length=self.config.generator.spectral_encoder.win_length
            )
            
            if self.gradient_checkpointing and self.training:
                feat_spec = checkpoint.checkpoint(self.spectral_encoder, mag, phase, use_reentrant=False)
            else:
                feat_spec = self.spectral_encoder(mag, phase)

        # Fuse features
        if self.use_waveform and self.use_spectral:
            if self.use_attention and self.fusion is not None:
                if self.gradient_checkpointing and self.training:
                    latent = checkpoint.checkpoint(self.fusion, feat_wave, feat_spec, use_reentrant=False)
                else:
                    latent = self.fusion(feat_wave, feat_spec)
            else:
                # If sizes differ along the time axis, interpolate spectral features to match waveform features
                if feat_wave.shape[-1] != feat_spec.shape[-1]:
                    feat_spec = nn.functional.interpolate(
                        feat_spec, size=feat_wave.shape[-1], mode='linear', align_corners=False
                    )
                latent = torch.cat([feat_wave, feat_spec], dim=1)
        elif self.use_waveform:
            latent = feat_wave
        else:
            latent = feat_spec

        # Decode features back to waveform
        if self.gradient_checkpointing and self.training:
            out = checkpoint.checkpoint(self.decoder, latent, use_reentrant=False)
        else:
            out = self.decoder(latent)
            
        # Optional residual skip connection to add low frequencies directly
        # Since x is low-pass filtered, we can add it to output to preserve low frequency phase
        # The generator learns to add high-frequency details.
        return out
