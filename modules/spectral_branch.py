import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm
from typing import Any

class SpectralEncoder(nn.Module):
    """Spectral branch encoder that processes magnitude and phase spectrograms using 2D CNN."""
    def __init__(self, config_spec: Any):
        super().__init__()
        base_channels = config_spec.channels
        num_layers = config_spec.num_layers
        
        self.convs = nn.ModuleList()
        curr_channels = 2 # Magnitude and Phase stacked as 2 input channels
        
        # Build 2D convolutions that progressively downsample frequency while keeping time resolution
        for idx in range(num_layers):
            next_channels = base_channels * (2 ** idx)
            self.convs.append(
                nn.Sequential(
                    weight_norm(nn.Conv2d(
                        curr_channels, next_channels,
                        kernel_size=(5, 3), stride=(2, 1), padding=(2, 1)
                    )),
                    nn.LeakyReLU(0.1)
                )
            )
            curr_channels = next_channels
            
        # Frequency pooling to squeeze the frequency dimension to 1
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))
        
        # Out channels of the spectral branch
        self.out_channels = curr_channels

    def forward(self, magnitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        """
        Args:
            magnitude: Magnitude spectrogram of shape (batch, freq_bins, time_frames).
            phase: Phase spectrogram of shape (batch, freq_bins, time_frames).
            
        Returns:
            Spectral representations of shape (batch, out_channels, time_frames).
        """
        # Stack along channel dimension: shape (batch, 2, freq_bins, time_frames)
        x = torch.stack([magnitude, phase], dim=1)
        
        for layer in self.convs:
            x = layer(x)
            
        # Reduce frequency dimension to 1: shape (batch, out_channels, 1, time_frames)
        x = self.freq_pool(x)
        
        # Squeeze frequency axis: shape (batch, out_channels, time_frames)
        x = x.squeeze(2)
        return x
