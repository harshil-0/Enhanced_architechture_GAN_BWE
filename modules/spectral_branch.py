import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm
from typing import Any

class SpectralEncoder(nn.Module):
    """Modular Spectral branch encoder that processes magnitude and phase spectrograms.
    Dynamically supports both the original standard 2D CNN architecture and the new
    Dual-Path (lightweight 2D spatial + 1D temporal) architecture for full backward compatibility.
    """
    def __init__(self, config_spec: Any):
        super().__init__()
        base_channels = config_spec.channels  # e.g., 32
        num_layers = config_spec.num_layers  # e.g., 4
        
        # Flag to toggle between old standard 2D and new Dual-Path architectures
        self.use_dual_path = True
        
        # -------------------------------------------------------------
        # 1. Old/Baseline Path: Standard 2D CNN
        # -------------------------------------------------------------
        self.convs = nn.ModuleList()
        curr_channels = 2
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
        self.freq_pool_old = nn.AdaptiveAvgPool2d((1, None))
        
        # -------------------------------------------------------------
        # 2. New Path: Dual-Path Lightweight CNN (Path A + Path B)
        # -------------------------------------------------------------
        # We split the output channel capacity equally between 2D and 1D paths
        original_out = base_channels * (2 ** (num_layers - 1))
        self.path_out_channels = original_out // 2
        
        # Path A: Lightweight 2D Spatial Path (ch_2d is halved to save parameters)
        self.path_2d = nn.ModuleList()
        curr_channels_2d = 2
        ch_2d = base_channels // 2
        for idx in range(num_layers):
            next_channels = ch_2d * (2 ** idx)
            self.path_2d.append(
                nn.Sequential(
                    weight_norm(nn.Conv2d(
                        curr_channels_2d, next_channels,
                        kernel_size=(5, 3), stride=(2, 1), padding=(2, 1)
                    )),
                    nn.LeakyReLU(0.1)
                )
            )
            curr_channels_2d = next_channels
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))
        
        # Path B: Squeezed 1D Temporal Path
        self.freq_pool_input = nn.AdaptiveAvgPool2d((1, None))
        self.path_1d = nn.Sequential(
            weight_norm(nn.Conv1d(2, 64, kernel_size=3, padding=1)),
            nn.LeakyReLU(0.1),
            weight_norm(nn.Conv1d(64, 128, kernel_size=3, padding=1)),
            nn.LeakyReLU(0.1),
            weight_norm(nn.Conv1d(128, self.path_out_channels, kernel_size=3, padding=1)),
            nn.LeakyReLU(0.1)
        )
        
        # Set output channel count for Generator branch fusion alignment
        self.out_channels = 256

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """Hook to auto-detect if the loaded checkpoint contains the old 'convs' layers
        or the new dual-path layers, toggle the use_dual_path flag accordingly,
        and filter out expected missing keys from the strict error list.
        """
        has_old_convs = False
        for key in state_dict.keys():
            if key.startswith(prefix + "convs"):
                has_old_convs = True
                break
        
        if has_old_convs:
            self.use_dual_path = False
        else:
            self.use_dual_path = True
            
        # Get list size before loading
        old_missing_len = len(missing_keys)
        
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)
                                      
        # Filter out keys that are expected to be missing based on loaded model type
        if self.use_dual_path:
            # Lightweight model: old 2D path keys will be reported missing by PyTorch
            to_remove = [k for k in missing_keys[old_missing_len:] if k.startswith(prefix + "convs") or k.startswith(prefix + "freq_pool_old")]
            for k in to_remove:
                missing_keys.remove(k)
        else:
            # Baseline model: new dual path keys will be reported missing by PyTorch
            to_remove = [k for k in missing_keys[old_missing_len:] if k.startswith(prefix + "path_2d") or k.startswith(prefix + "path_1d") or k.startswith(prefix + "freq_pool")]
            for k in to_remove:
                missing_keys.remove(k)

    def forward(self, magnitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        """
        Args:
            magnitude: Magnitude spectrogram of shape (batch, freq_bins, time_frames).
            phase: Phase spectrogram of shape (batch, freq_bins, time_frames).
            
        Returns:
            Spectral representations of shape (batch, out_channels, time_frames).
        """
        # Stack along channel dimension: shape (batch, 2, freq_bins, time_frames)
        x_in = torch.stack([magnitude, phase], dim=1)
        
        if not self.use_dual_path:
            # Execute standard 2D path
            x = x_in
            for layer in self.convs:
                x = layer(x)
            out = self.freq_pool_old(x).squeeze(2)
            return out
        else:
            # Execute Dual-Path
            # 1. Process Path A (2D)
            x_2d = x_in
            for layer in self.path_2d:
                x_2d = layer(x_2d)
            x_2d = self.freq_pool(x_2d).squeeze(2)
            
            # 2. Process Path B (1D)
            x_1d = self.freq_pool_input(x_in).squeeze(2)
            x_1d = self.path_1d(x_1d)
            
            # 3. Fuse by concatenation
            out = torch.cat([x_2d, x_1d], dim=1)
            return out
