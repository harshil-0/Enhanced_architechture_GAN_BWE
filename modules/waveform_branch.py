import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm
from typing import List, Tuple, Any

class ResBlock1D(nn.Module):
    """Residual block with dilated 1D convolutions."""
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilations: List[int] = [1, 3, 5]
    ):
        super().__init__()
        self.convs1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()
        
        for d in dilations:
            # Padding is adjusted to keep sequence length unchanged: (kernel_size - 1) * dilation // 2
            padding1 = (kernel_size - 1) * d // 2
            padding2 = (kernel_size - 1) // 2
            self.convs1.append(
                weight_norm(nn.Conv1d(
                    channels, channels, kernel_size,
                    stride=1, padding=padding1, dilation=d
                ))
            )
            self.convs2.append(
                weight_norm(nn.Conv1d(
                    channels, channels, kernel_size,
                    stride=1, padding=padding2, dilation=1
                ))
            )
            
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, channels, time_steps).
            
        Returns:
            Output tensor of shape (batch, channels, time_steps).
        """
        for c1, c2 in zip(self.convs1, self.convs2):
            residual = x
            x = self.activation(x)
            x = c1(x)
            x = self.activation(x)
            x = c2(x)
            x = x + residual
        return x


class WaveformEncoder(nn.Module):
    """Waveform encoder that downsamples a time-domain signal into a latent space."""
    def __init__(self, config_enc: Any):
        super().__init__()
        in_channels = config_enc.in_channels
        base_channels = config_enc.channels
        strides = config_enc.strides
        kernel_sizes = config_enc.kernel_sizes
        dilations = config_enc.dilations
        
        # Initial convolution
        self.conv_in = weight_norm(nn.Conv1d(in_channels, base_channels, kernel_size=7, stride=1, padding=3))
        
        # Progressive downsampling blocks
        self.down_blocks = nn.ModuleList()
        curr_channels = base_channels
        
        for stride, k_size in zip(strides, kernel_sizes):
            next_channels = curr_channels * 2
            self.down_blocks.append(
                nn.Sequential(
                    nn.LeakyReLU(0.1),
                    weight_norm(nn.Conv1d(
                        curr_channels, next_channels, kernel_size=k_size,
                        stride=stride, padding=(k_size - 1) // 2
                    )),
                    ResBlock1D(next_channels, kernel_size=3, dilations=dilations)
                )
            )
            curr_channels = next_channels
            
        self.out_channels = curr_channels
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input waveform tensor of shape (batch, 1, samples).
            
        Returns:
            Encoded features of shape (batch, out_channels, frames).
        """
        x = self.conv_in(x)
        for block in self.down_blocks:
            x = block(x)
        return x


class WaveformDecoder(nn.Module):
    """Waveform decoder that upsamples latent representations back into a time-domain waveform."""
    def __init__(self, config_dec: Any, in_channels: int):
        super().__init__()
        upsample_rates = config_dec.upsample_rates
        upsample_kernel_sizes = config_dec.upsample_kernel_sizes
        base_channels = config_dec.channels
        resblock_kernel_sizes = config_dec.resblock_kernel_sizes
        resblock_dilations = config_dec.resblock_dilations
        
        self.conv_in = weight_norm(nn.Conv1d(in_channels, base_channels * (2 ** len(upsample_rates)), kernel_size=3, stride=1, padding=1))
        
        # Progressive upsampling blocks
        self.up_blocks = nn.ModuleList()
        curr_channels = base_channels * (2 ** len(upsample_rates))
        
        for rate, k_size in zip(upsample_rates, upsample_kernel_sizes):
            next_channels = curr_channels // 2
            
            # Upsample block consists of a ConvTranspose1d followed by parallel ResBlocks
            up_conv = weight_norm(nn.ConvTranspose1d(
                curr_channels, next_channels, kernel_size=k_size,
                stride=rate, padding=(k_size - rate) // 2
            ))
            
            res_blocks = nn.ModuleList()
            for r_k_size, r_dilations in zip(resblock_kernel_sizes, resblock_dilations):
                res_blocks.append(ResBlock1D(next_channels, kernel_size=r_k_size, dilations=r_dilations))
                
            self.up_blocks.append(nn.ModuleDict({
                "upsample": up_conv,
                "resblocks": res_blocks
            }))
            
            curr_channels = next_channels
            
        self.activation = nn.LeakyReLU(0.1)
        self.conv_out = weight_norm(nn.Conv1d(curr_channels, 1, kernel_size=7, stride=1, padding=3))
        self.tanh = nn.Tanh()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Latent features of shape (batch, in_channels, frames).
            
        Returns:
            Reconstructed waveform of shape (batch, 1, samples).
        """
        x = self.conv_in(x)
        
        for block in self.up_blocks:
            x = self.activation(x)
            x = block["upsample"](x)
            
            # Sum predictions from multiple parallel residual blocks
            res_out = torch.zeros_like(x)
            for resblock in block["resblocks"]:
                res_out = res_out + resblock(x)
            x = res_out / len(block["resblocks"])
            
        x = self.activation(x)
        x = self.conv_out(x)
        x = self.tanh(x)
        return x
