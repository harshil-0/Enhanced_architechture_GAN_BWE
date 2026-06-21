import torch
import torch.nn as nn
from typing import Any

class CrossAttentionFusion(nn.Module):
    """Fuses waveform and spectral features using Multi-Head Cross-Attention."""
    def __init__(self, config_attn: Any, wave_channels: int, spec_channels: int):
        super().__init__()
        self.dim = config_attn.dim
        self.num_heads = config_attn.num_heads
        
        # Linear projections to align channel dimensions (implemented as Conv1d for temporal sequences)
        self.proj_wave = nn.Conv1d(wave_channels, self.dim, kernel_size=1)
        self.proj_spec = nn.Conv1d(spec_channels, self.dim, kernel_size=1)
        
        # Multi-Head Attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.dim,
            num_heads=self.num_heads,
            batch_first=True
        )
        
        self.norm = nn.LayerNorm(self.dim)
        
        # Project fused attention back to output channels (we set output channels to embed_dim)
        self.out_channels = self.dim

    def forward(self, feat_wave: torch.Tensor, feat_spec: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat_wave: Waveform feature tensor of shape (batch, wave_channels, time_steps).
            feat_spec: Spectral feature tensor of shape (batch, spec_channels, time_steps_spec).
            
        Returns:
            Fused latent tensor of shape (batch, out_channels, time_steps).
        """
        # 1. Align time steps of spectral features to match waveform features if they differ
        if feat_spec.shape[-1] != feat_wave.shape[-1]:
            feat_spec = nn.functional.interpolate(
                feat_spec, size=feat_wave.shape[-1], mode='linear', align_corners=False
            )
            
        # 2. Project to shared dimension: (batch, dim, time_steps)
        q = self.proj_wave(feat_wave)
        kv = self.proj_spec(feat_spec)
        
        # 3. Transpose to sequence format: (batch, time_steps, dim)
        q_t = q.transpose(1, 2)
        kv_t = kv.transpose(1, 2)
        
        # 4. Multi-Head Cross-Attention
        # Query = Waveform, Key/Value = Spectral
        attn_out, _ = self.cross_attn(query=q_t, key=kv_t, value=kv_t)
        
        # 5. Residual + Normalization
        fused = self.norm(q_t + attn_out)
        
        # 6. Transpose back: (batch, dim, time_steps)
        return fused.transpose(1, 2)
