import time
import torch
import torchaudio
import numpy as np
from typing import Any, Dict, Tuple

# Dynamic imports for compiled/external metric libraries to avoid load-time crashes
PESQ_AVAILABLE = False
STOI_AVAILABLE = False

try:
    from pesq import pesq
    PESQ_AVAILABLE = True
except ImportError:
    pass

try:
    from pystoi import stoi
    STOI_AVAILABLE = True
except ImportError:
    pass


def calculate_sisdr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Calculate Scale-Invariant Signal-to-Distortion Ratio (SI-SDR).
    
    Args:
        pred: Predicted audio waveform tensor.
        target: Target ground-truth waveform tensor.
        
    Returns:
        SI-SDR value in dB.
    """
    pred = pred.clone().detach().view(-1)
    target = target.clone().detach().view(-1)
    
    # Zero-mean
    pred = pred - torch.mean(pred)
    target = target - torch.mean(target)
    
    alpha = torch.sum(pred * target) / (torch.sum(target ** 2) + 1e-8)
    target_scaled = alpha * target
    
    noise = pred - target_scaled
    val = 10 * torch.log10(torch.sum(target_scaled ** 2) / (torch.sum(noise ** 2) + 1e-8) + 1e-8)
    return val.item()


def calculate_lsd(
    pred: torch.Tensor,
    target: torch.Tensor,
    fft_size: int = 2048,
    hop_length: int = 512,
    win_length: int = 2048
) -> float:
    """Calculate Log-Spectral Distance (LSD).
    
    Args:
        pred: Predicted audio waveform tensor.
        target: Target ground-truth waveform tensor.
        
    Returns:
        LSD value in dB.
    """
    pred_sq = pred.clone().detach().squeeze()
    target_sq = target.clone().detach().squeeze()
    
    # Ensure 1D
    if pred_sq.dim() > 1:
        pred_sq = pred_sq[0]
    if target_sq.dim() > 1:
        target_sq = target_sq[0]
        
    window = torch.hann_window(win_length, device=pred.device)
    
    stft_p = torch.stft(
        pred_sq, n_fft=fft_size, hop_length=hop_length, win_length=win_length,
        window=window, center=True, pad_mode='reflect', normalized=False,
        onesided=True, return_complex=True
    )
    stft_t = torch.stft(
        target_sq, n_fft=fft_size, hop_length=hop_length, win_length=win_length,
        window=window, center=True, pad_mode='reflect', normalized=False,
        onesided=True, return_complex=True
    )
    
    mag_p = torch.abs(stft_p)
    mag_t = torch.abs(stft_t)
    
    log_p = 20 * torch.log10(mag_p + 1e-7)
    log_t = 20 * torch.log10(mag_t + 1e-7)
    
    # Root mean square over frequency bins
    dist_per_frame = torch.sqrt(torch.mean((log_t - log_p) ** 2, dim=0) + 1e-8)
    lsd_val = torch.mean(dist_per_frame)
    return lsd_val.item()


def calculate_pesq(
    pred: torch.Tensor,
    target: torch.Tensor,
    sr: int
) -> float:
    """Calculate Perceptual Evaluation of Speech Quality (PESQ).
    
    Forces sample rate to 16000 Hz if different.
    
    Args:
        pred: Predicted audio waveform tensor.
        target: Target ground-truth waveform tensor.
        sr: Current sample rate of inputs.
        
    Returns:
        PESQ score (1.0 to 4.5), or float('nan') if not available.
    """
    if not PESQ_AVAILABLE:
        return float('nan')
        
    pred_np = pred.clone().detach().squeeze().cpu().numpy()
    target_np = target.clone().detach().squeeze().cpu().numpy()
    
    # Ensure 1D
    if pred_np.ndim > 1:
        pred_np = pred_np[0]
    if target_np.ndim > 1:
        target_np = target_np[0]
        
    # PESQ only supports 8000 Hz (narrowband) and 16000 Hz (wideband)
    if sr != 16000 and sr != 8000:
        # Resample to 16000 for Wideband PESQ
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        pred_16 = resampler(pred.cpu()).squeeze().numpy()
        target_16 = resampler(target.cpu()).squeeze().numpy()
        fs = 16000
    else:
        pred_16 = pred_np
        target_16 = target_np
        fs = sr
        
    mode = 'wb' if fs == 16000 else 'nb'
    
    try:
        score = pesq(fs, target_16, pred_16, mode)
        return float(score)
    except Exception as e:
        # PESQ can occasionally crash on pure silence or very distorted audio
        return float('nan')


def calculate_stoi(
    pred: torch.Tensor,
    target: torch.Tensor,
    sr: int
) -> float:
    """Calculate Short-Time Objective Intelligibility (STOI).
    
    Args:
        pred: Predicted audio waveform tensor.
        target: Target ground-truth waveform tensor.
        sr: Current sample rate.
        
    Returns:
        STOI score (0.0 to 1.0), or float('nan') if not available.
    """
    if not STOI_AVAILABLE:
        return float('nan')
        
    pred_np = pred.clone().detach().squeeze().cpu().numpy()
    target_np = target.clone().detach().squeeze().cpu().numpy()
    
    # Ensure 1D
    if pred_np.ndim > 1:
        pred_np = pred_np[0]
    if target_np.ndim > 1:
        target_np = target_np[0]
        
    try:
        score = stoi(target_np, pred_np, sr, extended=False)
        return float(score)
    except Exception as e:
        return float('nan')


def count_parameters(model: torch.nn.Module) -> int:
    """Return the number of trainable parameters in a PyTorch module."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def benchmark_inference(
    generator: torch.nn.Module,
    x: torch.Tensor,
    sr: int,
    num_runs: int = 20
) -> Tuple[float, float, float]:
    """Measure the latency and Real-Time Factor (RTF) of a generator model.
    
    Args:
        generator: The Generator model.
        x: Input waveform tensor of shape (1, 1, samples).
        sr: Sample rate of the input.
        num_runs: Number of benchmark loops to run.
        
    Returns:
        Tuple of (average latency in ms, RTF, GFLOPs estimation [simulated/placeholder]).
    """
    generator.eval()
    duration = x.shape[-1] / sr
    
    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = generator(x)
            
    # Benchmark
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.perf_counter()
    
    with torch.no_grad():
        for _ in range(num_runs):
            _ = generator(x)
            
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end_time = time.perf_counter()
    
    avg_latency = (end_time - start_time) / num_runs # seconds
    rtf = avg_latency / duration
    
    # Simple estimate of GFLOPs based on parameter count and receptive field operations
    # (Provided as a research estimation since exact dynamic flops count on variable input requires fvcore)
    gparams = count_parameters(generator)
    # Estimated GFLOPs: params * 2 * operations_per_sample * 1e-9
    # We report latency and RTF as the primary hardware constraints metrics
    estimated_gflops = (gparams * 2 * (x.shape[-1] / 1000.0)) * 1e-9
    
    return avg_latency * 1000.0, rtf, estimated_gflops
