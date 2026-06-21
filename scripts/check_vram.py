import os
import torch
import torch.nn as nn
from utils.config import load_config
from models.generator import Generator
from models.discriminator import DiscriminatorGroup
from losses.adversarial import generator_loss, discriminator_loss, feature_matching_loss
from losses.spectral import MultiResolutionSTFTLoss, PhaseConsistencyLoss, MelSpectrogramLoss

def main():
    if not torch.cuda.is_available():
        print("CUDA is not available. VRAM verification requires a GPU.")
        return

    # Load default configuration
    config_path = "configs/config.yaml"
    if not os.path.exists(config_path):
        print(f"Config not found at {config_path}")
        return
        
    config = load_config(config_path)
    
    device = torch.device("cuda")
    print(f"Using device: {torch.cuda.get_device_name(0)}")
    
    # 1. Reset memory statistics
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    initial_memory = torch.cuda.memory_allocated() / (1024 ** 2)
    print(f"Initial allocated memory: {initial_memory:.2f} MB")
    
    # 2. Instantiate hybrid models
    print("\nInstantiating Generator and Discriminators...")
    generator = Generator(config).to(device)
    discriminator = DiscriminatorGroup(config).to(device)
    
    model_load_memory = torch.cuda.memory_allocated() / (1024 ** 2)
    print(f"Memory after loading models: {model_load_memory:.2f} MB")
    
    # 3. Setup optimizers
    optimizer_g = torch.optim.AdamW(generator.parameters(), lr=1e-4)
    optimizer_d = torch.optim.AdamW(discriminator.parameters(), lr=1e-4)
    
    # Setup losses
    stft_loss_fn = MultiResolutionSTFTLoss(
        fft_sizes=config.discriminator.spectral.fft_sizes,
        hop_lengths=config.discriminator.spectral.hop_lengths,
        win_lengths=config.discriminator.spectral.win_lengths
    ).to(device)
    phase_loss_fn = PhaseConsistencyLoss().to(device)
    mel_loss_fn = MelSpectrogramLoss(sample_rate=config.audio.target_sr).to(device)
    l1_loss_fn = nn.L1Loss().to(device)
    
    scaler_g = torch.amp.GradScaler('cuda', enabled=config.train.mixed_precision)
    scaler_d = torch.amp.GradScaler('cuda', enabled=config.train.mixed_precision)
    
    # 4. Mock forward and backward passes
    # Batch size 4, 1 channel, segment length 16384
    batch_size = config.dataset.batch_size
    seg_len = config.audio.segment_length
    print(f"\nRunning mock training step (Batch Size: {batch_size}, Segment Length: {seg_len})...")
    
    x_nb = torch.randn(batch_size, 1, seg_len, device=device)
    y_real = torch.randn(batch_size, 1, seg_len, device=device)
    
    # --- Discriminator Step ---
    optimizer_d.zero_grad()
    with torch.amp.autocast('cuda', enabled=config.train.mixed_precision):
        y_fake = generator(x_nb)
        d_real_scores, _ = discriminator(y_real)
        d_fake_scores, _ = discriminator(y_fake.detach())
        loss_d = discriminator_loss(d_real_scores, d_fake_scores)
        
    scaler_d.scale(loss_d).backward()
    scaler_d.step(optimizer_d)
    scaler_d.update()
    
    # --- Generator Step ---
    optimizer_g.zero_grad()
    with torch.amp.autocast('cuda', enabled=config.train.mixed_precision):
        d_real_scores, d_real_fmaps = discriminator(y_real)
        d_fake_scores, d_fake_fmaps = discriminator(y_fake)
        
        loss_g_adv = generator_loss(d_fake_scores)
        loss_g_fm = feature_matching_loss(d_real_fmaps, d_fake_fmaps)
        
        loss_g_l1 = l1_loss_fn(y_fake, y_real)
        sc, lm = stft_loss_fn(y_fake, y_real)
        loss_g_stft = sc + lm
        loss_g_phase = phase_loss_fn(y_fake, y_real)
        loss_g_mel = mel_loss_fn(y_fake, y_real)
        
        w = config.losses
        loss_g = (
            w.adv_g_weight * loss_g_adv +
            w.fm_weight * loss_g_fm +
            w.waveform_l1_weight * loss_g_l1 +
            w.mr_stft_weight * loss_g_stft +
            w.phase_consistency_weight * loss_g_phase +
            w.mel_spectrogram_weight * loss_g_mel
        )
        
    scaler_g.scale(loss_g).backward()
    scaler_g.step(optimizer_g)
    scaler_g.update()
    
    # 5. Measure peak memory
    peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
    print("\n--- Memory Performance Results ---")
    print(f"Peak VRAM Allocated: {peak_memory:.2f} MB ({peak_memory / 1024.0:.3f} GB)")
    print(f"VRAM Budget limit:   7200.00 MB (7.031 GB)")
    print(f"Target VRAM safety:  {peak_memory < 7200.0} (Peak < Budget)")
    
if __name__ == "__main__":
    main()
