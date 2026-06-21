import os
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from typing import Any, Dict, List, Tuple
from losses.adversarial import generator_loss, discriminator_loss, feature_matching_loss
from losses.spectral import MultiResolutionSTFTLoss, PhaseConsistencyLoss, MelSpectrogramLoss

class Trainer:
    """Trainer class for HybridGAN-BWE."""
    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Any,
        device: torch.device
    ):
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        self.accum_steps = config.train.gradient_accumulation_steps
        self.mixed_precision = config.train.mixed_precision
        self.grad_clip_val = config.train.grad_clip_val
        self.checkpoint_dir = config.logging.checkpoint_dir
        self.csv_log_file = config.logging.csv_log_file
        
        # Optimizers
        self.optimizer_g = torch.optim.AdamW(
            self.generator.parameters(),
            lr=config.train.learning_rate,
            betas=(config.train.adam_beta1, config.train.adam_beta2)
        )
        self.optimizer_d = torch.optim.AdamW(
            self.discriminator.parameters(),
            lr=config.train.learning_rate,
            betas=(config.train.adam_beta1, config.train.adam_beta2)
        )
        
        # Schedulers
        self.scheduler_g = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_g, gamma=config.train.lr_decay)
        self.scheduler_d = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_d, gamma=config.train.lr_decay)
        
        # Mixed Precision Scalers
        self.scaler_g = torch.amp.GradScaler('cuda', enabled=self.mixed_precision)
        self.scaler_d = torch.amp.GradScaler('cuda', enabled=self.mixed_precision)
        
        # Loss Functions
        self.stft_loss_fn = MultiResolutionSTFTLoss(
            fft_sizes=config.discriminator.spectral.fft_sizes,
            hop_lengths=config.discriminator.spectral.hop_lengths,
            win_lengths=config.discriminator.spectral.win_lengths
        )
        self.phase_loss_fn = PhaseConsistencyLoss()
        self.mel_loss_fn = MelSpectrogramLoss(sample_rate=config.audio.target_sr)
        self.l1_loss_fn = nn.L1Loss()
        
        # Loggers
        self.writer = SummaryWriter(log_dir=config.logging.tensorboard_dir)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.csv_log_file), exist_ok=True)
        
        self.start_epoch = 1
        self.best_val_loss = float('inf')
        self.early_stop_counter = 0
        
        if config.train.resume_checkpoint:
            self.load_checkpoint(config.train.resume_checkpoint)

    def train(self):
        """Main training loop."""
        for epoch in range(self.start_epoch, self.config.train.epochs + 1):
            train_metrics = self.train_epoch(epoch)
            val_loss, val_metrics = self.validate(epoch)
            
            # Decay learning rates
            self.scheduler_g.step()
            self.scheduler_d.step()
            
            # TensorBoard logging
            self.log_tensorboard(epoch, train_metrics, val_metrics)
            
            # CSV logging
            self.log_csv(epoch, train_metrics, val_metrics)
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.early_stop_counter = 0
                self.save_checkpoint(epoch, val_loss, filename="best_model.pth")
                print(f"--> Epoch {epoch}: Saved new best model checkpoint. Val Loss: {val_loss:.4f}")
            else:
                self.early_stop_counter += 1
                
            if epoch % self.config.train.save_interval_epochs == 0:
                self.save_checkpoint(epoch, val_loss, filename=f"checkpoint_epoch_{epoch}.pth")
                
            # Early stopping
            if self.early_stop_counter >= self.config.train.early_stopping_patience:
                print(f"Early stopping triggered after {self.early_stop_counter} epochs without improvement.")
                break
                
        self.writer.close()

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.generator.train()
        self.discriminator.train()
        
        epoch_metrics = {
            "loss_g": 0.0, "loss_g_adv": 0.0, "loss_g_fm": 0.0, "loss_g_l1": 0.0,
            "loss_g_stft": 0.0, "loss_g_phase": 0.0, "loss_g_mel": 0.0, "loss_d": 0.0
        }
        
        self.optimizer_g.zero_grad()
        self.optimizer_d.zero_grad()
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} Training")
        for step, (x_nb, y_real) in enumerate(pbar):
            x_nb = x_nb.to(self.device, non_blocking=True)
            y_real = y_real.to(self.device, non_blocking=True)
            
            # ---------------------
            # 1. Train Discriminator
            # ---------------------
            with torch.amp.autocast('cuda', enabled=self.mixed_precision):
                y_fake = self.generator(x_nb)
                
                # Discriminator output on Real and Fake
                d_real_scores, _ = self.discriminator(y_real)
                d_fake_scores, _ = self.discriminator(y_fake.detach())
                
                loss_d = discriminator_loss(d_real_scores, d_fake_scores)
                # Scale by accumulation steps
                loss_d_scaled = loss_d / self.accum_steps
                
            self.scaler_d.scale(loss_d_scaled).backward()
            
            # ---------------------
            # 2. Train Generator
            # ---------------------
            with torch.amp.autocast('cuda', enabled=self.mixed_precision):
                d_real_scores, d_real_fmaps = self.discriminator(y_real)
                d_fake_scores, d_fake_fmaps = self.discriminator(y_fake)
                
                # Adversarial and Feature Matching losses
                loss_g_adv = generator_loss(d_fake_scores)
                loss_g_fm = feature_matching_loss(d_real_fmaps, d_fake_fmaps)
                
                # Reconstruction / Spectral losses
                loss_g_l1 = self.l1_loss_fn(y_fake, y_real)
                
                sc_loss, log_mag_loss = self.stft_loss_fn(y_fake, y_real)
                loss_g_stft = sc_loss + log_mag_loss
                
                loss_g_phase = self.phase_loss_fn(y_fake, y_real)
                loss_g_mel = self.mel_loss_fn(y_fake, y_real)
                
                # Final generator loss
                w = self.config.losses
                loss_g = (
                    w.adv_g_weight * loss_g_adv +
                    w.fm_weight * loss_g_fm +
                    w.waveform_l1_weight * loss_g_l1 +
                    w.mr_stft_weight * loss_g_stft +
                    w.phase_consistency_weight * loss_g_phase +
                    w.mel_spectrogram_weight * loss_g_mel
                )
                
                loss_g_scaled = loss_g / self.accum_steps
                
            self.scaler_g.scale(loss_g_scaled).backward()
            
            # Step optimizers after accumulation steps
            if (step + 1) % self.accum_steps == 0 or (step + 1) == len(self.train_loader):
                # Update Discriminator
                self.scaler_d.unscale_(self.optimizer_d)
                if self.grad_clip_val > 0:
                    nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.grad_clip_val)
                self.scaler_d.step(self.optimizer_d)
                self.scaler_d.update()
                self.optimizer_d.zero_grad()
                
                # Update Generator
                self.scaler_g.unscale_(self.optimizer_g)
                if self.grad_clip_val > 0:
                    nn.utils.clip_grad_norm_(self.generator.parameters(), self.grad_clip_val)
                self.scaler_g.step(self.optimizer_g)
                self.scaler_g.update()
                self.optimizer_g.zero_grad()
                
            # Log metrics
            epoch_metrics["loss_d"] += loss_d.item()
            epoch_metrics["loss_g"] += loss_g.item()
            epoch_metrics["loss_g_adv"] += loss_g_adv.item()
            epoch_metrics["loss_g_fm"] += loss_g_fm.item()
            epoch_metrics["loss_g_l1"] += loss_g_l1.item()
            epoch_metrics["loss_g_stft"] += loss_g_stft.item()
            epoch_metrics["loss_g_phase"] += loss_g_phase.item()
            epoch_metrics["loss_g_mel"] += loss_g_mel.item()
            
            pbar.set_postfix({
                "G_Loss": f"{loss_g.item():.3f}",
                "D_Loss": f"{loss_d.item():.3f}"
            })
            
        # Average epoch metrics
        num_batches = len(self.train_loader)
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
            
        return epoch_metrics

    @torch.no_grad()
    def validate(self, epoch: int) -> Tuple[float, Dict[str, float]]:
        self.generator.eval()
        self.discriminator.eval()
        
        val_metrics = {
            "val_loss": 0.0, "val_l1": 0.0, "val_stft": 0.0,
            "val_phase": 0.0, "val_mel": 0.0
        }
        
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} Validation")
        for x_nb, y_real in pbar:
            x_nb = x_nb.to(self.device, non_blocking=True)
            y_real = y_real.to(self.device, non_blocking=True)
            
            y_fake = self.generator(x_nb)
            
            loss_l1 = self.l1_loss_fn(y_fake, y_real)
            sc_loss, log_mag_loss = self.stft_loss_fn(y_fake, y_real)
            loss_stft = sc_loss + log_mag_loss
            loss_phase = self.phase_loss_fn(y_fake, y_real)
            loss_mel = self.mel_loss_fn(y_fake, y_real)
            
            # Validation target metric is a combination of L1 and Spectral losses
            val_loss = loss_l1 + loss_stft + loss_phase + loss_mel
            
            val_metrics["val_loss"] += val_loss.item()
            val_metrics["val_l1"] += loss_l1.item()
            val_metrics["val_stft"] += loss_stft.item()
            val_metrics["val_phase"] += loss_phase.item()
            val_metrics["val_mel"] += loss_mel.item()
            
        num_batches = len(self.val_loader)
        for key in val_metrics:
            val_metrics[key] /= num_batches
            
        # Log a few audio comparisons to Tensorboard
        if len(self.val_loader) > 0:
            # Take last validation sample
            self.writer.add_audio("val/audio_input_nb", x_nb[0].cpu(), epoch, sample_rate=self.config.audio.target_sr)
            self.writer.add_audio("val/audio_pred_wb", y_fake[0].cpu(), epoch, sample_rate=self.config.audio.target_sr)
            self.writer.add_audio("val/audio_true_wb", y_real[0].cpu(), epoch, sample_rate=self.config.audio.target_sr)
            
        return val_metrics["val_loss"], val_metrics

    def log_tensorboard(self, epoch: int, train_metrics: Dict[str, float], val_metrics: Dict[str, float]):
        for key, val in train_metrics.items():
            self.writer.add_scalar(f"train/{key}", val, epoch)
        for key, val in val_metrics.items():
            self.writer.add_scalar(f"val/{key}", val, epoch)
            
        # Log learning rates
        self.writer.add_scalar("lr/generator", self.optimizer_g.param_groups[0]["lr"], epoch)
        self.writer.add_scalar("lr/discriminator", self.optimizer_d.param_groups[0]["lr"], epoch)

    def log_csv(self, epoch: int, train_metrics: Dict[str, float], val_metrics: Dict[str, float]):
        file_exists = os.path.exists(self.csv_log_file)
        
        row = {"epoch": epoch, "lr_g": self.optimizer_g.param_groups[0]["lr"]}
        row.update(train_metrics)
        row.update(val_metrics)
        
        with open(self.csv_log_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    def save_checkpoint(self, epoch: int, val_loss: float, filename: str):
        filepath = os.path.join(self.checkpoint_dir, filename)
        state = {
            "epoch": epoch,
            "generator_state": self.generator.state_dict(),
            "discriminator_state": self.discriminator.state_dict(),
            "optimizer_g_state": self.optimizer_g.state_dict(),
            "optimizer_d_state": self.optimizer_d.state_dict(),
            "scheduler_g_state": self.scheduler_g.state_dict(),
            "scheduler_d_state": self.scheduler_d.state_dict(),
            "scaler_g_state": self.scaler_g.state_dict(),
            "scaler_d_state": self.scaler_d.state_dict(),
            "val_loss": val_loss,
            "best_val_loss": self.best_val_loss,
            "config": self.config.to_dict()
        }
        torch.save(state, filepath)

    def load_checkpoint(self, filepath: str):
        print(f"Resuming training from checkpoint: {filepath}")
        state = torch.load(filepath, map_location=self.device, weights_only=False)
        self.start_epoch = state["epoch"] + 1
        self.generator.load_state_dict(state["generator_state"])
        self.discriminator.load_state_dict(state["discriminator_state"])
        self.optimizer_g.load_state_dict(state["optimizer_g_state"])
        self.optimizer_d.load_state_dict(state["optimizer_d_state"])
        self.scheduler_g.load_state_dict(state["scheduler_g_state"])
        self.scheduler_d.load_state_dict(state["scheduler_d_state"])
        self.scaler_g.load_state_dict(state["scaler_g_state"])
        self.scaler_d.load_state_dict(state["scaler_d_state"])
        self.best_val_loss = state["best_val_loss"]
        print(f"Checkpoint loaded successfully. Resuming from Epoch {self.start_epoch}")
