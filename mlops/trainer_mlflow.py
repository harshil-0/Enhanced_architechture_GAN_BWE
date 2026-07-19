import os
import torch
from typing import Any
from torch.utils.data import DataLoader
from trainer.trainer import Trainer
from mlops.tracker import MLflowTracker

class TrainerMLflow(Trainer):
    """MLflow-wrapped extension of the core Trainer class.
    Logs per-epoch metrics and model checkpoints automatically to MLflow.
    """
    def __init__(
        self,
        generator: torch.nn.Module,
        discriminator: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Any,
        device: torch.device,
        tracker: MLflowTracker,
        config_path: str = "configs/config.yaml"
    ):
        super().__init__(generator, discriminator, train_loader, val_loader, config, device)
        self.tracker = tracker
        self.config_path = config_path

        # Log parameters and config artifact at startup
        self.tracker.log_params_from_config(self.config)
        if os.path.exists(self.config_path):
            self.tracker.log_artifact(self.config_path, artifact_dir="config")

    def train(self):
        """Main training loop overridden with MLflow logging."""
        for epoch in range(self.start_epoch, self.config.train.epochs + 1):
            train_metrics = self.train_epoch(epoch)
            val_loss, val_metrics = self.validate(epoch)

            current_lr = self.optimizer_g.param_groups[0]['lr']

            # Step schedulers
            self.scheduler_g.step()
            self.scheduler_d.step()

            # TensorBoard logging
            self.log_tensorboard(epoch, train_metrics, val_metrics)

            # CSV logging
            self.log_csv(epoch, train_metrics, val_metrics)

            # MLflow logging
            self.tracker.log_epoch_metrics(epoch, train_metrics, val_metrics, current_lr)

            # Checkpoint saving & MLflow artifact upload
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.early_stop_counter = 0
                best_ckpt_path = self.save_checkpoint(epoch, val_loss, filename="best_model.pth")
                print(f"--> Epoch {epoch}: Saved new best model checkpoint. Val Loss: {val_loss:.4f}")
                
                # Upload best checkpoint as MLflow artifact
                best_full_path = os.path.join(self.checkpoint_dir, "best_model.pth")
                self.tracker.log_artifact(best_full_path, artifact_dir="checkpoints")
            else:
                self.early_stop_counter += 1

            if epoch % self.config.train.save_interval_epochs == 0:
                epoch_ckpt_path = self.save_checkpoint(epoch, val_loss, filename=f"checkpoint_epoch_{epoch}.pth")
                epoch_full_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
                self.tracker.log_artifact(epoch_full_path, artifact_dir="checkpoints")

            if self.early_stop_counter >= self.config.train.early_stopping_patience:
                print(f"[MLOps] Early stopping triggered after {self.early_stop_counter} epochs without improvement.")
                break

        self.writer.close()
