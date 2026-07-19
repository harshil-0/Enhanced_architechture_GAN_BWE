import os
import sys
import yaml
import mlflow
import torch
import torchaudio
import librosa
import numpy as np
import scipy
import gradio as gr
from typing import Any, Dict, Optional

class MLflowTracker:
    """Centralized MLflow tracking manager for HybridGAN-BWE."""

    def __init__(
        self,
        experiment_name: str = "HybridGAN-BWE",
        tracking_uri: str = "mlruns",
        run_name: Optional[str] = None
    ):
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.run_name = run_name
        self.active_run = None

        # Configure tracking URI
        mlflow.set_tracking_uri(self.tracking_uri)

    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None):
        """Start an MLflow tracking run."""
        name = run_name if run_name else self.run_name
        mlflow.set_experiment(self.experiment_name)
        self.active_run = mlflow.start_run(run_name=name, tags=tags)
        print(f"[MLOps] Started MLflow Run: '{self.active_run.info.run_name}' (ID: {self.active_run.info.run_id})")
        return self.active_run

    def log_params_from_config(self, config: Any, extra_params: Optional[Dict[str, Any]] = None):
        """Log all config hyperparameters, system versions, optimizer, and scheduler settings."""
        if not mlflow.active_run():
            self.start_run()

        params = {}

        # Audio parameters
        params["audio_input_sr"] = config.audio.input_sr
        params["audio_target_sr"] = config.audio.target_sr
        params["audio_segment_length"] = config.audio.segment_length
        params["audio_degradation_type"] = config.audio.degradation_type

        # Dataset parameters
        params["dataset_batch_size"] = config.dataset.batch_size
        params["dataset_crop_type"] = config.dataset.crop_type
        params["dataset_num_workers"] = config.dataset.num_workers

        # Training parameters
        params["train_epochs"] = config.train.epochs
        params["train_learning_rate"] = config.train.learning_rate
        params["train_adam_beta1"] = config.train.adam_beta1
        params["train_adam_beta2"] = config.train.adam_beta2
        params["train_lr_decay"] = config.train.lr_decay
        params["train_gradient_accumulation_steps"] = config.train.gradient_accumulation_steps
        params["train_mixed_precision"] = config.train.mixed_precision
        params["train_grad_clip_val"] = config.train.grad_clip_val
        params["train_early_stopping_patience"] = config.train.early_stopping_patience
        params["optimizer"] = "AdamW"
        params["scheduler"] = "ExponentialLR"

        # Model Architecture
        params["generator_use_waveform"] = config.generator.use_waveform_branch
        params["generator_use_spectral"] = config.generator.use_spectral_branch
        params["generator_use_cross_attention"] = config.generator.use_cross_attention
        params["spectral_encoder_channels"] = config.generator.spectral_encoder.channels
        params["waveform_encoder_channels"] = config.generator.waveform_encoder.channels
        params["attention_dim"] = config.generator.attention.dim
        params["discriminator_spectral_channels"] = config.discriminator.spectral.channels
        params["discriminator_mpd_channels"] = config.discriminator.mpd.channels
        params["discriminator_msd_channels"] = config.discriminator.msd.channels

        # Loss weights
        params["loss_weight_adv_g"] = config.losses.adv_g_weight
        params["loss_weight_fm"] = config.losses.fm_weight
        params["loss_weight_waveform_l1"] = config.losses.waveform_l1_weight
        params["loss_weight_mr_stft"] = config.losses.mr_stft_weight
        params["loss_weight_phase_consistency"] = config.losses.phase_consistency_weight
        params["loss_weight_mel_spectrogram"] = config.losses.mel_spectrogram_weight

        # System & Library Versions
        params["sys_python_version"] = sys.version.split()[0]
        params["lib_torch_version"] = torch.__version__
        params["lib_torchaudio_version"] = torchaudio.__version__
        params["lib_librosa_version"] = librosa.__version__
        params["lib_scipy_version"] = scipy.__version__
        params["lib_numpy_version"] = np.__version__
        params["lib_gradio_version"] = gr.__version__
        params["lib_mlflow_version"] = mlflow.__version__

        if extra_params:
            params.update(extra_params)

        mlflow.log_params(params)
        print(f"[MLOps] Logged {len(params)} hyperparameters and system environment parameters to MLflow.")

    def log_epoch_metrics(self, epoch: int, train_metrics: Dict[str, float], val_metrics: Dict[str, float], lr_g: float):
        """Log per-epoch training and validation loss metrics."""
        if not mlflow.active_run():
            self.start_run()

        metrics = {}
        for k, v in train_metrics.items():
            metrics[f"train_{k}"] = float(v)
        for k, v in val_metrics.items():
            metrics[f"{k}"] = float(v)
            
        metrics["lr_g"] = float(lr_g)

        mlflow.log_metrics(metrics, step=epoch)

    def log_test_metrics(self, test_metrics: Dict[str, float]):
        """Log final evaluation quality metrics (PESQ, STOI, SI-SDR, LSD, RTF, Latency)."""
        if not mlflow.active_run():
            self.start_run()

        formatted_metrics = {}
        for k, v in test_metrics.items():
            prefix = "" if k.startswith("test_") else "test_"
            formatted_metrics[f"{prefix}{k}"] = float(v)

        mlflow.log_metrics(formatted_metrics)
        print(f"[MLOps] Logged test evaluation metrics to MLflow: {formatted_metrics}")

    def log_artifact(self, artifact_path: str, artifact_dir: Optional[str] = None):
        """Log a file or directory as an MLflow artifact."""
        if not mlflow.active_run():
            self.start_run()

        if os.path.exists(artifact_path):
            mlflow.log_artifact(artifact_path, artifact_path=artifact_dir)
            print(f"[MLOps] Logged artifact: '{artifact_path}'")
        else:
            print(f"[MLOps Warning] Artifact file not found at: '{artifact_path}'")

    def end_run(self):
        """End current active MLflow run."""
        if mlflow.active_run():
            run_id = mlflow.active_run().info.run_id
            mlflow.end_run()
            print(f"[MLOps] Ended MLflow Run (ID: {run_id}).")
