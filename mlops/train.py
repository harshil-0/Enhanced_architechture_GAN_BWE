import os
import argparse
import torch
import soundfile as sf
import numpy as np

from utils.config import load_config
from datasets.manifest import create_manifests
from datasets.dataset import get_dataloader
from models.generator import Generator
from models.discriminator import DiscriminatorGroup

from mlops.seed import set_seed
from mlops.tracker import MLflowTracker
from mlops.trainer_mlflow import TrainerMLflow

def generate_dummy_dataset(dummy_dir: str, num_files: int = 10, sample_rate: int = 16000, duration: float = 2.0):
    """Generate mock WAV files for dry-run testing."""
    os.makedirs(dummy_dir, exist_ok=True)
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    for idx in range(num_files):
        freq = np.random.randint(150, 1000)
        audio = 0.4 * np.sin(2 * np.pi * freq * t) + 0.1 * np.sin(2 * np.pi * 2 * freq * t)
        filepath = os.path.join(dummy_dir, f"dummy_{idx}.wav")
        sf.write(filepath, audio, sample_rate)

def main():
    parser = argparse.ArgumentParser(description="Train HybridGAN-BWE System with MLflow Tracking")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config YAML file")
    parser.add_argument("--dataset_dir", type=str, default=None, help="Path to dataset directory")
    parser.add_argument("--device", type=str, default=None, help="Device to train on (cuda/cpu)")
    parser.add_argument("--experiment_name", type=str, default="HybridGAN-BWE", help="MLflow experiment name")
    parser.add_argument("--run_name", type=str, default=None, help="MLflow run name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    # 1. Set seed for reproducibility
    set_seed(args.seed)

    # 2. Load config
    config = load_config(args.config)

    # 3. Setup manifests
    train_manifest = config.dataset.train_manifest
    val_manifest = config.dataset.val_manifest
    test_manifest = config.dataset.test_manifest

    manifests_exist = (
        os.path.exists(train_manifest) and 
        os.path.exists(val_manifest) and 
        os.path.exists(test_manifest)
    )

    if args.dataset_dir is not None:
        print(f"[MLOps] Scanning dataset directory: {args.dataset_dir}")
        create_manifests(
            dataset_dir=args.dataset_dir,
            output_dir=os.path.dirname(train_manifest),
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1
        )
    elif not manifests_exist:
        print("[MLOps] Dataset manifests not found. Initializing dry-run setup...")
        dummy_dir = "dummy_dataset"
        if not os.path.exists(dummy_dir) or len(os.listdir(dummy_dir)) == 0:
            generate_dummy_dataset(dummy_dir, num_files=10, sample_rate=config.audio.target_sr)
            
        create_manifests(
            dataset_dir=dummy_dir,
            output_dir=os.path.dirname(train_manifest),
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15
        )

    # 4. Setup Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[MLOps] Using device: {device}")

    # 5. Initialize MLflow Tracker
    tracker = MLflowTracker(experiment_name=args.experiment_name)
    tracker.start_run(run_name=args.run_name, tags={"framework": "pytorch", "task": "speech_bwe"})

    try:
        # 6. Load Dataloaders
        train_loader = get_dataloader(train_manifest, config, is_validation=False)
        val_loader = get_dataloader(val_manifest, config, is_validation=True)

        # 7. Initialize Models
        generator = Generator(config)
        discriminator = DiscriminatorGroup(config)

        g_params = sum(p.numel() for p in generator.parameters() if p.requires_grad)
        d_params = sum(p.numel() for p in discriminator.parameters() if p.requires_grad)
        print(f"[MLOps] Generator params: {g_params:,} | Discriminator params: {d_params:,}")

        # 8. Run Trainer with MLflow Logging
        trainer = TrainerMLflow(
            generator=generator,
            discriminator=discriminator,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device,
            tracker=tracker,
            config_path=args.config
        )
        
        trainer.train()
        print("[MLOps] Training completed successfully.")

    finally:
        tracker.end_run()

if __name__ == "__main__":
    main()
