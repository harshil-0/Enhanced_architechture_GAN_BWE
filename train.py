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
from trainer.trainer import Trainer

def generate_dummy_dataset(dummy_dir: str, num_files: int = 10, sample_rate: int = 16000, duration: float = 2.0):
    """Generate mock WAV files for dry-run training."""
    os.makedirs(dummy_dir, exist_ok=True)
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    print(f"Creating synthetic dummy dataset in '{dummy_dir}' ({num_files} files)...")
    for idx in range(num_files):
        freq = np.random.randint(150, 1000)
        # Combine sine waves
        audio = 0.4 * np.sin(2 * np.pi * freq * t) + 0.1 * np.sin(2 * np.pi * 2 * freq * t)
        filepath = os.path.join(dummy_dir, f"dummy_{idx}.wav")
        sf.write(filepath, audio, sample_rate)
    print("Dummy dataset created.")


def main():
    parser = argparse.ArgumentParser(description="Train HybridGAN-BWE Speech Bandwidth Extension System")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config YAML file")
    parser.add_argument("--dataset_dir", type=str, default=None, help="Path to dataset directory to scan & build manifests")
    parser.add_argument("--device", type=str, default=None, help="Device to train on (cuda/cpu)")
    args = parser.parse_args()

    # 1. Load config
    config = load_config(args.config)

    # 2. Setup manifests
    train_manifest = config.dataset.train_manifest
    val_manifest = config.dataset.val_manifest
    test_manifest = config.dataset.test_manifest

    # If dataset_dir is provided or manifests are missing, scan and build them
    manifests_exist = (
        os.path.exists(train_manifest) and 
        os.path.exists(val_manifest) and 
        os.path.exists(test_manifest)
    )

    if args.dataset_dir is not None:
        print(f"Scanning provided dataset directory: {args.dataset_dir}")
        create_manifests(
            dataset_dir=args.dataset_dir,
            output_dir=os.path.dirname(train_manifest),
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1
        )
    elif not manifests_exist:
        print("Dataset manifests not found. Checking for synthetic dry-run setup...")
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

    # 3. Setup device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for training: {device}")

    # 4. Load Dataloaders
    print("Loading datasets...")
    train_loader = get_dataloader(train_manifest, config, is_validation=False)
    val_loader = get_dataloader(val_manifest, config, is_validation=True)
    print(f"Dataloaders initialized. Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # 5. Initialize Models
    print("Initializing Generator...")
    generator = Generator(config)
    
    print("Initializing Discriminators...")
    discriminator = DiscriminatorGroup(config)

    # Print model parameters
    g_params = sum(p.numel() for p in generator.parameters() if p.requires_grad)
    d_params = sum(p.numel() for p in discriminator.parameters() if p.requires_grad)
    print(f"Generator parameters: {g_params:,}")
    print(f"Discriminator parameters: {d_params:,}")

    # 6. Initialize Trainer & Train
    print("Starting training pipeline...")
    trainer = Trainer(
        generator=generator,
        discriminator=discriminator,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )
    
    trainer.train()

if __name__ == "__main__":
    main()
