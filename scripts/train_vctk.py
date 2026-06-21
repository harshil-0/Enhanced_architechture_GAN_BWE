import os
import sys
import subprocess
from datasets.manifest import create_manifests

def main():
    print("==================================================")
    print("HybridGAN-BWE: VCTK Training Pipeline")
    print("==================================================")
    
    # 1. Generate Manifests
    print("\nStep 1: Generating VCTK split manifests in parallel...")
    create_manifests(
        dataset_dir="wav48_silence_trimmed",
        output_dir="manifests",
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1
    )
    
    # 2. Launch Training
    print("\nStep 2: Manifests saved. Starting train.py training loop on VCTK...")
    cmd = [sys.executable, "train.py"]
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nError running training loop: {e}")

if __name__ == "__main__":
    main()
