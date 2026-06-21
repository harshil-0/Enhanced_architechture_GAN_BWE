import os
import json
import random
import soundfile as sf
from tqdm import tqdm
from typing import List, Dict, Tuple

def scan_audio_files(directory: str) -> List[str]:
    """Recursively search for WAV/FLAC files in a directory.
    
    Args:
        directory: Root directory path to scan.
        
    Returns:
        List of absolute file paths.
    """
    audio_files = []
    supported_extensions = {".wav", ".flac", ".mp3"}
    for root, _, files in os.walk(directory):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in supported_extensions:
                audio_files.append(os.path.abspath(os.path.join(root, file)))
    return audio_files


import concurrent.futures

def get_file_metadata(filepath: str) -> dict:
    """Read metadata for a single audio file."""
    try:
        info = sf.info(filepath)
        return {
            "filepath": filepath,
            "duration": info.duration,
            "samples": info.frames,
            "samplerate": info.samplerate
        }
    except Exception as e:
        return None


def create_manifests(
    dataset_dir: str,
    output_dir: str = "manifests",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[str, str, str]:
    """Scan audio directory, create train/val/test splits, and save manifests.
    
    Args:
        dataset_dir: Directory containing the audio files.
        output_dir: Output directory to write JSON manifests.
        train_ratio: Ratio for training split.
        val_ratio: Ratio for validation split.
        test_ratio: Ratio for test split.
        seed: Random seed for deterministic splits.
        
    Returns:
        Tuple of paths (train_manifest, val_manifest, test_manifest)
    """
    # Fix seed for reproducibility
    random.seed(seed)
    
    print(f"Scanning directory: {dataset_dir} for audio files...")
    all_files = scan_audio_files(dataset_dir)
    print(f"Found {len(all_files)} audio files.")
    
    if not all_files:
        raise ValueError(f"No audio files found in {dataset_dir}")
        
    # Get metadata for each file in parallel
    print("Extracting metadata in parallel...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        results = list(tqdm(executor.map(get_file_metadata, all_files), total=len(all_files)))
        manifest_entries = [r for r in results if r is not None]
            
    # Shuffle and split
    random.shuffle(manifest_entries)
    n = len(manifest_entries)
    
    # Calculate indexes
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    train_split = manifest_entries[:n_train]
    val_split = manifest_entries[n_train:n_train + n_val]
    test_split = manifest_entries[n_train + n_val:]
    
    # Ensure directories exist
    os.makedirs(output_dir, exist_ok=True)
    
    train_path = os.path.join(output_dir, "train.json")
    val_path = os.path.join(output_dir, "val.json")
    test_path = os.path.join(output_dir, "test.json")
    
    with open(train_path, "w") as f:
        json.dump(train_split, f, indent=4)
        
    with open(val_path, "w") as f:
        json.dump(val_split, f, indent=4)
        
    with open(test_path, "w") as f:
        json.dump(test_split, f, indent=4)
        
    print(f"Manifests created successfully in '{output_dir}':")
    print(f"  Train: {len(train_split)} files ({train_path})")
    print(f"  Val:   {len(val_split)} files ({val_path})")
    print(f"  Test:  {len(test_split)} files ({test_path})")
    
    return train_path, val_path, test_path
