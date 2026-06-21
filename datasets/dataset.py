import json
import os
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from typing import Any, Dict, List, Tuple
from utils.degradation import DegradationPipeline
from utils.audio import pad_to_multiple

class BWEAudioDataset(Dataset):
    """Dataset for Speech Bandwidth Extension (BWE) with dynamic telephone codec degradation."""
    def __init__(
        self,
        manifest_path: str,
        config: Any,
        is_validation: bool = False,
        is_evaluation: bool = False
    ):
        """
        Args:
            manifest_path: Path to the JSON manifest containing file entries.
            config: Config node.
            is_validation: If True, uses deterministic fixed crop.
            is_evaluation: If True, returns full-length audio without cropping.
        """
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Manifest path does not exist: {manifest_path}")
            
        with open(manifest_path, "r") as f:
            self.entries = json.load(f)
            
        self.config = config
        self.is_validation = is_validation
        self.is_evaluation = is_evaluation
        
        self.target_sr = config.audio.target_sr
        self.segment_length = config.audio.segment_length
        self.crop_type = config.dataset.crop_type
        
        # Initialize degradation pipeline
        self.degrader = DegradationPipeline(config.audio)
        
    def __len__(self) -> int:
        return len(self.entries)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (degraded_nb_waveform, gt_wb_waveform).
        
        Both waveforms will have sample rate `target_sr` and shape (1, samples).
        """
        entry = self.entries[idx]
        filepath = entry["filepath"]
        
        # Load audio (channels, samples)
        wav, sr = torchaudio.load(filepath)
        
        # Convert to mono if multi-channel
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
            
        # Resample GT wideband if it does not match config target_sr
        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sr)
            wav = resampler(wav)
            
        # Apply degradation pipeline to obtain narrowband audio at target_sr
        nb_wav = self.degrader(wav)
        
        # If in evaluation mode, do not crop, just pad to multiple of 256 for STFT compatibility
        if self.is_evaluation:
            # Pad to multiple of 256
            padded_wav = pad_to_multiple(wav, 256)
            padded_nb = pad_to_multiple(nb_wav, 256)
            return padded_nb, padded_wav
            
        # Apply cropping/padding for training & validation
        samples = wav.shape[-1]
        if samples > self.segment_length:
            if self.is_validation:
                # Deterministic center crop
                start = (samples - self.segment_length) // 2
            else:
                # Random crop
                start = torch.randint(0, samples - self.segment_length, (1,)).item()
                
            wav_crop = wav[:, start:start + self.segment_length]
            nb_crop = nb_wav[:, start:start + self.segment_length]
        else:
            # Pad if shorter than segment_length
            diff = self.segment_length - samples
            wav_crop = torch.nn.functional.pad(wav, (0, diff), mode='constant')
            nb_crop = torch.nn.functional.pad(nb_wav, (0, diff), mode='constant')
            
        return nb_crop, wav_crop


def get_dataloader(
    manifest_path: str,
    config: Any,
    is_validation: bool = False,
    is_evaluation: bool = False
) -> DataLoader:
    """Creates a PyTorch DataLoader for BWEAudioDataset.
    
    Args:
        manifest_path: Path to the JSON manifest.
        config: Config node.
        is_validation: Validation flag.
        is_evaluation: Evaluation flag.
        
    Returns:
        DataLoader object.
    """
    dataset = BWEAudioDataset(
        manifest_path=manifest_path,
        config=config,
        is_validation=is_validation,
        is_evaluation=is_evaluation
    )
    
    # Validation/Evaluation uses batch size 1 to handle varying audio lengths or exact evaluations
    batch_size = 1 if (is_evaluation or is_validation) else config.dataset.batch_size
    shuffle = not (is_validation or is_evaluation)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0 if os.name == 'nt' else config.dataset.num_workers, # NT workers can be slower/leak memory in Windows
        pin_memory=config.dataset.pin_memory,
        drop_last=not (is_validation or is_evaluation)
    )
