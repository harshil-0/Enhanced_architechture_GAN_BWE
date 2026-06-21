import os
import unittest
import json
import torch
import soundfile as sf
import numpy as np
import shutil
from utils.config import ConfigNode
from utils.degradation import DegradationPipeline
from datasets.manifest import create_manifests
from datasets.dataset import BWEAudioDataset, get_dataloader

class TestDatasetAndDegradation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a temp directory for dummy audio and manifests
        cls.temp_dir = os.path.abspath("temp_test_data")
        os.makedirs(cls.temp_dir, exist_ok=True)
        
        # Generate 3 dummy WAV files (sine waves of different frequencies)
        cls.test_files = []
        cls.sample_rate = 16000
        cls.duration = 2.0  # seconds
        t = np.linspace(0, cls.duration, int(cls.sample_rate * cls.duration), endpoint=False)
        
        for idx, freq in enumerate([220, 440, 880]):
            audio = 0.5 * np.sin(2 * np.pi * freq * t)
            filepath = os.path.join(cls.temp_dir, f"sine_{idx}.wav")
            sf.write(filepath, audio, cls.sample_rate)
            cls.test_files.append(filepath)
            
        # Set up a mock configuration
        cls.config_dict = {
            "audio": {
                "input_sr": 8000,
                "target_sr": 16000,
                "segment_length": 16384,
                "degradation_type": "g711_mu"
            },
            "dataset": {
                "train_manifest": os.path.join(cls.temp_dir, "train.json"),
                "val_manifest": os.path.join(cls.temp_dir, "val.json"),
                "test_manifest": os.path.join(cls.temp_dir, "test.json"),
                "batch_size": 2,
                "num_workers": 0,
                "pin_memory": False,
                "crop_type": "random"
            }
        }
        cls.config = ConfigNode(cls.config_dict)
        
        # Generate manifests
        cls.train_m, cls.val_m, cls.test_m = create_manifests(
            dataset_dir=cls.temp_dir,
            output_dir=cls.temp_dir,
            train_ratio=0.34,
            val_ratio=0.33,
            test_ratio=0.33,
            seed=42
        )

    @classmethod
    def tearDownClass(cls):
        # Clean up temporary test directory
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)

    def test_degradation_pipeline(self):
        degrader = DegradationPipeline(self.config.audio)
        # Generate a fake wideband wave: shape (1, 32000)
        wav = torch.randn(1, 32000)
        
        # Run degradation with mu-law
        nb_wav = degrader(wav)
        self.assertEqual(nb_wav.shape, wav.shape)
        
        # Test A-law
        degrader.degradation_type = "g711_a"
        nb_wav_a = degrader(wav)
        self.assertEqual(nb_wav_a.shape, wav.shape)
        
        # Test GSM-sim
        degrader.degradation_type = "gsm_sim"
        nb_wav_gsm = degrader(wav)
        self.assertEqual(nb_wav_gsm.shape, wav.shape)

    def test_dataset_loading(self):
        # Instantiate dataset
        dataset = BWEAudioDataset(
            manifest_path=self.train_m,
            config=self.config,
            is_validation=False
        )
        
        # Check dataset length
        self.assertGreater(len(dataset), 0)
        
        # Load item
        nb_wav, gt_wav = dataset[0]
        self.assertEqual(nb_wav.shape, (1, self.config.audio.segment_length))
        self.assertEqual(gt_wav.shape, (1, self.config.audio.segment_length))

    def test_dataloader(self):
        dataloader = get_dataloader(
            manifest_path=self.train_m,
            config=self.config,
            is_validation=False
        )
        
        for nb_batch, gt_batch in dataloader:
            self.assertEqual(nb_batch.shape, (1, 1, self.config.audio.segment_length))
            self.assertEqual(gt_batch.shape, (1, 1, self.config.audio.segment_length))
            break

if __name__ == "__main__":
    unittest.main()
