import unittest
import torch
from utils.config import ConfigNode
from models.generator import Generator
from models.discriminator import DiscriminatorGroup

class TestBWEModels(unittest.TestCase):
    def setUp(self):
        self.config_dict = {
            "train": {
                "gradient_checkpointing": False
            },
            "generator": {
                "use_waveform_branch": True,
                "use_spectral_branch": False,
                "use_cross_attention": False,
                "waveform_encoder": {
                    "in_channels": 1,
                    "channels": 16,
                    "num_blocks": 2,
                    "strides": [2, 2, 2],
                    "kernel_sizes": [15, 15, 15],
                    "dilations": [1, 2, 4]
                },
                "decoder": {
                    "upsample_rates": [2, 2, 2],
                    "upsample_kernel_sizes": [16, 16, 16],
                    "channels": 16,
                    "resblock_kernel_sizes": [3, 7],
                    "resblock_dilations": [[1, 3], [1, 3]]
                }
            },
            "discriminator": {
                "mpd": {
                    "periods": [2, 3, 5],
                    "channels": 8
                },
                "msd": {
                    "scales": [1, 2],
                    "channels": 8
                },
                "spectral": {
                    "use_magnitude": True,
                    "use_phase": True,
                    "fft_sizes": [128, 256],
                    "hop_lengths": [32, 64],
                    "win_lengths": [128, 256],
                    "channels": 8
                }
            }
        }
        self.config = ConfigNode(self.config_dict)

    def test_generator_waveform_only(self):
        generator = Generator(self.config)
        x = torch.randn(2, 1, 8192) # smaller length for speed
        y = generator(x)
        self.assertEqual(y.shape, (2, 1, 8192))
        
        y.sum().backward()
        for p in generator.parameters():
            if p.requires_grad:
                self.assertIsNotNone(p.grad)

    def test_discriminators(self):
        disc_group = DiscriminatorGroup(self.config)
        y = torch.randn(2, 1, 8192)
        
        scores, fmaps = disc_group(y)
        
        # Expected number of discriminators:
        # MPD (3 periods) + MSD (2 scales) + Mag (2 FFTs) + Phase (2 FFTs) = 9 sub-discriminators
        self.assertEqual(len(scores), 9)
        self.assertEqual(len(fmaps), 9)
        
        # Verify scores shape
        for score in scores:
            self.assertEqual(score.shape[0], 2) # batch size
            
        # Verify backward pass on discriminators
        sum_scores = sum([score.sum() for score in scores])
        sum_scores.backward()
        
        for p in disc_group.parameters():
            if p.requires_grad:
                self.assertIsNotNone(p.grad)

    def test_generator_hybrid(self):
        # Enable spectral branch and cross-attention
        hybrid_config_dict = self.config_dict.copy()
        hybrid_config_dict["generator"] = self.config_dict["generator"].copy()
        hybrid_config_dict["generator"]["use_spectral_branch"] = True
        hybrid_config_dict["generator"]["use_cross_attention"] = True
        hybrid_config_dict["generator"]["spectral_encoder"] = {
            "fft_size": 256,
            "hop_length": 64,
            "win_length": 256,
            "channels": 8,
            "num_layers": 2
        }
        hybrid_config_dict["generator"]["attention"] = {
            "dim": 16,
            "num_heads": 2
        }
        config = ConfigNode(hybrid_config_dict)
        
        generator = Generator(config)
        
        x = torch.randn(2, 1, 8192)
        y = generator(x)
        self.assertEqual(y.shape, (2, 1, 8192))
        
        y.sum().backward()
        for p in generator.parameters():
            if p.requires_grad:
                self.assertIsNotNone(p.grad)

if __name__ == "__main__":
    unittest.main()
