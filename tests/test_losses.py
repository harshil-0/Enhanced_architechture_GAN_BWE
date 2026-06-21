import unittest
import torch
from losses.adversarial import generator_loss, discriminator_loss, feature_matching_loss
from losses.spectral import MultiResolutionSTFTLoss, PhaseConsistencyLoss, MelSpectrogramLoss

class TestBWELosses(unittest.TestCase):
    def test_adversarial_losses(self):
        # Fake scores from 3 sub-discriminators, batch size 2
        fake_scores = [torch.randn(2, 1, requires_grad=True) for _ in range(3)]
        real_scores = [torch.randn(2, 1, requires_grad=True) for _ in range(3)]
        
        # Generator adversarial loss
        g_loss = generator_loss(fake_scores)
        self.assertTrue(g_loss.requires_grad)
        g_loss.backward()
        for s in fake_scores:
            self.assertIsNotNone(s.grad)
            
        # Zero out gradients
        for s in fake_scores:
            s.grad = None
            
        # Discriminator adversarial loss
        d_loss = discriminator_loss(real_scores, fake_scores)
        self.assertTrue(d_loss.requires_grad)
        d_loss.backward()
        for s in fake_scores:
            self.assertIsNotNone(s.grad)
        for s in real_scores:
            self.assertIsNotNone(s.grad)

    def test_feature_matching_loss(self):
        # Fake and real feature maps from 2 sub-discriminators, each with 2 layers of features
        real_fmaps = [
            [torch.randn(2, 8, 32, requires_grad=True), torch.randn(2, 16, 16, requires_grad=True)],
            [torch.randn(2, 8, 32, requires_grad=True), torch.randn(2, 16, 16, requires_grad=True)]
        ]
        fake_fmaps = [
            [torch.randn(2, 8, 32, requires_grad=True), torch.randn(2, 16, 16, requires_grad=True)],
            [torch.randn(2, 8, 32, requires_grad=True), torch.randn(2, 16, 16, requires_grad=True)]
        ]
        
        fm_loss = feature_matching_loss(real_fmaps, fake_fmaps)
        self.assertTrue(fm_loss.requires_grad)
        fm_loss.backward()
        
        # Check that fake features get gradients (generator parameters flow through them)
        for sub_fmap in fake_fmaps:
            for feat in sub_fmap:
                self.assertIsNotNone(feat.grad)

    def test_spectral_losses(self):
        y_pred = torch.randn(2, 1, 8192, requires_grad=True)
        y_true = torch.randn(2, 1, 8192)
        
        # 1. Multi-Resolution STFT Loss
        mr_stft_loss_fn = MultiResolutionSTFTLoss()
        sc_loss, log_mag_loss = mr_stft_loss_fn(y_pred, y_true)
        self.assertTrue(sc_loss.requires_grad)
        self.assertTrue(log_mag_loss.requires_grad)
        
        # 2. Phase Consistency Loss
        phase_loss_fn = PhaseConsistencyLoss()
        p_loss = phase_loss_fn(y_pred, y_true)
        self.assertTrue(p_loss.requires_grad)
        
        # 3. Mel Spectrogram Loss
        mel_loss_fn = MelSpectrogramLoss(sample_rate=16000)
        m_loss = mel_loss_fn(y_pred, y_true)
        self.assertTrue(m_loss.requires_grad)
        
        # Backward check
        total_spectral = sc_loss + log_mag_loss + p_loss + m_loss
        total_spectral.backward()
        self.assertIsNotNone(y_pred.grad)

if __name__ == "__main__":
    unittest.main()
