import torch
import torch.nn as nn
from typing import List, Tuple

def discriminator_loss(
    real_scores: List[torch.Tensor],
    fake_scores: List[torch.Tensor]
) -> torch.Tensor:
    """Calculate Least Squares GAN (LSGAN) loss for discriminators.
    
    L_D = 0.5 * sum( (D(y) - 1)^2 + D(G(x))^2 )
    
    Args:
        real_scores: List of scores for real audio from all sub-discriminators.
        fake_scores: List of scores for fake audio from all sub-discriminators.
        
    Returns:
        Discriminator loss tensor.
    """
    loss = 0.0
    for r_score, f_score in zip(real_scores, fake_scores):
        r_loss = torch.mean((r_score - 1.0) ** 2)
        f_loss = torch.mean(f_score ** 2)
        loss = loss + 0.5 * (r_loss + f_loss)
    return loss


def generator_loss(fake_scores: List[torch.Tensor]) -> torch.Tensor:
    """Calculate Least Squares GAN (LSGAN) loss for generator.
    
    L_G = 0.5 * sum( (D(G(x)) - 1)^2 )
    
    Args:
        fake_scores: List of scores for fake audio from all sub-discriminators.
        
    Returns:
        Generator adversarial loss tensor.
    """
    loss = 0.0
    for f_score in fake_scores:
        loss = loss + 0.5 * torch.mean((f_score - 1.0) ** 2)
    return loss


def feature_matching_loss(
    real_fmaps: List[List[torch.Tensor]],
    fake_fmaps: List[List[torch.Tensor]]
) -> torch.Tensor:
    """Calculate feature matching loss between discriminator feature maps.
    
    Args:
        real_fmaps: Feature maps of real audio for each sub-discriminator.
        fake_fmaps: Feature maps of generated audio for each sub-discriminator.
        
    Returns:
        Feature matching loss tensor.
    """
    loss = 0.0
    for r_fmap, f_fmap in zip(real_fmaps, fake_fmaps):
        for r_feat, f_feat in zip(r_fmap, f_fmap):
            loss = loss + torch.mean(torch.abs(r_feat.detach() - f_feat))
    return loss
