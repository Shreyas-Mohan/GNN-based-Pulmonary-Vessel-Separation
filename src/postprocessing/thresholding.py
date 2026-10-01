import torch
import numpy as np


def caliber_adaptive_threshold(
    prob_vein,
    radii,
    base_threshold=0.45,
    delta=0.05,
    sigma=1.5,
):
    """
    Caliber-Adaptive Thresholding: T(r) = base_threshold - delta * exp(- r / sigma).
    For large trunks (r > 2.0mm), T -> base_threshold (0.45).
    For small peripheral vessels (r <= 1.5mm, relevant to BV5), T -> 0.40,
    boosting peripheral vein recall where graph models typically lose sensitivity.

    Args:
        prob_vein (Tensor): Predicted vein probabilities [N]
        radii (Tensor or np.ndarray): Estimated vessel radius per node [N]
        base_threshold (float): Baseline trunk threshold (default 0.45)
        delta (float): Maximum threshold lowering for small calibers (default 0.05)
        sigma (float): Decay factor (default 1.5)

    Returns:
        Tensor: Binary node predictions [N] (0: Artery, 1: Vein)
    """
    if isinstance(radii, np.ndarray):
        radii_t = torch.from_numpy(radii).to(prob_vein.device).float()
    else:
        radii_t = radii.to(prob_vein.device).float()

    threshold_r = base_threshold - delta * torch.exp(- radii_t / sigma)
    return (prob_vein > threshold_r).long()
