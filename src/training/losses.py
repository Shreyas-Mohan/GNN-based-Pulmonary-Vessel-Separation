import torch


def trial5_loss(
    predictions,
    targets,
    seed_mask,
    edge_index,
    radii,
    class_weights,
    gamma=2.0,
    lambda_topo=0.15,
    lambda_seed=0.05,
):
    """
    Trial 5 Multi-Objective Loss Formulation:
    1. Radius-Weighted Focal Loss (boosts focus on small peripheral vessels <= 1.5 for BV5).
    2. Differentiable Edge-Consistency Regularization (penalizes label fractures across adjacent connected nodes).
    3. Hilum Seed Loss (anchors confident anatomical origins).

    Args:
        predictions (Tensor): Log-softmax predictions [N, 2]
        targets (Tensor): Ground truth labels [N] (0: artery, 1: vein)
        seed_mask (Tensor): Boolean mask of hilum seed nodes [N]
        edge_index (Tensor): Graph edge indices [2, E]
        radii (Tensor): Node radius estimates [N]
        class_weights (Tensor): Class balancing weights [2]
        gamma (float): Focal loss focusing parameter (default 2.0)
        lambda_topo (float): Topological regularization weight (default 0.15)
        lambda_seed (float): Seed anchoring weight (default 0.05)

    Returns:
        tuple[Tensor, float, float]: (total_loss, focal_loss_val, topo_loss_val)
    """
    device = predictions.device
    probs = torch.exp(predictions)  # [N, 2]
    p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1).clamp(min=1e-7, max=1.0 - 1e-7)
    log_p_t = predictions.gather(1, targets.unsqueeze(1)).squeeze(1)

    # 1. Radius weights for small peripheral vessels (BV5 clinical biomarker focus)
    r_clamped = torch.clamp(radii, min=0.5, max=5.0)
    w_radius = 1.0 / torch.sqrt(r_clamped)
    w_radius = w_radius / w_radius.mean().clamp_min(1e-6)

    # Class balancing weight
    cw = class_weights.to(device)
    alpha_t = cw[targets]

    # Focal modulation term: (1 - p_t)^gamma
    focal_weight = torch.pow(1.0 - p_t, gamma)

    # Node-wise focal loss
    node_loss = - w_radius * alpha_t * focal_weight * log_p_t
    main_loss = node_loss.mean()

    # 2. Hilum Seed Loss (anchoring central vessel trunks)
    if seed_mask.any():
        seed_p_t = p_t[seed_mask]
        seed_log_p_t = log_p_t[seed_mask]
        seed_alpha_t = alpha_t[seed_mask]
        seed_focal = torch.pow(1.0 - seed_p_t, gamma)
        seed_loss = (- seed_alpha_t * seed_focal * seed_log_p_t).mean()
    else:
        seed_loss = 0.0

    # 3. Differentiable Edge-Consistency Regularization (Topological BMC Loss)
    src, dst = edge_index[0], edge_index[1]
    prob_vein = probs[:, 1]
    topo_diff = prob_vein[src] - prob_vein[dst]
    topo_loss = torch.mean(topo_diff ** 2)

    total_loss = (1.0 - lambda_seed) * main_loss + lambda_seed * seed_loss + lambda_topo * topo_loss
    return total_loss, main_loss.item(), topo_loss.item()
