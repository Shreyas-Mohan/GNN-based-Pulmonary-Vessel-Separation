import torch


def confidence_guided_diffusion(
    edge_index,
    prob_vein,
    threshold=0.45,
    high_conf=0.70,
    low_conf=0.30,
    iterations=3,
    affinity_sigma=0.35,
):
    """
    Confidence-Guided Edge-Affinity Diffusion (Track 1 Post-Processing).

    1. Identifies high-confidence anchor nodes (P >= high_conf or P <= low_conf).
    2. Computes soft edge affinities: edges between nodes with similar probabilities have weight ~ 1.0;
       spurious cross-vessel shortcut edges between opposite-class vessels are downweighted towards 0.
    3. Diffuses spatial-topological consensus from high-confidence trunks to ambiguous peripheral vessels (BV5).
    4. Enforces continuous branches and cuts BMC fractures without retraining.

    Args:
        edge_index (Tensor): Edge indices [2, E]
        prob_vein (Tensor): Predicted vein probabilities [N]
        threshold (float): Decision threshold (default 0.45)
        high_conf (float): Vein anchor confidence threshold (default 0.70)
        low_conf (float): Artery anchor confidence threshold (default 0.30)
        iterations (int): Diffusion steps (default 3)
        affinity_sigma (float): Affinity Gaussian bandwidth (default 0.35)

    Returns:
        tuple[Tensor, Tensor]: (binary_class_predictions, diffused_probabilities)
    """
    device = prob_vein.device
    src, dst = edge_index[0], edge_index[1]
    num_nodes = prob_vein.size(0)
    current_probs = prob_vein.clone().float()

    is_anchor = (prob_vein >= high_conf) | (prob_vein <= low_conf)

    for _ in range(iterations):
        prob_diff = torch.abs(current_probs[src] - current_probs[dst])
        edge_weight = torch.exp(- (prob_diff / affinity_sigma) ** 2)

        weighted_sum = torch.zeros(num_nodes, device=device)
        weighted_sum.index_add_(0, src, current_probs[dst] * edge_weight)

        deg = torch.zeros(num_nodes, device=device)
        deg.index_add_(0, src, edge_weight)
        deg = deg.clamp_min(1e-6)

        diffused = weighted_sum / deg
        current_probs = torch.where(is_anchor, current_probs * 0.7 + diffused * 0.3, diffused)

    return (current_probs > threshold).long(), current_probs
