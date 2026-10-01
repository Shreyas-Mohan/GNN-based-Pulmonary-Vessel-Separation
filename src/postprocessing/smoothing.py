import torch


def smooth_graph_predictions(edge_index, predictions, iterations=2):
    """
    Vectorized majority voting smoothing over graph neighbors.
    Resolves isolated label noise by assigning the rounded mean neighbor label.

    Args:
        edge_index (Tensor): Edge indices [2, E]
        predictions (Tensor): Binary or categorical node predictions [N]
        iterations (int): Smoothing passes (default 2)

    Returns:
        Tensor: Smoothed predictions of shape [N]
    """
    smoothed = predictions.clone().float()
    num_nodes = predictions.size(0)
    src, dst = edge_index[0], edge_index[1]
    deg = torch.bincount(src, minlength=num_nodes).float()
    has_neighbors = deg > 0
    safe_deg = deg.clamp_min(1)

    for _ in range(iterations):
        neighbor_sum = torch.zeros(num_nodes, dtype=torch.float, device=predictions.device)
        neighbor_sum.index_add_(0, src, smoothed[dst])
        neighbor_mean = neighbor_sum / safe_deg
        smoothed = torch.where(has_neighbors, torch.round(neighbor_mean), smoothed)

    return smoothed.long()
