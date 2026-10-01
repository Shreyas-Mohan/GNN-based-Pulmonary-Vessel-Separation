import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score


def evaluate_graph_predictions(y_true, y_pred, edge_index=None):
    """
    Computes standard topological and classification metrics on graph predictions.

    Args:
        y_true (np.ndarray or Tensor): Ground truth binary node labels (0: Artery, 1: Vein)
        y_pred (np.ndarray or Tensor): Predicted binary node labels (0: Artery, 1: Vein)
        edge_index (Tensor, optional): Graph edges [2, E] to compute BMC (Branch Mismatch Count)

    Returns:
        dict: Metrics dictionary containing dice, ppv, recall, accuracy, bmc, confusion counts.
    """
    if torch.is_tensor(y_true):
        y_true = y_true.detach().cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.detach().cpu().numpy()

    dice = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    ppv = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

    gt_art = int(np.sum(y_true == 0))
    gt_vein = int(np.sum(y_true == 1))
    pred_art = int(np.sum(y_pred == 0))
    pred_vein = int(np.sum(y_pred == 1))

    aa = int(np.sum((y_true == 0) & (y_pred == 0)))
    vv = int(np.sum((y_true == 1) & (y_pred == 1)))
    av = int(np.sum((y_true == 0) & (y_pred == 1)))
    va = int(np.sum((y_true == 1) & (y_pred == 0)))

    total_nodes = len(y_true)
    accuracy = 100.0 * (aa + vv) / total_nodes if total_nodes > 0 else 0.0

    bmc = 0.0
    if edge_index is not None:
        if torch.is_tensor(edge_index):
            edges = edge_index.detach().cpu()
        else:
            edges = torch.from_numpy(edge_index)
        pred_tensor = torch.from_numpy(y_pred)
        mismatches = (pred_tensor[edges[0]] != pred_tensor[edges[1]]).sum().item()
        bmc = mismatches / 2.0

    pct_art = 100.0 * pred_art / total_nodes if total_nodes > 0 else 0.0
    pct_vein = 100.0 * pred_vein / total_nodes if total_nodes > 0 else 0.0

    vein_rec = 100.0 * vv / gt_vein if gt_vein > 0 else 0.0
    art_rec = 100.0 * aa / gt_art if gt_art > 0 else 0.0
    vein_ppv = 100.0 * vv / pred_vein if pred_vein > 0 else 0.0
    art_ppv = 100.0 * aa / pred_art if pred_art > 0 else 0.0

    return {
        "dice": dice,
        "ppv": ppv,
        "recall": rec,
        "accuracy": accuracy,
        "bmc": bmc,
        "gt_art": gt_art,
        "gt_vein": gt_vein,
        "pred_art": pred_art,
        "pred_vein": pred_vein,
        "pct_art": pct_art,
        "pct_vein": pct_vein,
        "aa": aa,
        "vv": vv,
        "av": av,
        "va": va,
        "vein_recall": vein_rec,
        "artery_recall": art_rec,
        "vein_ppv": vein_ppv,
        "artery_ppv": art_ppv,
    }
