import os
import time
import torch
import numpy as np
from sklearn.metrics import f1_score
from ..postprocessing.smoothing import smooth_graph_predictions
from .losses import trial5_loss


def train_one_epoch(
    model,
    train_loader,
    optimizer,
    class_weights,
    epoch_idx,
    total_epochs,
    device="cpu",
    gamma=2.0,
    lambda_topo=0.15,
    lambda_seed=0.05,
    log_interval=5,
):
    """Executes a single training epoch across all graph batches."""
    model.train()
    epoch_start = time.perf_counter()
    epoch_loss = 0.0
    epoch_focal = 0.0
    epoch_topo = 0.0

    print(f"\n[Epoch {epoch_idx + 1:02d}/{total_epochs}] Starting training ({len(train_loader)} batches)...")

    for batch_index, batch in enumerate(train_loader, start=1):
        batch_start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        batch = batch.to(device)
        batch.x = batch.x.float()

        out = model(batch)
        loss, f_val, t_val = trial5_loss(
            out,
            batch.y,
            batch.seed_mask,
            batch.edge_index,
            batch.radii,
            class_weights.to(device),
            gamma=gamma,
            lambda_topo=lambda_topo,
            lambda_seed=lambda_seed,
        )
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_focal += f_val
        epoch_topo += t_val

        if batch_index % log_interval == 0 or batch_index == len(train_loader):
            print(
                f"  Batch [{batch_index:02d}/{len(train_loader)}]: "
                f"loss={loss.item():.4f} (focal={f_val:.4f}, topo={t_val:.4f}), "
                f"avg={(epoch_loss/batch_index):.4f}, "
                f"time={time.perf_counter() - batch_start:.2f}s"
            )

    avg_loss = epoch_loss / len(train_loader)
    avg_focal = epoch_focal / len(train_loader)
    avg_topo = epoch_topo / len(train_loader)
    epoch_time = time.perf_counter() - epoch_start
    return avg_loss, avg_focal, avg_topo, epoch_time


def validate_one_epoch(
    model,
    val_loader,
    device="cpu",
    threshold=0.45,
    smooth_iterations=2,
):
    """Executes a validation pass and returns accuracy and Macro S-Dice."""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for val_batch in val_loader:
            val_batch = val_batch.to(device)
            val_batch.x = val_batch.x.float()
            val_out = model(val_batch)
            val_prob_vein = torch.exp(val_out[:, 1])
            val_preds = (val_prob_vein > threshold).long()
            val_smoothed = smooth_graph_predictions(val_batch.edge_index, val_preds, iterations=smooth_iterations)
            all_preds.append(val_smoothed.cpu())
            all_targets.append(val_batch.y.cpu())

    y_val_all = torch.cat(all_targets).numpy()
    p_val_all = torch.cat(all_preds).numpy()

    val_acc = 100.0 * (p_val_all == y_val_all).mean()
    val_dice = float(f1_score(y_val_all, p_val_all, average="macro"))
    return val_acc, val_dice
