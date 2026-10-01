#!/usr/bin/env python
"""
Step 2: Train Graph-PAVNet Model.
Trains Hierarchical Graph Attention Network (HGAT) with:
1. Radius-Weighted Focal Loss (gamma=2.0, w=1/sqrt(r)) for peripheral vessels (BV5)
2. Differentiable Edge-Consistency Regularization (lambda_topo=0.15)
3. Zero-Patient Leakage Group Splitting
4. Validation tracking on Macro S-Dice with T=0.45 optimal threshold
"""

import os
import sys
import argparse
import time
from pathlib import Path
import torch
from torch.optim import Adam
from torch_geometric.loader import DataLoader

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.models import HierarchicalGATNet
from src.data import OptimizedLungDataset, patient_wise_split
from src.training.trainer import train_one_epoch, validate_one_epoch


def main():
    parser = argparse.ArgumentParser(description="Train Graph-PAVNet on thoracic skeleton graphs.")
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs (default: 40)")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size (default: 4)")
    parser.add_argument("--lr", type=float, default=0.0004, help="Initial learning rate (default: 0.0004)")
    parser.add_argument("--gamma", type=float, default=2.0, help="Focal loss gamma parameter (default: 2.0)")
    parser.add_argument("--lambda-topo", type=float, default=0.15, help="Topological regularization weight (default: 0.15)")
    parser.add_argument("--lambda-seed", type=float, default=0.05, help="Hilum seed loss weight (default: 0.05)")
    parser.add_argument("--output", type=str, default=str(config.DEFAULT_CHECKPOINT), help="Path to save best checkpoint")
    args = parser.parse_args()

    torch.set_num_threads(max(1, (os.cpu_count() or 1) // 2))
    num_workers = 0 if os.name == "nt" else 2
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[+] Device: {device}")
    print(f"[+] Loading dataset from: {config.CLOUDS_TR_DIR}")
    full_dataset = OptimizedLungDataset(
        clouds_dir=str(config.CLOUDS_TR_DIR),
        features_dir=str(config.FEATURES_TR_DIR),
        max_hops=config.MAX_HOPS,
    )

    # Patient-Wise Group Splitting (Zero Leakage)
    train_dataset, val_dataset, train_pts, val_pts = patient_wise_split(
        full_dataset, train_ratio=0.8, seed=config.RANDOM_SEED
    )
    print(f"\n[+] Patient-Wise Group Splitting (Zero Patient Leakage):")
    print(f"    Total Unique Patients: {len(train_pts) + len(val_pts)}")
    print(f"    Train Patients:        {len(train_pts)} ({len(train_dataset)} scans)")
    print(f"    Validation Patients:   {len(val_pts)} ({len(val_dataset)} scans)")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)

    # Class balance weights calculation
    class_counts = torch.zeros(2, dtype=torch.long)
    for file_name in full_dataset.file_names:
        label_path = os.path.join(full_dataset.clouds_dir, "artery_vein", file_name)
        labels = torch.load(label_path, map_location="cpu")[0].long() - 1
        class_counts += torch.bincount(labels, minlength=2)
    class_frequencies = class_counts.float() / class_counts.sum().clamp_min(1)
    class_weights = (1.0 / class_frequencies.clamp_min(1e-6)).float()
    print(f"Class counts: {class_counts.tolist()}, weights: {class_weights.tolist()}")

    model = HierarchicalGATNet(in_channels=config.FEATURE_DIM, out_channels=config.NUM_CLASSES, max_hops=config.MAX_HOPS).float().to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    checkpoint_path = Path(args.output)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_dice = 0.0
    best_val_acc = 0.0

    print(f"\n[+] Starting Training for {args.epochs} epochs...")
    print(f"[+] Key Improvements:")
    print(f"    1. Radius-Weighted Focal Loss (gamma={args.gamma}, w=1/sqrt(r)) for BV5 small peripheral vessels")
    print(f"    2. Differentiable Edge-Consistency Regularization (lambda_topo={args.lambda_topo}) to prevent BMC fractures")
    print(f"    3. Validation tracking on Macro S-Dice with T={config.DEFAULT_OPERATING_THRESHOLD} optimal threshold")
    print(f"[+] Model checkpoint will be saved to: {checkpoint_path}\n")

    for epoch in range(args.epochs):
        avg_loss, avg_focal, avg_topo, epoch_time = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            class_weights=class_weights,
            epoch_idx=epoch,
            total_epochs=args.epochs,
            device=device,
            gamma=args.gamma,
            lambda_topo=args.lambda_topo,
            lambda_seed=args.lambda_seed,
        )
        scheduler.step()

        # Validation Phase
        print(f"  Validating on {len(val_loader)} batches (T={config.DEFAULT_OPERATING_THRESHOLD})...")
        val_acc, val_dice = validate_one_epoch(
            model=model,
            val_loader=val_loader,
            device=device,
            threshold=config.DEFAULT_OPERATING_THRESHOLD,
        )

        checkpoint_msg = ""
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_val_acc = val_acc
            torch.save(model.state_dict(), str(checkpoint_path))
            checkpoint_msg = " | >>> CHECKPOINT SAVED (New Best S-Dice!) <<<"

        print(
            f"[Epoch {epoch + 1:02d}/{args.epochs:02d}] Complete | Loss: {avg_loss:.4f} (Focal: {avg_focal:.4f}, Topo: {avg_topo:.4f}) | "
            f"Val Acc: {val_acc:.2f}% | Val S-Dice: {val_dice:.4f} (Best: {best_val_dice:.4f}) | "
            f"Time: {epoch_time:.1f}s"
            f"{checkpoint_msg}"
        )

    print(f"\n[OK] Training complete! Best validation S-Dice: {best_val_dice:.4f} (Acc: {best_val_acc:.2f}%)")


if __name__ == "__main__":
    main()
