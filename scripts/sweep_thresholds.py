#!/usr/bin/env python
"""
Threshold Sweeping & Post-Processing Optimization.
Evaluates graph predictions across thresholds from 0.55 down to 0.05 to analyze
operating curves, vein/artery balance, and branch-mismatch count (BMC).
"""

import os
import sys
import time
import argparse
from pathlib import Path
import torch
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.models import HierarchicalGATNet
from src.data import OptimizedLungDataset
from src.postprocessing import smooth_graph_predictions
from src.evaluation import evaluate_graph_predictions


def main():
    parser = argparse.ArgumentParser(description="Sweep decision thresholds across the test split.")
    parser.add_argument("--checkpoint", type=str, default=str(config.EVAL_CHECKPOINT), help="Model weights path")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        checkpoint_path = config.DEFAULT_CHECKPOINT
        if not checkpoint_path.is_file():
            print(f"[-] Checkpoint not found: {args.checkpoint} or {checkpoint_path}")
            return

    print(f"[+] Loading model from {checkpoint_path}...", flush=True)
    model = HierarchicalGATNet(in_channels=config.FEATURE_DIM, out_channels=config.NUM_CLASSES, max_hops=config.MAX_HOPS).float()
    model.load_state_dict(torch.load(str(checkpoint_path), map_location="cpu"))
    model.eval()

    test_dataset = OptimizedLungDataset(
        clouds_dir=str(config.CLOUDS_TS_DIR),
        features_dir=str(config.FEATURES_TS_DIR),
        max_hops=config.MAX_HOPS,
    )

    print(f"[+] Precomputing graph forward passes for all {len(test_dataset)} test cases...", flush=True)
    t0 = time.time()
    cached_cases = []
    with torch.no_grad():
        for idx in range(len(test_dataset)):
            data = test_dataset[idx]
            data.x = data.x.float()
            out = model(data)
            prob_vein = torch.exp(out[:, 1])
            cached_cases.append({
                "name": test_dataset.file_names[idx],
                "prob_vein": prob_vein,
                "y": data.y.cpu().numpy(),
                "edge_index": data.edge_index.cpu(),
                "num_nodes": data.y.size(0),
            })
    print(f"[OK] Forward passes completed in {time.time() - t0:.2f}s", flush=True)

    thresholds = [0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05]

    print("\n" + "=" * 126, flush=True)
    print(f"{'Thresh':<8} {'Macro-Dice':<12} {'Macro-PPV':<11} {'Macro-Rec':<11} {'Vein Rec':<11} {'Art Rec':<11} {'Vein PPV':<11} {'Art PPV':<11} {'Art %':<9} {'Vein %':<9} {'Mean BMC':<10} {'Accuracy':<9}", flush=True)
    print("=" * 126, flush=True)

    for th in thresholds:
        dice_list, ppv_list, rec_list, bmc_list = [], [], [], []
        total_gt_art, total_gt_vein = 0, 0
        total_pred_art, total_pred_vein = 0, 0
        total_aa, total_vv = 0, 0

        for case in cached_cases:
            preds = (case["prob_vein"] > th).long()
            smoothed = smooth_graph_predictions(case["edge_index"], preds, iterations=2)
            
            res = evaluate_graph_predictions(case["y"], smoothed, case["edge_index"])
            dice_list.append(res["dice"])
            ppv_list.append(res["ppv"])
            rec_list.append(res["recall"])
            bmc_list.append(res["bmc"])

            total_gt_art += res["gt_art"]
            total_gt_vein += res["gt_vein"]
            total_pred_art += res["pred_art"]
            total_pred_vein += res["pred_vein"]
            total_aa += res["aa"]
            total_vv += res["vv"]

        vein_rec = 100.0 * total_vv / total_gt_vein if total_gt_vein else 0
        art_rec = 100.0 * total_aa / total_gt_art if total_gt_art else 0
        vein_ppv = 100.0 * total_vv / total_pred_vein if total_pred_vein else 0
        art_ppv = 100.0 * total_aa / total_pred_art if total_pred_art else 0
        art_pct = 100.0 * total_pred_art / (total_pred_art + total_pred_vein)
        vein_pct = 100.0 * total_pred_vein / (total_pred_art + total_pred_vein)
        mean_dice = np.mean(dice_list)
        mean_ppv = np.mean(ppv_list)
        mean_rec = np.mean(rec_list)
        mean_bmc = np.mean(bmc_list)
        accuracy = 100.0 * (total_aa + total_vv) / (total_gt_art + total_gt_vein)

        print(
            f"{th:<8.2f} {mean_dice:<12.4f} {mean_ppv:<11.4f} {mean_rec:<11.4f} "
            f"{vein_rec:<10.2f}% {art_rec:<10.2f}% {vein_ppv:<10.2f}% {art_ppv:<10.2f}% "
            f"{art_pct:<8.1f}% {vein_pct:<8.1f}% {mean_bmc:<10.1f} {accuracy:<8.2f}%",
            flush=True,
        )

    print("=" * 126, flush=True)


if __name__ == "__main__":
    main()
