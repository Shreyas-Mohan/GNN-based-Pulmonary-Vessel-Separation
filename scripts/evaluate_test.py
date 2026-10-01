#!/usr/bin/env python
"""
Step 3: Evaluate Test Set & Reconstruct 3D NIfTI Volumes.
- Runs inference on the test split skeleton graphs.
- Applies vectorized graph smoothing (majority voting).
- Generates detailed case-by-case metrics and confusion matrix.
- Reconstructs dense 3D NIfTI segmentation volumes (masked to lung parenchyma).
- Computes COPD clinical imaging biomarkers (LAA-950, AVR, BV5).
"""

import os
import sys
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
from src.postprocessing import smooth_graph_predictions, map_skeleton_to_volume
from src.evaluation import evaluate_graph_predictions, calculate_copd_biomarkers


def main():
    parser = argparse.ArgumentParser(description="Evaluate Graph-PAVNet on the test split.")
    parser.add_argument("--checkpoint", type=str, default=str(config.EVAL_CHECKPOINT), help="Path to trained model weights")
    parser.add_argument("--threshold", type=float, default=config.DEFAULT_OPERATING_THRESHOLD, help="Vein decision threshold (default: 0.45)")
    parser.add_argument("--reconstruct-first", action="store_true", default=True, help="Reconstruct 3D NIfTI volume for first case")
    parser.add_argument("--reconstruct-all", action="store_true", default=False, help="Reconstruct 3D NIfTI volume for all test cases")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        # Fall back to default checkpoint if eval checkpoint doesn't exist
        checkpoint_path = config.DEFAULT_CHECKPOINT
        if not checkpoint_path.is_file():
            print(f"[-] Checkpoint not found: {args.checkpoint} or {checkpoint_path}")
            print("    Please train the model first with: python scripts/train_model.py")
            return

    print(f"[+] Loading model checkpoint from: {checkpoint_path}")
    model = HierarchicalGATNet(in_channels=config.FEATURE_DIM, out_channels=config.NUM_CLASSES, max_hops=config.MAX_HOPS).float()
    model.load_state_dict(torch.load(str(checkpoint_path), map_location="cpu"))
    model.eval()

    test_dataset = OptimizedLungDataset(
        clouds_dir=str(config.CLOUDS_TS_DIR),
        features_dir=str(config.FEATURES_TS_DIR),
        max_hops=config.MAX_HOPS,
    )

    out_dir = config.PREDICTIONS_DIR
    os.makedirs(out_dir, exist_ok=True)

    metrics_list = []
    cases_records = []
    total_gt_art, total_gt_vein = 0, 0
    total_pred_art, total_pred_vein = 0, 0
    total_aa, total_vv, total_av, total_va = 0, 0, 0, 0

    print(f"\n--- Evaluating Test Set ({len(test_dataset)} Cases, Threshold = {args.threshold:.2f}) ---")

    with torch.no_grad():
        for idx in range(len(test_dataset)):
            data = test_dataset[idx]
            data.x = data.x.float()

            prob_vein = torch.exp(model(data)[:, 1])
            preds = (prob_vein > args.threshold).long()
            smoothed = smooth_graph_predictions(data.edge_index, preds, iterations=2)

            res = evaluate_graph_predictions(data.y, smoothed, data.edge_index)
            metrics_list.append(res)

            total_gt_art += res["gt_art"]
            total_gt_vein += res["gt_vein"]
            total_pred_art += res["pred_art"]
            total_pred_vein += res["pred_vein"]
            total_aa += res["aa"]
            total_vv += res["vv"]
            total_av += res["av"]
            total_va += res["va"]

            fname = test_dataset.file_names[idx]
            cases_records.append({
                "idx": idx + 1,
                "name": fname,
                "dice": res["dice"],
                "ppv": res["ppv"],
                "recall": res["recall"],
                "bmc": res["bmc"],
                "gt_art": res["gt_art"],
                "gt_vein": res["gt_vein"],
                "pred_art": res["pred_art"],
                "pred_vein": res["pred_vein"],
                "aa": res["aa"],
                "vv": res["vv"],
                "av": res["av"],
                "va": res["va"],
                "pct_art": res["pct_art"],
            })

    print("\n" + "=" * 135)
    print(f"               CASE-BY-CASE BREAKDOWN (Threshold = {args.threshold:.2f})")
    print("=" * 135)
    print(f"{'Idx':<4} {'Case Name':<15} {'Dice':<8} {'PPV':<8} {'Recall':<8} {'BMC':<7} {'GT Art':<8} {'GT Vein':<9} {'Pred Art':<10} {'Pred Vein':<10} {'A->A':<7} {'V->V':<7} {'A->V':<7} {'V->A':<7} {'Art %':<7}")
    print("-" * 135)
    for r in cases_records:
        print(f"{r['idx']:<4} {r['name']:<15} {r['dice']:<8.4f} {r['ppv']:<8.4f} {r['recall']:<8.4f} {r['bmc']:<7.1f} {r['gt_art']:<8d} {r['gt_vein']:<9d} {r['pred_art']:<10d} {r['pred_vein']:<10d} {r['aa']:<7d} {r['vv']:<7d} {r['av']:<7d} {r['va']:<7d} {r['pct_art']:<6.1f}%")
    print("=" * 135)
    
    mean_dice = np.mean([m["dice"] for m in metrics_list])
    mean_ppv = np.mean([m["ppv"] for m in metrics_list])
    mean_rec = np.mean([m["recall"] for m in metrics_list])
    mean_bmc = np.mean([m["bmc"] for m in metrics_list])
    total_pct_art = 100.0 * total_pred_art / (total_pred_art + total_pred_vein)
    print(f"{'TOTAL (30 Cases)':<20} {mean_dice:<8.4f} {mean_ppv:<8.4f} {mean_rec:<8.4f} {mean_bmc:<7.1f} {total_gt_art:<8d} {total_gt_vein:<9d} {total_pred_art:<10d} {total_pred_vein:<10d} {total_aa:<7d} {total_vv:<7d} {total_av:<7d} {total_va:<7d} {total_pct_art:<6.1f}%")

    vein_rec = 100.0 * total_vv / total_gt_vein if total_gt_vein else 0
    art_rec = 100.0 * total_aa / total_gt_art if total_gt_art else 0
    vein_ppv = 100.0 * total_vv / total_pred_vein if total_pred_vein else 0
    art_ppv = 100.0 * total_aa / total_pred_art if total_pred_art else 0
    total_nodes = total_gt_art + total_gt_vein
    overall_acc = 100.0 * (total_aa + total_vv) / total_nodes

    print("\n-------------------------------------------------------")
    print(f"      CONFUSION MATRIX SUMMARY (Threshold = {args.threshold:.2f})")
    print("-------------------------------------------------------")
    print("                     | Predicted Artery | Predicted Vein  ")
    print("-------------------------------------------------------")
    print(f"Ground Truth Artery  | {total_aa:<16d} | {total_av:<16d} (Total: {total_gt_art})")
    print(f"Ground Truth Vein    | {total_va:<16d} | {total_vv:<16d} (Total: {total_gt_vein})")
    print("-------------------------------------------------------")
    print(f"Total Predicted      | {total_pred_art:<16d} | {total_pred_vein:<16d} | Total: {total_nodes}")
    print("-------------------------------------------------------")
    print(f"Overall Accuracy:                  {overall_acc:.2f}%")
    print(f"Mean S-Dice:                       {mean_dice:.4f} +/- {np.std([m['dice'] for m in metrics_list]):.4f}")
    print(f"Mean S-PPV:                        {mean_ppv:.4f} +/- {np.std([m['ppv'] for m in metrics_list]):.4f}")
    print(f"Mean S-Recall:                     {mean_rec:.4f} +/- {np.std([m['recall'] for m in metrics_list]):.4f}")
    print(f"Mean BMC (Fractures):              {mean_bmc:.1f} +/- {np.std([m['bmc'] for m in metrics_list]):.1f}")
    print(f"Vein Sensitivity (Recall):         {vein_rec:.2f}% ({total_vv}/{total_gt_vein})")
    print(f"Artery Sensitivity (Recall):       {art_rec:.2f}% ({total_aa}/{total_gt_art})")
    print(f"Vein Precision (PPV):              {vein_ppv:.2f}% ({total_vv}/{total_pred_vein})")
    print(f"Artery Precision (PPV):            {art_ppv:.2f}% ({total_aa}/{total_pred_art})")
    print(f"Artery-to-Vein Error Rate (A->V):  {100.0*total_av/total_gt_art:.2f}% ({total_av}/{total_gt_art})")
    print(f"Vein-to-Artery Error Rate (V->A):  {100.0*total_va/total_gt_vein:.2f}% ({total_va}/{total_gt_vein})")

    # 3D NIfTI Volume Reconstruction
    cases_to_reconstruct = range(len(test_dataset)) if args.reconstruct_all else ([0] if args.reconstruct_first else [])
    for c_idx in cases_to_reconstruct:
        case_name = test_dataset.file_names[c_idx]
        vol_name = case_name.replace(".pth", ".nii.gz")
        seg_path = config.SEG_TS_DIR / vol_name
        pred_path = out_dir / vol_name
        
        if seg_path.is_file():
            img_path = config.IMAGES_TS_DIR / vol_name
            mask_path = config.MASKS_TS_DIR / vol_name

            print(f"\n[+] Reconstructing 3D volume for {vol_name} (masked to lung parenchyma)...")
            case_data = test_dataset[c_idx]
            case_preds = (torch.exp(model(case_data)[:, 1]) > args.threshold).long()
            case_smoothed = smooth_graph_predictions(case_data.edge_index, case_preds, iterations=2)
            
            map_skeleton_to_volume(
                case_data.original_coords,
                case_smoothed,
                str(seg_path),
                str(pred_path),
                lung_mask_path=str(mask_path) if mask_path.is_file() else None,
            )
            print(f"[OK] Saved 3D prediction volume to: {pred_path}")

            if img_path.is_file() and mask_path.is_file():
                bm = calculate_copd_biomarkers(str(img_path), str(mask_path), str(pred_path))
                print(f"[+] Clinical Biomarkers for {vol_name}:")
                for k, v in bm.items():
                    print(f"    - {k}: {v}")


if __name__ == "__main__":
    main()
