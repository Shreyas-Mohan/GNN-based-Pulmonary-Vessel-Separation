#!/usr/bin/env python
"""
Step 1: Extract Aligned Multi-Planar Visual Features (Fvv).
Extracts 9-orthogonal plane patch embeddings using a deterministic PatchExtractor.
Guarantees identical feature projection bases across train and test sets.
"""

import os
import sys
import argparse
from pathlib import Path
import torch
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.models import PatchExtractor


def get_or_create_extractor(weights_path=config.PATCH_EXTRACTOR_WEIGHTS, patch_size=config.PATCH_SIZE, embed_dim=config.EMBED_DIM, seed=config.RANDOM_SEED):
    """
    Initializes PatchExtractor with a deterministic seed and saves its weights.
    Loads existing weights if present to guarantee identical feature projections.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    extractor = PatchExtractor(patch_size=patch_size, embed_dim=embed_dim)

    if os.path.isfile(weights_path):
        print(f"[+] Loading existing deterministic PatchExtractor weights from:\n    {weights_path}")
        extractor.load_state_dict(torch.load(weights_path, map_location="cpu"))
    else:
        print(f"[+] Creating deterministic PatchExtractor with seed={seed} and saving to:\n    {weights_path}")
        os.makedirs(os.path.dirname(os.path.abspath(weights_path)), exist_ok=True)
        torch.save(extractor.state_dict(), weights_path)

    extractor.eval()
    return extractor


def extract_set(extractor, clouds_dir, images_dir, out_dir, force=False, chunk_size=512):
    os.makedirs(out_dir, exist_ok=True)
    coords_dir = os.path.join(clouds_dir, "coordinates")

    file_names = sorted(
        f for f in os.listdir(coords_dir)
        if f.endswith(".pth") and os.path.isfile(os.path.join(images_dir, f.replace(".pth", ".nii.gz")))
    )

    print(f"\n=======================================================")
    print(f"Extracting features: {clouds_dir} -> {out_dir}")
    print(f"Found {len(file_names)} matching pairs.")
    print(f"=======================================================")

    with torch.no_grad():
        for f in tqdm(file_names, desc=f"Processing {os.path.basename(out_dir)}"):
            out_file = os.path.join(out_dir, f)
            if os.path.exists(out_file) and not force:
                continue

            coords = torch.load(os.path.join(coords_dir, f))[0]
            vol_path = os.path.join(images_dir, f.replace(".pth", ".nii.gz"))
            vol_img = sitk.ReadImage(vol_path)
            vol_arr = np.transpose(sitk.GetArrayFromImage(vol_img), (2, 1, 0))  # [X, Y, Z]

            fvv_chunks = []
            for start in range(0, len(coords), chunk_size):
                chunk_coords = coords[start : start + chunk_size]
                chunk_patches = [
                    extractor.extract_patch(vol_arr, tuple(center.int().tolist()))
                    for center in chunk_coords
                ]
                batch_patches = torch.stack(chunk_patches, dim=0)
                emb = extractor.transformer(batch_patches)  # [chunk_size, 576]
                fvv_chunks.append(emb)

            fvv_tensor = torch.cat(fvv_chunks, dim=0).float()
            assert fvv_tensor.shape == (len(coords), 576), f"Shape mismatch: {fvv_tensor.shape}"
            torch.save(fvv_tensor, out_file)


def main():
    parser = argparse.ArgumentParser(description="Extract aligned 576-dim transformer visual features.")
    parser.add_argument("--force", action="store_true", help="Force re-extraction of existing feature tensors")
    parser.add_argument("--split", choices=["all", "train", "test"], default="all", help="Target split to extract")
    args = parser.parse_args()

    extractor = get_or_create_extractor()

    if args.split in ("all", "train"):
        extract_set(
            extractor=extractor,
            clouds_dir=str(config.CLOUDS_TR_DIR),
            images_dir=str(config.IMAGES_TR_DIR),
            out_dir=str(config.FEATURES_TR_DIR),
            force=args.force,
        )

    if args.split in ("all", "test"):
        extract_set(
            extractor=extractor,
            clouds_dir=str(config.CLOUDS_TS_DIR),
            images_dir=str(config.IMAGES_TS_DIR),
            out_dir=str(config.FEATURES_TS_DIR),
            force=args.force,
        )

    print("\n[OK] Feature extraction completed successfully!")


if __name__ == "__main__":
    main()
