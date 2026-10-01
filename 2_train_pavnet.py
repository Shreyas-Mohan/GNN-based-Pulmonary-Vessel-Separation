"""
Backward-compatibility wrapper for Step 2: Training Graph-PAVNet.
Main modular implementation is in `scripts/train_model.py` and `src`.
Re-exports key classes and functions for legacy compatibility.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Legacy re-exports
from src.data import OptimizedLungDataset, build_skeleton_graph, _knn_edges, patient_wise_split
from src.postprocessing import smooth_graph_predictions
from src.training import trial5_loss
from scripts.train_model import main

__all__ = [
    "OptimizedLungDataset",
    "build_skeleton_graph",
    "_knn_edges",
    "patient_wise_split",
    "smooth_graph_predictions",
    "trial5_loss",
    "main",
]

if __name__ == "__main__":
    main()
