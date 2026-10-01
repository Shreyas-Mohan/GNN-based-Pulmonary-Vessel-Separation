"""
Backward-compatibility wrapper for Step 3: Test Evaluation.
Main modular implementation is in `scripts/evaluate_test.py` and `src`.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Legacy re-exports
from src.postprocessing import smooth_graph_predictions, map_skeleton_to_volume
from src.evaluation import calculate_copd_biomarkers, evaluate_graph_predictions
from scripts.evaluate_test import main

__all__ = [
    "smooth_graph_predictions",
    "map_skeleton_to_volume",
    "calculate_copd_biomarkers",
    "evaluate_graph_predictions",
    "main",
]

if __name__ == "__main__":
    main()