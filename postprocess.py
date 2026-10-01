"""
Backward-compatibility wrapper for post-processing routines.
All core algorithms now live in `src.postprocessing`.
"""

from src.postprocessing import (
    confidence_guided_diffusion,
    caliber_adaptive_threshold,
    smooth_graph_predictions,
    map_skeleton_to_volume,
)

__all__ = [
    "confidence_guided_diffusion",
    "caliber_adaptive_threshold",
    "smooth_graph_predictions",
    "map_skeleton_to_volume",
]
