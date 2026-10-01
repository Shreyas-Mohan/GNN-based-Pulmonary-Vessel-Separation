import os
from pathlib import Path

# Project Root Directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Core Data Paths
DATASET_DIR = PROJECT_ROOT / "dataset"
CLOUDS_TR_DIR = DATASET_DIR / "cloudsTr"
CLOUDS_TS_DIR = DATASET_DIR / "cloudsTs"
FEATURES_TR_DIR = DATASET_DIR / "featuresTr"
FEATURES_TS_DIR = DATASET_DIR / "featuresTs"
IMAGES_TR_DIR = DATASET_DIR / "imagesTr"
IMAGES_TS_DIR = DATASET_DIR / "imagesTs"
MASKS_TR_DIR = DATASET_DIR / "masksTr"
MASKS_TS_DIR = DATASET_DIR / "masksTs"
SEG_TR_DIR = DATASET_DIR / "segTr"
SEG_TS_DIR = DATASET_DIR / "segTs"

# Models & Weights Paths
MODELS_DIR = PROJECT_ROOT / "models"
PATCH_EXTRACTOR_WEIGHTS = MODELS_DIR / "patch_extractor.pth"
DEFAULT_CHECKPOINT = MODELS_DIR / "best_pavnet_final-5.pth"
EVAL_CHECKPOINT = MODELS_DIR / "best_pavnet_final-4.pth"

# Output Paths
PREDICTIONS_DIR = PROJECT_ROOT / "predictions"
RESULTS_DIR = PROJECT_ROOT / "results"
BENCHMARKS_DIR = RESULTS_DIR / "benchmarks"
VISUALIZATIONS_DIR = RESULTS_DIR / "visualizations"
DOCS_DIR = PROJECT_ROOT / "docs"

# Model Hyperparameters
RANDOM_SEED = 42
FEATURE_DIM = 584        # 3 coords + 1 radius + 4 direction/distance + 576 Fvv
NUM_CLASSES = 2          # 0: Artery, 1: Vein
MAX_HOPS = 3
PATCH_SIZE = 16
EMBED_DIM = 64
DEFAULT_OPERATING_THRESHOLD = 0.45

# Ensure critical execution environment variable is present
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
