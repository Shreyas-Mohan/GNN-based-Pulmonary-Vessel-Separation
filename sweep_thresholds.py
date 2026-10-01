"""
Backward-compatibility wrapper for threshold sweep analysis.
Main modular implementation is in `scripts/sweep_thresholds.py` and `src`.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.sweep_thresholds import main

if __name__ == "__main__":
    main()
