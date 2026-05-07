"""MFCC CNN implementation package for Stage 3 replication.

The package is imported by `scripts/run_mfcc_cnn.py` after that runner inserts
the `paper_replication` root into `sys.path`, so it works without installing the
course project as a Python package.
"""

from .dataset import CLASSES, CLASS_TO_IDX
from .train_eval import train_and_evaluate_fold

__all__ = ["CLASSES", "CLASS_TO_IDX", "train_and_evaluate_fold"]
