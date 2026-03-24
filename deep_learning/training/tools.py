"""
Shared utility functions for the IdiomX deep learning pipeline.

Author: Ayman Sharara
Project: IdiomX: A Multi-Task Framework for Idiom Understanding Using Transformer Models
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import sys

def get_project_root(current_file: Optional[str] = None, levels_up: int = 2) -> Path:
    """
    Resolve project root robustly.

    Parameters
    ----------
    current_file : str | None
        Usually pass __file__ from a Python script.
        If None, returns Path.cwd().
    levels_up : int
        Number of parent levels to move up when current_file is provided.

    Returns
    -------
    Path
        Project root path.
    """
    if current_file is None:
        return Path.cwd()

    return Path(current_file).resolve().parents[levels_up]


def ensure_dir(path: Path) -> Path:
    """Create directory if it does not exist and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def clean_text_column(series: pd.Series) -> pd.Series:
    """Convert a pandas Series to clean stripped strings."""
    return series.fillna("").astype(str).str.strip()

def clean_text_series(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()

def clean_context_target_df(
    df: pd.DataFrame,
    input_col: str = "input_text",
    target_col: str = "target_text"
) -> pd.DataFrame:
    """
    Clean a context-target dataframe and remove empty rows.
    """
    df = df.copy()

    if input_col in df.columns:
        df[input_col] = clean_text_column(df[input_col])

    if target_col in df.columns:
        df[target_col] = clean_text_column(df[target_col])

    if input_col in df.columns and target_col in df.columns:
        df = df[
            (df[input_col] != "") &
            (df[target_col] != "")
        ].copy()

    return df.reset_index(drop=True)


def load_csv_checked(path: Path, low_memory: bool = False) -> pd.DataFrame:
    """Load CSV with a clear existence check."""
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    return pd.read_csv(path, low_memory=low_memory)


def print_basic_df_info(name: str, df: pd.DataFrame) -> None:
    """Print concise dataframe diagnostics."""
    print(f"{name}: {df.shape[0]:,} rows × {df.shape[1]:,} columns")
    print(f"Columns: {list(df.columns)}")


def get_default_data_paths(base_dir: Path) -> dict:
    """Return standard project data paths."""
    return {
        "data_dir": base_dir / "data",
        "raw_dir": base_dir / "data" / "raw",
        "processed_dir": base_dir / "data" / "processed",
        "splits_dir": base_dir / "data" / "splits",
        "figures_dir": base_dir / "figures",
    }

# Evaluation Metrics (Ranking Tasks)
# ============================================

def compute_topk_accuracy(predictions, targets, k=1):
    """
    predictions: list of lists (ranked predictions per sample)
    targets: list of true labels
    """
    correct = 0

    for preds, target in zip(predictions, targets):
        if target in preds[:k]:
            correct += 1

    return correct / len(targets)


def compute_mrr(predictions, targets):
    """
    Mean Reciprocal Rank
    """
    total_score = 0.0

    for preds, target in zip(predictions, targets):
        if target in preds:
            rank = preds.index(target) + 1
            total_score += 1.0 / rank

    return total_score / len(targets)

def ensure_text_pair_columns(df: pd.DataFrame, input_col: str, target_col: str) -> pd.DataFrame:
    df = df.copy()
    df[input_col] = clean_text_series(df[input_col])
    df[target_col] = clean_text_series(df[target_col])
    df = df[(df[input_col] != "") & (df[target_col] != "")].reset_index(drop=True)
    return df


def ensure_single_text_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df = df.copy()
    df[col] = clean_text_series(df[col])
    df = df[df[col] != ""].reset_index(drop=True)
    return df