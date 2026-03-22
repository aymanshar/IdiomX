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