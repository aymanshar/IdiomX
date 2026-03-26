# ============================================================
# Task 3 — Arabic Context -> English Canonical Idiom
# Retrieval Baseline
#
# Reproducible script:
# - runnable from CMD
# - callable from notebook via subprocess
# - saves metrics, predictions, plots, and embedding cache
# ============================================================

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------------------------------------
# Local shared tools (same folder import)
# ------------------------------------------------------------
from tools import (
    set_seed,
    load_csv_checked,
    ensure_text_pair_columns,
    ensure_dir,
    compute_topk_accuracy,
    compute_mrr,
)


# ------------------------------------------------------------
# Project root resolution
# ------------------------------------------------------------
def find_project_root(start: Path) -> Path:
    start = start.resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "deep_learning").exists() and (candidate / "notebooks").exists():
            return candidate
    raise FileNotFoundError(f"Could not locate IdiomX project root from: {start}")


PROJECT_ROOT = find_project_root(Path(__file__).resolve())

# ------------------------------------------------------------
# Standard paths
# ------------------------------------------------------------
DATASET_DIR = PROJECT_ROOT / "deep_learning" / "datasets" / "arabic_context_to_idiom"
IDIOM_BANK_PATH = PROJECT_ROOT / "deep_learning" / "datasets" / "idiom_bank" / "idiom_bank.csv"

TRAIN_PATH = DATASET_DIR / "train.csv"
VAL_PATH = DATASET_DIR / "validation.csv"
TEST_PATH = DATASET_DIR / "test.csv"

OUTPUT_DIR = PROJECT_ROOT / "deep_learning" / "models" / "arabic_context_to_idiom" / "retrieval"
CACHE_DIR = OUTPUT_DIR / "cache"
PRED_DIR = OUTPUT_DIR / "predictions"
PLOT_DIR = OUTPUT_DIR / "plots"

ensure_dir(OUTPUT_DIR)
ensure_dir(CACHE_DIR)
ensure_dir(PRED_DIR)
ensure_dir(PLOT_DIR)


# ------------------------------------------------------------
# CLI arguments
# ------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Task 3 Retrieval Baseline: Arabic Context -> English Canonical Idiom"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        help="SentenceTransformer model name"
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--top_k_max", type=int, default=5, help="Maximum K for ranking output")
    parser.add_argument("--mrr_cutoff", type=int, default=100, help="MRR cutoff rank")
    parser.add_argument(
        "--bank_mode",
        type=str,
        default="auto",
        choices=["auto", "idiom_bank", "train_only"],
        help=(
            "auto: use idiom_bank if present, otherwise fallback to train idioms; "
            "idiom_bank: force external idiom bank; "
            "train_only: use unique train target_text only"
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--rebuild_cache",
        action="store_true",
        help="Re-encode candidate bank even if cached embeddings exist"
    )
    return parser.parse_args()


# ------------------------------------------------------------
# Data loading
# ------------------------------------------------------------
def load_splits() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = load_csv_checked(TRAIN_PATH)
    val_df = load_csv_checked(VAL_PATH)
    test_df = load_csv_checked(TEST_PATH)

    train_df = ensure_text_pair_columns(train_df, input_col="input_text", target_col="target_text")
    val_df = ensure_text_pair_columns(val_df, input_col="input_text", target_col="target_text")
    test_df = ensure_text_pair_columns(test_df, input_col="input_text", target_col="target_text")

    return train_df, val_df, test_df


# ------------------------------------------------------------
# Candidate bank building
# ------------------------------------------------------------
def build_bank_from_train(train_df: pd.DataFrame) -> pd.DataFrame:
    bank_df = (
        train_df[["target_text"]]
        .drop_duplicates()
        .rename(columns={"target_text": "idiom_canonical"})
        .copy()
    )
    bank_df["idiom_canonical"] = bank_df["idiom_canonical"].astype(str).str.strip()
    bank_df["retrieval_text"] = bank_df["idiom_canonical"]
    return bank_df.reset_index(drop=True)


def build_bank_from_idiom_bank(idiom_bank_path: Path) -> pd.DataFrame:
    bank_df = load_csv_checked(idiom_bank_path)

    if "idiom_canonical" not in bank_df.columns:
        raise ValueError("idiom_bank.csv must contain column: idiom_canonical")

    bank_df = bank_df[bank_df["idiom_canonical"].notna()].copy()
    bank_df["idiom_canonical"] = bank_df["idiom_canonical"].astype(str).str.strip()

    # Best retrieval text available
    if "retrieval_text_multilingual" in bank_df.columns:
        retrieval_col = "retrieval_text_multilingual"
    elif "retrieval_text" in bank_df.columns:
        retrieval_col = "retrieval_text"
    elif "meaning_en" in bank_df.columns:
        retrieval_col = "meaning_en"
    else:
        retrieval_col = "idiom_canonical"

    bank_df[retrieval_col] = bank_df[retrieval_col].fillna("").astype(str).str.strip()

    # Fallback for empty retrieval text
    bank_df["retrieval_text"] = bank_df[retrieval_col]
    bank_df.loc[bank_df["retrieval_text"] == "", "retrieval_text"] = bank_df["idiom_canonical"]

    bank_df = bank_df[["idiom_canonical", "retrieval_text"]].drop_duplicates().reset_index(drop=True)
    return bank_df


def build_candidate_bank(train_df: pd.DataFrame, bank_mode: str) -> Tuple[pd.DataFrame, str]:
    if bank_mode == "train_only":
        return build_bank_from_train(train_df), "train_only"

    if bank_mode == "idiom_bank":
        if not IDIOM_BANK_PATH.exists():
            raise FileNotFoundError(f"idiom_bank.csv not found at: {IDIOM_BANK_PATH}")
        return build_bank_from_idiom_bank(IDIOM_BANK_PATH), "idiom_bank"

    # auto
    if IDIOM_BANK_PATH.exists():
        return build_bank_from_idiom_bank(IDIOM_BANK_PATH), "idiom_bank"
    return build_bank_from_train(train_df), "train_only"


# ------------------------------------------------------------
# Embedding cache
# ------------------------------------------------------------
def get_cache_paths(model_name: str, bank_name: str) -> Tuple[Path, Path]:
    safe_model = model_name.replace("/", "__").replace(":", "_")
    emb_path = CACHE_DIR / f"{bank_name}__{safe_model}__embeddings.npy"
    meta_path = CACHE_DIR / f"{bank_name}__{safe_model}__metadata.csv"
    return emb_path, meta_path


def encode_or_load_bank(
    model: SentenceTransformer,
    bank_df: pd.DataFrame,
    model_name: str,
    bank_name: str,
    batch_size: int,
    rebuild_cache: bool,
) -> np.ndarray:
    emb_path, meta_path = get_cache_paths(model_name, bank_name)

    if emb_path.exists() and meta_path.exists() and not rebuild_cache:
        print("\nLoading cached bank embeddings...")
        cached_meta = pd.read_csv(meta_path)
        if len(cached_meta) == len(bank_df):
            print("Cache hit.")
            return np.load(emb_path)

        print("Cache metadata length mismatch. Rebuilding cache...")

    print("\nEncoding candidate bank...")
    bank_embeddings = model.encode(
        bank_df["retrieval_text"].tolist(),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    np.save(emb_path, bank_embeddings)
    bank_df.to_csv(meta_path, index=False, encoding="utf-8-sig")
    print(f"Saved bank embeddings cache to: {emb_path}")
    print(f"Saved bank metadata cache to: {meta_path}")

    return bank_embeddings


# ------------------------------------------------------------
# Ranking / evaluation
# ------------------------------------------------------------
def reciprocal_rank(ranked_preds: List[str], gold: str, cutoff: int) -> float:
    for idx, pred in enumerate(ranked_preds[:cutoff], start=1):
        if pred == gold:
            return 1.0 / idx
    return 0.0


def evaluate_split(
    model: SentenceTransformer,
    split_df: pd.DataFrame,
    bank_df: pd.DataFrame,
    bank_embeddings: np.ndarray,
    batch_size: int,
    top_k_max: int,
    mrr_cutoff: int,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    print("\nEncoding split inputs...")
    query_texts = split_df["input_text"].tolist()
    gold_targets = split_df["target_text"].tolist()

    query_embeddings = model.encode(
        query_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    print("Computing cosine similarities...")
    sim = cosine_similarity(query_embeddings, bank_embeddings)

    bank_idioms = bank_df["idiom_canonical"].tolist()

    ranked_predictions = []
    prediction_rows = []

    top1_scores = []
    rr_scores = []
    found_rank_positions = []

    for i in range(sim.shape[0]):
        row_scores = sim[i]
        ranked_idx = np.argsort(row_scores)[::-1]
        ranked_idioms = [bank_idioms[j] for j in ranked_idx[:top_k_max]]
        ranked_predictions.append(ranked_idioms)

        gold = gold_targets[i]
        pred_top1 = ranked_idioms[0] if ranked_idioms else ""

        rr = reciprocal_rank(
            [bank_idioms[j] for j in ranked_idx[:mrr_cutoff]],
            gold,
            cutoff=mrr_cutoff,
        )
        rr_scores.append(rr)
        top1_scores.append(float(row_scores[ranked_idx[0]]))

        rank_found = None
        for rank, j in enumerate(ranked_idx[:mrr_cutoff], start=1):
            if bank_idioms[j] == gold:
                rank_found = rank
                break
        found_rank_positions.append(rank_found if rank_found is not None else 0)

        row = {
            "input_text": query_texts[i],
            "gold_idiom": gold,
            "pred_top1": pred_top1,
            "top1_similarity": float(row_scores[ranked_idx[0]]),
            "rank_of_gold_within_mrr_cutoff": rank_found if rank_found is not None else -1,
        }

        for k in range(1, top_k_max + 1):
            row[f"pred_top{k}"] = ranked_idioms[k - 1] if len(ranked_idioms) >= k else ""

        prediction_rows.append(row)

    metrics = {
        "num_samples": int(len(split_df)),
        "num_bank_candidates": int(len(bank_df)),
        "top1_accuracy": float(compute_topk_accuracy(ranked_predictions, gold_targets, k=1)),
        "top3_accuracy": float(compute_topk_accuracy(ranked_predictions, gold_targets, k=min(3, top_k_max))),
        "top5_accuracy": float(compute_topk_accuracy(ranked_predictions, gold_targets, k=min(5, top_k_max))),
        "mrr_at_cutoff": float(compute_mrr(
            [preds[:mrr_cutoff] for preds in ranked_predictions],
            gold_targets
        )),
        "mean_top1_similarity": float(np.mean(top1_scores)),
        "gold_found_rate_within_mrr_cutoff": float(np.mean(np.array(found_rank_positions) > 0)),
    }

    predictions_df = pd.DataFrame(prediction_rows)
    return predictions_df, metrics


# ------------------------------------------------------------
# Plotting
# ------------------------------------------------------------
def save_metrics_bar_plot(metrics: Dict[str, float], split_name: str, output_path: Path) -> None:
    labels = ["Top-1", "Top-3", "Top-5", "MRR"]
    values = [
        metrics["top1_accuracy"],
        metrics["top3_accuracy"],
        metrics["top5_accuracy"],
        metrics["mrr_at_cutoff"],
    ]

    plt.figure(figsize=(8, 5))
    plt.bar(labels, values)
    plt.ylim(0, 1)
    plt.title(f"Retrieval Metrics — {split_name}")
    plt.ylabel("Score")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_rank_histogram(predictions_df: pd.DataFrame, split_name: str, output_path: Path) -> None:
    ranks = predictions_df["rank_of_gold_within_mrr_cutoff"].copy()
    ranks = ranks[ranks > 0]

    plt.figure(figsize=(8, 5))
    if len(ranks) > 0:
        plt.hist(ranks, bins=30)
    plt.title(f"Gold Rank Distribution — {split_name}")
    plt.xlabel("Rank of Gold Idiom")
    plt.ylabel("Count")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


# ------------------------------------------------------------
# Save outputs
# ------------------------------------------------------------
def save_split_outputs(split_name: str, predictions_df: pd.DataFrame, metrics: Dict[str, float]) -> None:
    pred_path = PRED_DIR / f"{split_name}_predictions.csv"
    metrics_csv_path = OUTPUT_DIR / f"{split_name}_metrics.csv"
    metrics_json_path = OUTPUT_DIR / f"{split_name}_metrics.json"

    predictions_df.to_csv(pred_path, index=False, encoding="utf-8-sig")
    pd.DataFrame([metrics]).to_csv(metrics_csv_path, index=False, encoding="utf-8-sig")

    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    save_metrics_bar_plot(metrics, split_name, PLOT_DIR / f"{split_name}_metrics_bar.png")
    save_rank_histogram(predictions_df, split_name, PLOT_DIR / f"{split_name}_gold_rank_hist.png")

    print(f"\nSaved {split_name} predictions to: {pred_path}")
    print(f"Saved {split_name} metrics CSV to: {metrics_csv_path}")
    print(f"Saved {split_name} metrics JSON to: {metrics_json_path}")


# ------------------------------------------------------------
# Main experiment
# ------------------------------------------------------------
def run_experiment(args: argparse.Namespace) -> Dict[str, Dict[str, float]]:
    print("=" * 70)
    print("Task 3 Retrieval Baseline")
    print("=" * 70)
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("DATASET_DIR:", DATASET_DIR)
    print("OUTPUT_DIR:", OUTPUT_DIR)

    set_seed(args.seed)

    print("\nLoading dataset splits...")
    train_df, val_df, test_df = load_splits()
    print(f"Train size: {len(train_df):,}")
    print(f"Validation size: {len(val_df):,}")
    print(f"Test size: {len(test_df):,}")

    print("\nBuilding candidate bank...")
    bank_df, bank_name = build_candidate_bank(train_df, args.bank_mode)
    print(f"Bank source: {bank_name}")
    print(f"Bank size: {len(bank_df):,}")

    print("\nLoading embedding model...")
    print("Model:", args.model_name)
    model = SentenceTransformer(args.model_name)

    bank_embeddings = encode_or_load_bank(
        model=model,
        bank_df=bank_df,
        model_name=args.model_name,
        bank_name=bank_name,
        batch_size=args.batch_size,
        rebuild_cache=args.rebuild_cache,
    )

    print("\nEvaluating validation split...")
    val_predictions, val_metrics = evaluate_split(
        model=model,
        split_df=val_df,
        bank_df=bank_df,
        bank_embeddings=bank_embeddings,
        batch_size=args.batch_size,
        top_k_max=args.top_k_max,
        mrr_cutoff=args.mrr_cutoff,
    )
    save_split_outputs("validation", val_predictions, val_metrics)

    print("\nEvaluating test split...")
    test_predictions, test_metrics = evaluate_split(
        model=model,
        split_df=test_df,
        bank_df=bank_df,
        bank_embeddings=bank_embeddings,
        batch_size=args.batch_size,
        top_k_max=args.top_k_max,
        mrr_cutoff=args.mrr_cutoff,
    )
    save_split_outputs("test", test_predictions, test_metrics)

    summary = {
        "validation": val_metrics,
        "test": test_metrics,
    }

    summary_path = OUTPUT_DIR / "summary_metrics.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("Final Summary")
    print("=" * 70)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSaved summary to: {summary_path}")

    return summary

# ------------------------------------------------------------
# Public API (for notebook)
# ------------------------------------------------------------
def run_arabic_context_retrieval(args):
    return run_experiment(args)

def main():
    args = parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()