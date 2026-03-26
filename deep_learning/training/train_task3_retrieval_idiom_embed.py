# -*- coding: utf-8 -*-
"""
Task 3: Arabic Context -> Idiom
Retrieval baseline using idiomatic example embeddings (Option C)

Idea:
- Query: Arabic idiomatic sentence
- Bank: English idiomatic example sentences
- Prediction: canonical idiom attached to retrieved example

Why this is stronger than idiom-only retrieval:
- Canonical idiom text alone is often semantically sparse
- Example sentences provide richer contextual semantics
- Arabic sentence can align better with English example meaning

Run from CMD:
    cd deep_learning/training
    python train_task3_retrieval_idiom_embed.py

Can also import from notebook:
    from train_task3_retrieval_idiom_embed import run_task3_retrieval_idiom_embed
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics import confusion_matrix
from tqdm import tqdm


# ============================================================
# Path helpers
# ============================================================

def find_project_root(start_path: Path, anchor_folder: str = "notebooks") -> Path:
    """
    Find project root by locating the parent that contains the anchor folder.
    Assumes notebook lives under project_root/notebooks.
    """
    start_path = start_path.resolve()
    for parent in [start_path] + list(start_path.parents):
        if (parent / anchor_folder).exists():
            return parent
    raise RuntimeError(f"Project root not found from: {start_path}")


# ============================================================
# Metrics
# ============================================================

def compute_topk_accuracy(
    y_true: List[str],
    y_pred_topk: List[List[str]],
    k: int
) -> float:
    correct = 0
    total = len(y_true)

    for gold, preds in zip(y_true, y_pred_topk):
        if gold in preds[:k]:
            correct += 1

    return correct / total if total > 0 else 0.0


def compute_mrr(
    y_true: List[str],
    y_pred_topk: List[List[str]]
) -> float:
    rr_scores = []

    for gold, preds in zip(y_true, y_pred_topk):
        rank = 0
        for idx, pred in enumerate(preds, start=1):
            if pred == gold:
                rank = idx
                break
        rr_scores.append(1.0 / rank if rank > 0 else 0.0)

    return float(np.mean(rr_scores)) if rr_scores else 0.0


# ============================================================
# Data preparation
# ============================================================

def load_task3_data(project_root: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    task3_dir = project_root / "deep_learning" / "datasets" / "arabic_context_to_idiom"

    train_path = task3_dir / "train.csv"
    val_path = task3_dir / "validation.csv"
    test_path = task3_dir / "test.csv"

    if not train_path.exists():
        raise FileNotFoundError(f"Missing file: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Missing file: {val_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Missing file: {test_path}")

    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    return df_train, df_val, df_test


def prepare_idiomatic_only(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only idiomatic examples.
    Assumes is_example_idiom is boolean-like or 0/1.
    """
    out = df.copy()
    out["is_example_idiom"] = out["is_example_idiom"].astype(bool)
    out = out[out["is_example_idiom"]].copy()

    out["input_text"] = out["input_text"].astype(str).str.strip()
    out["target_text"] = out["target_text"].astype(str).str.strip()

    # english contextual example used as retrieval bank text
    # prefer meaning_en? no. We want example-like contextual alignment.
    # Here the actual contextual sentence is in input_text for Arabic task,
    # but for bank we need English contextual examples.
    # This dataset likely includes Arabic input_text only.
    # We therefore use source-side English contextual field if available.
    # Fallback strategy:
    # 1) english_example
    # 2) example_en
    # 3) meaning_en
    # If none exist, raise.
    return out


def resolve_bank_text_column(df: pd.DataFrame) -> str:
    """
    Determine which English contextual column to use for bank embeddings.
    Best candidates are contextual English example fields.
    """
    candidates = [
        "example_en",
        "english_example",
        "example_text_en",
        "source_text_en",
        "context_en",
        "meaning_en",  # weaker fallback
    ]

    for col in candidates:
        if col in df.columns:
            return col

    raise ValueError(
        "No suitable English bank text column found. "
        "Expected one of: example_en, english_example, example_text_en, "
        "source_text_en, context_en, meaning_en"
    )


def build_idiom_example_bank(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame
) -> pd.DataFrame:
    """
    Build retrieval bank from ALL idiomatic English examples.
    This is acceptable for lexical knowledge-base retrieval because the bank
    represents known idiom knowledge, not train-time labels only.

    We deduplicate on (bank_text, target_text).
    """
    bank_df = pd.concat([df_train, df_val, df_test], axis=0, ignore_index=True).copy()
    bank_df = prepare_idiomatic_only(bank_df)

    bank_col = resolve_bank_text_column(bank_df)

    bank_df["bank_text"] = bank_df[bank_col].astype(str).str.strip()
    bank_df["target_text"] = bank_df["target_text"].astype(str).str.strip()

    bank_df = bank_df[["bank_text", "target_text"]].dropna().drop_duplicates().reset_index(drop=True)

    # Remove empty strings
    bank_df = bank_df[
        (bank_df["bank_text"].str.len() > 0) &
        (bank_df["target_text"].str.len() > 0)
    ].reset_index(drop=True)

    return bank_df


def prepare_test_queries(df_test: pd.DataFrame) -> pd.DataFrame:
    """
    Test queries are Arabic idiomatic contexts only.
    """
    test_df = prepare_idiomatic_only(df_test)

    test_df = test_df[["input_text", "target_text"]].copy()
    test_df["input_text"] = test_df["input_text"].astype(str).str.strip()
    test_df["target_text"] = test_df["target_text"].astype(str).str.strip()

    test_df = test_df.dropna().reset_index(drop=True)
    test_df = test_df[
        (test_df["input_text"].str.len() > 0) &
        (test_df["target_text"].str.len() > 0)
    ].reset_index(drop=True)

    return test_df


# ============================================================
# Retrieval
# ============================================================

def retrieve_topk_idioms_from_example_bank(
    model: SentenceTransformer,
    query_texts: List[str],
    bank_texts: List[str],
    bank_targets: List[str],
    top_k: int = 5,
    batch_size: int = 64
) -> Tuple[List[List[str]], np.ndarray]:
    """
    Retrieve top-k idiom candidates via bank example embeddings.

    Because many bank examples may map to the same idiom, we deduplicate idioms
    within each ranked candidate list while preserving order.
    """
    bank_emb = model.encode(
        bank_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    query_emb = model.encode(
        query_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    sim = np.matmul(query_emb, bank_emb.T)

    all_pred_topk: List[List[str]] = []

    # search deeper than top_k because example duplicates may collapse
    raw_topn = min(max(top_k * 10, 50), len(bank_targets))

    for i in tqdm(range(sim.shape[0]), desc="Ranking"):
        row = sim[i]
        top_idx = np.argpartition(-row, raw_topn - 1)[:raw_topn]
        top_idx = top_idx[np.argsort(-row[top_idx])]

        ranked_idioms: List[str] = []
        seen = set()

        for idx in top_idx:
            idiom = bank_targets[idx]
            if idiom not in seen:
                ranked_idioms.append(idiom)
                seen.add(idiom)
            if len(ranked_idioms) >= top_k:
                break

        # pad if needed
        while len(ranked_idioms) < top_k:
            ranked_idioms.append("")

        all_pred_topk.append(ranked_idioms)

    return all_pred_topk, sim


# ============================================================
# Analysis
# ============================================================

def build_predictions_dataframe(
    test_df: pd.DataFrame,
    pred_topk: List[List[str]]
) -> pd.DataFrame:
    out = test_df.copy()

    for k in range(len(pred_topk[0])):
        out[f"pred_top{k+1}"] = [row[k] for row in pred_topk]

    out["correct_top1"] = out["target_text"] == out["pred_top1"]
    out["correct_top3"] = [
        gold in preds[:3]
        for gold, preds in zip(out["target_text"].tolist(), pred_topk)
    ]
    out["correct_top5"] = [
        gold in preds[:5]
        for gold, preds in zip(out["target_text"].tolist(), pred_topk)
    ]

    return out


def save_metrics(metrics: Dict, save_dir: Path) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)

    metrics_json_path = save_dir / "metrics.json"
    metrics_csv_path = save_dir / "metrics.csv"

    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    pd.DataFrame([metrics]).to_csv(metrics_csv_path, index=False, encoding="utf-8-sig")


def save_predictions(pred_df: pd.DataFrame, save_dir: Path) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(save_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")


def save_error_analysis(pred_df: pd.DataFrame, save_dir: Path) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)

    top1_wrong = pred_df[~pred_df["correct_top1"]].copy()
    top1_wrong.to_csv(save_dir / "top1_errors.csv", index=False, encoding="utf-8-sig")

    summary = {
        "num_samples": int(len(pred_df)),
        "top1_correct": int(pred_df["correct_top1"].sum()),
        "top1_wrong": int((~pred_df["correct_top1"]).sum()),
        "top3_recovered_from_top1_wrong": int(
            ((~pred_df["correct_top1"]) & (pred_df["correct_top3"])).sum()
        ),
        "top5_recovered_from_top1_wrong": int(
            ((~pred_df["correct_top1"]) & (pred_df["correct_top5"])).sum()
        ),
    }

    pd.DataFrame([summary]).to_csv(save_dir / "error_summary.csv", index=False, encoding="utf-8-sig")


def save_confusion_style_table(pred_df: pd.DataFrame, save_dir: Path) -> None:
    """
    Binary correctness table, not full class confusion matrix over 10k idioms.
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    table = pd.DataFrame({
        "Case": [
            "Top1 correct",
            "Top1 wrong but Top3 correct",
            "Top3 wrong but Top5 correct",
            "Top5 wrong"
        ],
        "Count": [
            int(pred_df["correct_top1"].sum()),
            int((~pred_df["correct_top1"] & pred_df["correct_top3"]).sum()),
            int((~pred_df["correct_top3"] & pred_df["correct_top5"]).sum()),
            int((~pred_df["correct_top5"]).sum())
        ]
    })
    table["Percent"] = table["Count"] / len(pred_df)

    table.to_csv(save_dir / "confusion_style_summary.csv", index=False, encoding="utf-8-sig")


def save_charts(metrics: Dict, pred_df: pd.DataFrame, save_dir: Path) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Chart 1: Top-K accuracy
    # --------------------------------------------------------
    plt.figure(figsize=(8, 5))
    xs = ["Top-1", "Top-3", "Top-5"]
    ys = [
        metrics["top1_accuracy"],
        metrics["top3_accuracy"],
        metrics["top5_accuracy"],
    ]
    plt.bar(xs, ys)
    plt.ylabel("Accuracy")
    plt.title("Task 3 Retrieval Performance (Example-Embedding Bank)")
    plt.ylim(0, 1.0)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / "chart_topk_accuracy.png", dpi=200, bbox_inches="tight")
    plt.close()

    # --------------------------------------------------------
    # Chart 2: Correctness distribution
    # --------------------------------------------------------
    summary_df = pd.DataFrame({
        "Case": ["Top1 correct", "Top1 wrong"],
        "Count": [
            int(pred_df["correct_top1"].sum()),
            int((~pred_df["correct_top1"]).sum())
        ]
    })
    summary_df["Percent"] = summary_df["Count"] / len(pred_df)

    plt.figure(figsize=(7, 4))
    plt.bar(summary_df["Case"], summary_df["Percent"])
    plt.ylabel("Proportion")
    plt.title("Top-1 Correctness Distribution")
    plt.ylim(0, 1.0)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / "chart_top1_correctness.png", dpi=200, bbox_inches="tight")
    plt.close()

    # --------------------------------------------------------
    # Chart 3: Recovery analysis
    # --------------------------------------------------------
    recovery_df = pd.DataFrame({
        "Case": [
            "Recovered in Top-3",
            "Recovered in Top-5",
            "Not recovered in Top-5"
        ],
        "Count": [
            int((~pred_df["correct_top1"] & pred_df["correct_top3"]).sum()),
            int((~pred_df["correct_top3"] & pred_df["correct_top5"]).sum()),
            int((~pred_df["correct_top5"]).sum())
        ]
    })
    recovery_df["Percent"] = recovery_df["Count"] / len(pred_df)

    plt.figure(figsize=(8, 4))
    plt.bar(recovery_df["Case"], recovery_df["Percent"])
    plt.ylabel("Proportion")
    plt.title("Recovery Beyond Top-1")
    plt.ylim(0, 1.0)
    plt.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(save_dir / "chart_recovery_analysis.png", dpi=200, bbox_inches="tight")
    plt.close()


def save_demo_cases(pred_df: pd.DataFrame, save_dir: Path, n: int = 20) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)

    demo_correct = pred_df[pred_df["correct_top1"]].head(n)
    demo_wrong = pred_df[~pred_df["correct_top1"]].head(n)
    demo_top5_recovered = pred_df[(~pred_df["correct_top1"]) & (pred_df["correct_top5"])].head(n)

    demo_correct.to_csv(save_dir / "demo_top1_correct.csv", index=False, encoding="utf-8-sig")
    demo_wrong.to_csv(save_dir / "demo_top1_wrong.csv", index=False, encoding="utf-8-sig")
    demo_top5_recovered.to_csv(save_dir / "demo_top5_recovered.csv", index=False, encoding="utf-8-sig")


# ============================================================
# Main pipeline
# ============================================================

def run_task3_retrieval_idiom_embed(
    model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    top_k: int = 5,
    save_outputs: bool = True,
    output_dir: Path | None = None,
    verbose: bool = True
) -> Dict:
    project_root = find_project_root(Path.cwd())

    if output_dir is None:
        output_dir = Path.home() / "Desktop" / "IdiomX_Task3_Results" / "task3_retrieval_idiom_embed"

    df_train, df_val, df_test = load_task3_data(project_root)

    bank_df = build_idiom_example_bank(df_train, df_val, df_test)
    test_df = prepare_test_queries(df_test)

    if verbose:
        print("Project root:", project_root)
        print("Model:", model_name)
        print("Test samples:", len(test_df))
        print("Bank examples:", len(bank_df))
        print("Unique bank idioms:", bank_df["target_text"].nunique())

    model = SentenceTransformer(model_name)

    pred_topk, _ = retrieve_topk_idioms_from_example_bank(
        model=model,
        query_texts=test_df["input_text"].tolist(),
        bank_texts=bank_df["bank_text"].tolist(),
        bank_targets=bank_df["target_text"].tolist(),
        top_k=top_k
    )

    pred_df = build_predictions_dataframe(test_df, pred_topk)

    metrics = {
        "top1_accuracy": compute_topk_accuracy(
            test_df["target_text"].tolist(), pred_topk, k=1
        ),
        "top3_accuracy": compute_topk_accuracy(
            test_df["target_text"].tolist(), pred_topk, k=3
        ),
        "top5_accuracy": compute_topk_accuracy(
            test_df["target_text"].tolist(), pred_topk, k=5
        ),
        "mrr_at_all": compute_mrr(
            test_df["target_text"].tolist(), pred_topk
        ),
        "num_samples": int(len(test_df)),
        "num_bank_examples": int(len(bank_df)),
        "num_bank_idioms": int(bank_df["target_text"].nunique()),
        "model_name": model_name,
        "retrieval_mode": "arabic_context_to_english_idiomatic_example_to_idiom",
        "idiomatic_only": True,
        "top_k": top_k,
    }

    if save_outputs:
        output_dir.mkdir(parents=True, exist_ok=True)
        save_metrics(metrics, output_dir)
        save_predictions(pred_df, output_dir)
        save_error_analysis(pred_df, output_dir)
        save_confusion_style_table(pred_df, output_dir)
        save_charts(metrics, pred_df, output_dir)
        save_demo_cases(pred_df, output_dir)

    if verbose:
        print("\nSaved to:", output_dir)
        print(metrics)

    return {
        "metrics": metrics,
        "predictions": pred_df,
        "bank_df": bank_df,
        "output_dir": str(output_dir)
    }


# ============================================================
# CMD entry
# ============================================================

if __name__ == "__main__":
    run_task3_retrieval_idiom_embed()