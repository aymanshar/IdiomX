from pathlib import Path
import json
import math
import re
from typing import List, Dict

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt


# ============================================================
# Project paths
# ============================================================

def find_project_root(start_path: Path, target_folder: str = "deep_learning") -> Path:
    for parent in [start_path.resolve()] + list(start_path.resolve().parents):
        if (parent / target_folder).exists():
            return parent
    raise RuntimeError("Project root not found")


PROJECT_ROOT = find_project_root(Path.cwd())

DATA_DIR = PROJECT_ROOT / "datasets" / "arabic_context_to_idiom"
TRAIN_PATH = DATA_DIR / "train.csv"
VAL_PATH = DATA_DIR / "validation.csv"
TEST_PATH = DATA_DIR / "test.csv"

OUTPUT_DIR = Path.home() / "Desktop" / "IdiomX_Task3_Results" / "task3_retrieval_e5"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Config
# ============================================================

MODEL_NAME = "intfloat/multilingual-e5-base"
TOP_K = 5
IDIOMATIC_ONLY = True

# Retrieval mode:
# "arabic_context_to_english_idiomatic_example"
RETRIEVAL_MODE = "arabic_context_to_english_idiomatic_example"


# ============================================================
# Helpers
# ============================================================

def normalize_text(x: str) -> str:
    if pd.isna(x):
        return ""
    x = str(x).strip().lower()
    x = re.sub(r"\s+", " ", x)
    return x


def compute_topk_metrics(gold_list: List[str], ranked_lists: List[List[str]], ks=(1, 3, 5)) -> Dict[str, float]:
    metrics = {}
    n = len(gold_list)

    for k in ks:
        correct = 0
        for gold, ranked in zip(gold_list, ranked_lists):
            topk = ranked[:k]
            if gold in topk:
                correct += 1
        metrics[f"top{k}_accuracy"] = correct / n if n > 0 else 0.0

    return metrics


def compute_mrr(gold_list: List[str], ranked_lists: List[List[str]]) -> float:
    rr_sum = 0.0
    n = len(gold_list)

    for gold, ranked in zip(gold_list, ranked_lists):
        rank = 0
        for idx, pred in enumerate(ranked, start=1):
            if pred == gold:
                rank = idx
                break
        if rank > 0:
            rr_sum += 1.0 / rank

    return rr_sum / n if n > 0 else 0.0


def plot_topk(metrics: Dict[str, float], save_path: Path) -> None:
    k_labels = ["Top-1", "Top-3", "Top-5"]
    values = [
        metrics["top1_accuracy"],
        metrics["top3_accuracy"],
        metrics["top5_accuracy"],
    ]

    plt.figure(figsize=(8, 5))
    plt.bar(k_labels, values)
    plt.title("Task 3 Retrieval Performance (E5)")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.0)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_confusion_like_table(df_preds: pd.DataFrame, save_path: Path) -> None:
    summary = {
        "total_samples": len(df_preds),
        "top1_correct": int((df_preds["gold_norm"] == df_preds["pred_top1_norm"]).sum()),
        "top1_wrong": int((df_preds["gold_norm"] != df_preds["pred_top1_norm"]).sum()),
        "top3_hit": int(df_preds["gold_in_top3"].sum()),
        "top5_hit": int(df_preds["gold_in_top5"].sum()),
    }
    pd.DataFrame([summary]).to_csv(save_path, index=False)


# ============================================================
# Load data
# ============================================================

df_train = pd.read_csv(TRAIN_PATH)
df_val = pd.read_csv(VAL_PATH)
df_test = pd.read_csv(TEST_PATH)

if IDIOMATIC_ONLY:
    df_train = df_train[df_train["is_example_idiom"] == True].copy()
    df_val = df_val[df_val["is_example_idiom"] == True].copy()
    df_test = df_test[df_test["is_example_idiom"] == True].copy()

print(f"Project root: {PROJECT_ROOT}")
print(f"Model: {MODEL_NAME}")
print(f"Train shape: {df_train.shape}")
print(f"Validation shape: {df_val.shape}")
print(f"Test shape: {df_test.shape}")


# ============================================================
# Build retrieval bank
# ============================================================

# We use Arabic idiomatic examples as passages,
# but rank by their linked English canonical idiom target_text.
bank_df = pd.concat([df_train, df_val], axis=0, ignore_index=True).copy()

bank_df["input_text"] = bank_df["input_text"].astype(str)
bank_df["target_text"] = bank_df["target_text"].astype(str)

bank_df["bank_text"] = bank_df["input_text"].astype(str)
bank_df["gold_norm"] = bank_df["target_text"].apply(normalize_text)

# Keep all example passages, because contextual examples are the key
bank_df = bank_df.drop_duplicates(subset=["bank_text", "target_text"]).reset_index(drop=True)

unique_bank_idioms = sorted(bank_df["gold_norm"].unique())

print(f"Bank examples: {len(bank_df)}")
print(f"Unique bank idioms: {len(unique_bank_idioms)}")


# ============================================================
# Prepare test queries
# ============================================================

test_df = df_test.copy()
test_df["input_text"] = test_df["input_text"].astype(str)
test_df["target_text"] = test_df["target_text"].astype(str)
test_df["gold_norm"] = test_df["target_text"].apply(normalize_text)

print(f"Test samples: {len(test_df)}")


# ============================================================
# E5 formatting
# ============================================================

# VERY IMPORTANT:
# query -> "query: ..."
# passage -> "passage: ..."

bank_texts = ["passage: " + t for t in bank_df["bank_text"].tolist()]
query_texts = ["query: " + t for t in test_df["input_text"].tolist()]


# ============================================================
# Load model and encode
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer(MODEL_NAME, device=device)

bank_embeddings = model.encode(
    bank_texts,
    batch_size=64,
    convert_to_tensor=True,
    show_progress_bar=True,
    normalize_embeddings=True,
)

query_embeddings = model.encode(
    query_texts,
    batch_size=64,
    convert_to_tensor=True,
    show_progress_bar=True,
    normalize_embeddings=True,
)


# ============================================================
# Retrieval
# ============================================================

all_ranked_lists = []
prediction_rows = []

scores = util.cos_sim(query_embeddings, bank_embeddings)

for i in range(len(test_df)):
    score_row = scores[i]
    top_values, top_indices = torch.topk(score_row, k=TOP_K)

    ranked_idioms = []
    ranked_examples = []

    seen = set()

    for idx_tensor, score_tensor in zip(top_indices, top_values):
        idx = int(idx_tensor.item())
        score = float(score_tensor.item())

        idiom_norm = bank_df.iloc[idx]["gold_norm"]
        bank_text = bank_df.iloc[idx]["bank_text"]

        ranked_examples.append({
            "retrieved_example": bank_text,
            "retrieved_idiom": idiom_norm,
            "score": score,
        })

        if idiom_norm not in seen:
            ranked_idioms.append(idiom_norm)
            seen.add(idiom_norm)

    gold_norm = test_df.iloc[i]["gold_norm"]

    pred_top1 = ranked_idioms[0] if len(ranked_idioms) > 0 else ""
    pred_top3 = ranked_idioms[:3]
    pred_top5 = ranked_idioms[:5]

    all_ranked_lists.append(ranked_idioms)

    prediction_rows.append({
        "input_text": test_df.iloc[i]["input_text"],
        "gold_idiom": test_df.iloc[i]["target_text"],
        "gold_norm": gold_norm,
        "pred_top1": pred_top1,
        "pred_top1_norm": normalize_text(pred_top1),
        "pred_top3": " ||| ".join(pred_top3),
        "pred_top5": " ||| ".join(pred_top5),
        "gold_in_top3": gold_norm in pred_top3,
        "gold_in_top5": gold_norm in pred_top5,
        "top_examples_json": json.dumps(ranked_examples, ensure_ascii=False),
    })


pred_df = pd.DataFrame(prediction_rows)


# ============================================================
# Metrics
# ============================================================

gold_list = test_df["gold_norm"].tolist()

metrics = compute_topk_metrics(gold_list, all_ranked_lists, ks=(1, 3, 5))
metrics["mrr_at_all"] = compute_mrr(gold_list, all_ranked_lists)
metrics["num_samples"] = len(test_df)
metrics["num_bank_examples"] = len(bank_df)
metrics["num_bank_idioms"] = len(unique_bank_idioms)
metrics["model_name"] = MODEL_NAME
metrics["retrieval_mode"] = RETRIEVAL_MODE
metrics["idiomatic_only"] = IDIOMATIC_ONLY
metrics["top_k"] = TOP_K


# ============================================================
# Save outputs
# ============================================================

pred_path = OUTPUT_DIR / "test_predictions.csv"
metrics_path = OUTPUT_DIR / "metrics.json"
metrics_csv_path = OUTPUT_DIR / "metrics.csv"
chart_path = OUTPUT_DIR / "topk_chart.png"
summary_path = OUTPUT_DIR / "truth_table_summary.csv"

pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")

with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)

pd.DataFrame([metrics]).to_csv(metrics_csv_path, index=False, encoding="utf-8-sig")

plot_topk(metrics, chart_path)
save_confusion_like_table(pred_df, summary_path)


# ============================================================
# Console output
# ============================================================

print(f"\nSaved to: {OUTPUT_DIR}")
print(metrics)