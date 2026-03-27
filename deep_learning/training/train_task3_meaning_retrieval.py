from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, MT5ForConditionalGeneration


# =========================================================
# PROJECT ROOT
# =========================================================
def find_project_root(start_path: Path, target_folder: str = "deep_learning") -> Path:
    start_path = start_path.resolve()
    for candidate in [start_path] + list(start_path.parents):
        if (candidate / target_folder).exists():
            return candidate
    raise RuntimeError(f"Project root not found from: {start_path}")


PROJECT_ROOT = find_project_root(Path.cwd())


# =========================================================
# CONFIG
# =========================================================
@dataclass
class Config:
    # Meaning generator checkpoint
    # You can start with your trained mT5-base checkpoint
    meaning_model_dir: Path = (
        PROJECT_ROOT
        / "deep_learning"
        / "models"
        / "arabic_context_to_idiom_mt5_base_normalized"
        / "checkpoint-2108"
    )

    train_csv: Path = PROJECT_ROOT / "deep_learning" / "datasets" / "arabic_context_to_idiom" / "train.csv"
    val_csv: Path = PROJECT_ROOT / "deep_learning" / "datasets" / "arabic_context_to_idiom" / "validation.csv"
    test_csv: Path = PROJECT_ROOT / "deep_learning" / "datasets" / "arabic_context_to_idiom" / "test.csv"

    output_dir: Path = PROJECT_ROOT / "deep_learning" / "models" / "task3_meaning_retrieval"
    results_dir: Path = Path.home() / "Desktop" / "IdiomX_Task3_Results" / "task3_meaning_retrieval"

    input_col: str = "input_text"
    target_col: str = "target_text"
    idiom_flag_col: str = "is_example_idiom"

    # Meaning columns in your dataset
    meaning_en_col: str = "meaning_en"
    meaning_ar_col: str = "meaning_ar"

    # Meaning generator prompt
    meaning_prefix: str = "Explain the meaning of this Arabic sentence in simple English: "

    max_source_length: int = 128
    max_meaning_length: int = 48

    # Retrieval encoder
    retrieval_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    top_k: int = 5
    idiomatic_only: bool = True

    save_top_k_predictions: bool = True


CFG = Config()


# =========================================================
# UTILITIES
# =========================================================
def ensure_dirs() -> None:
    CFG.output_dir.mkdir(parents=True, exist_ok=True)
    CFG.results_dir.mkdir(parents=True, exist_ok=True)


def normalize_text(x: Any) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip().lower()


def token_overlap_score(a: str, b: str) -> float:
    a_tokens = set(normalize_text(a).split())
    b_tokens = set(normalize_text(b).split())
    if not a_tokens and not b_tokens:
        return 1.0
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


def exact_match(pred: str, gold: str) -> bool:
    return normalize_text(pred) == normalize_text(gold)


def topk_hit(pred_list: List[str], gold: str, k: int) -> bool:
    gold_norm = normalize_text(gold)
    return gold_norm in [normalize_text(x) for x in pred_list[:k]]


def mrr_at_k(pred_list: List[str], gold: str, k: int) -> float:
    gold_norm = normalize_text(gold)
    for i, pred in enumerate(pred_list[:k], start=1):
        if normalize_text(pred) == gold_norm:
            return 1.0 / i
    return 0.0


def save_bar_chart(df: pd.DataFrame, x_col: str, y_col: str, title: str, out_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.bar(df[x_col], df[y_col])
    plt.title(title)
    plt.ylabel(y_col)
    plt.xticks(rotation=20)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# =========================================================
# LOAD DATA
# =========================================================
def load_splits() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_train = pd.read_csv(CFG.train_csv)
    df_val = pd.read_csv(CFG.val_csv)
    df_test = pd.read_csv(CFG.test_csv)

    if CFG.idiomatic_only:
        df_train = df_train[df_train[CFG.idiom_flag_col] == 1].copy()
        df_val = df_val[df_val[CFG.idiom_flag_col] == 1].copy()
        df_test = df_test[df_test[CFG.idiom_flag_col] == 1].copy()

    for df in (df_train, df_val, df_test):
        df[CFG.input_col] = df[CFG.input_col].astype(str)
        df[CFG.target_col] = df[CFG.target_col].astype(str)
        if CFG.meaning_en_col in df.columns:
            df[CFG.meaning_en_col] = df[CFG.meaning_en_col].fillna("").astype(str)
        if CFG.meaning_ar_col in df.columns:
            df[CFG.meaning_ar_col] = df[CFG.meaning_ar_col].fillna("").astype(str)

    print("Train shape:", df_train.shape)
    print("Validation shape:", df_val.shape)
    print("Test shape:", df_test.shape)

    return df_train, df_val, df_test


# =========================================================
# BUILD MEANING BANK
# =========================================================
def build_meaning_bank(df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame) -> pd.DataFrame:
    all_df = pd.concat([df_train, df_val, df_test], axis=0, ignore_index=True)

    required_cols = [CFG.target_col, CFG.meaning_en_col]
    for col in required_cols:
        if col not in all_df.columns:
            raise ValueError(f"Column '{col}' not found in dataset")

    bank = (
        all_df[[CFG.target_col, CFG.meaning_en_col]]
        .dropna()
        .copy()
    )

    bank[CFG.target_col] = bank[CFG.target_col].astype(str).str.strip()
    bank[CFG.meaning_en_col] = bank[CFG.meaning_en_col].astype(str).str.strip()

    bank = bank[
        (bank[CFG.target_col] != "") &
        (bank[CFG.meaning_en_col] != "")
    ].copy()

    # keep unique idiom-meaning pairs
    bank = bank.drop_duplicates(subset=[CFG.target_col, CFG.meaning_en_col]).reset_index(drop=True)

    print("Meaning bank rows:", len(bank))
    print("Unique bank idioms:", bank[CFG.target_col].nunique())

    return bank


# =========================================================
# LOAD MODELS
# =========================================================
def load_meaning_generator():
    model_path = str(CFG.meaning_model_dir.resolve())
    print("Meaning model path:", model_path)
    print("Exists:", CFG.meaning_model_dir.exists())

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, local_files_only=True)
    model = MT5ForConditionalGeneration.from_pretrained(model_path, local_files_only=True)

    if torch.cuda.is_available():
        model.to("cuda")

    model.eval()
    return tokenizer, model


def load_retriever():
    print("Loading retrieval model:", CFG.retrieval_model_name)
    model = SentenceTransformer(CFG.retrieval_model_name)
    return model


# =========================================================
# GENERATE ENGLISH MEANING FROM ARABIC SENTENCE
# =========================================================
def generate_meaning(
    model: MT5ForConditionalGeneration,
    tokenizer,
    arabic_text: str,
) -> str:
    device = next(model.parameters()).device
    prompt = CFG.meaning_prefix + str(arabic_text)

    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=CFG.max_source_length,
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        output = model.generate(
            **encoded,
            max_length=CFG.max_meaning_length,
            num_beams=5,
            early_stopping=True,
        )

    return tokenizer.decode(output[0], skip_special_tokens=True).strip()


# =========================================================
# RETRIEVE IDIOM BY GENERATED MEANING
# =========================================================
def build_bank_embeddings(bank_df: pd.DataFrame, retriever: SentenceTransformer) -> np.ndarray:
    meanings = bank_df[CFG.meaning_en_col].tolist()
    emb = retriever.encode(meanings, show_progress_bar=True, convert_to_numpy=True)
    return emb


def retrieve_top_k(
    query_meaning: str,
    bank_df: pd.DataFrame,
    bank_embeddings: np.ndarray,
    retriever: SentenceTransformer,
    top_k: int = 5,
) -> Tuple[List[str], List[str], List[float]]:
    query_emb = retriever.encode([query_meaning], convert_to_numpy=True)
    sims = cosine_similarity(query_emb, bank_embeddings)[0]

    top_idx = np.argsort(-sims)[:top_k]

    idioms = bank_df.iloc[top_idx][CFG.target_col].tolist()
    meanings = bank_df.iloc[top_idx][CFG.meaning_en_col].tolist()
    scores = sims[top_idx].tolist()

    return idioms, meanings, scores


# =========================================================
# EVALUATION
# =========================================================
def evaluate_predictions(pred_df: pd.DataFrame) -> Dict[str, Any]:
    top1_acc = float(pred_df["top1_correct"].mean())
    top3_acc = float(pred_df["top3_correct"].mean())
    top5_acc = float(pred_df["top5_correct"].mean())
    mrr = float(pred_df["mrr_at_5"].mean())
    avg_meaning_overlap = float(pred_df["meaning_token_overlap"].mean())

    metrics = {
        "top1_accuracy": top1_acc,
        "top3_accuracy": top3_acc,
        "top5_accuracy": top5_acc,
        "mrr_at_5": mrr,
        "avg_meaning_token_overlap": avg_meaning_overlap,
        "num_test_samples": int(len(pred_df)),
        "meaning_model_dir": str(CFG.meaning_model_dir),
        "retrieval_model_name": CFG.retrieval_model_name,
        "idiomatic_only": bool(CFG.idiomatic_only),
    }
    return metrics


# =========================================================
# REPORT
# =========================================================
def save_report(metrics: Dict[str, Any], pred_df: pd.DataFrame, out_path: Path) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Task 3 - Meaning-based Retrieval\n")
        f.write("=" * 70 + "\n\n")

        f.write("Metrics\n")
        f.write("-" * 30 + "\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

        f.write("\nCorrect top-1 examples\n")
        f.write("-" * 30 + "\n")
        for _, row in pred_df[pred_df["top1_correct"] == 1].head(15).iterrows():
            f.write(f"Arabic input: {row['input_text']}\n")
            f.write(f"Generated meaning: {row['generated_meaning']}\n")
            f.write(f"Gold idiom: {row['gold_idiom']}\n")
            f.write(f"Top-1 idiom: {row['pred_top1']}\n")
            f.write(f"Top-1 meaning: {row['pred_top1_meaning']}\n")
            f.write(f"Top-1 score: {row['pred_top1_score']:.4f}\n\n")

        f.write("\nTop-5 hit but top-1 miss examples\n")
        f.write("-" * 30 + "\n")
        subset = pred_df[(pred_df["top1_correct"] == 0) & (pred_df["top5_correct"] == 1)]
        for _, row in subset.head(15).iterrows():
            f.write(f"Arabic input: {row['input_text']}\n")
            f.write(f"Generated meaning: {row['generated_meaning']}\n")
            f.write(f"Gold idiom: {row['gold_idiom']}\n")
            f.write(f"Top predictions: {row['top_k_idioms']}\n\n")

        f.write("\nHard wrong cases\n")
        f.write("-" * 30 + "\n")
        subset = pred_df[pred_df["top5_correct"] == 0]
        for _, row in subset.head(15).iterrows():
            f.write(f"Arabic input: {row['input_text']}\n")
            f.write(f"Generated meaning: {row['generated_meaning']}\n")
            f.write(f"Gold idiom: {row['gold_idiom']}\n")
            f.write(f"Gold meaning: {row['gold_meaning_en']}\n")
            f.write(f"Top predictions: {row['top_k_idioms']}\n")
            f.write(f"Top meanings: {row['top_k_meanings']}\n\n")


# =========================================================
# MAIN
# =========================================================
def run_task3_meaning_retrieval() -> Dict[str, Any]:
    ensure_dirs()

    print("Loading data...")
    df_train, df_val, df_test = load_splits()

    bank_df = build_meaning_bank(df_train, df_val, df_test)

    print("\nLoading meaning generator...")
    tokenizer, meaning_model = load_meaning_generator()

    print("\nLoading retriever...")
    retriever = load_retriever()

    print("\nBuilding meaning bank embeddings...")
    bank_embeddings = build_bank_embeddings(bank_df, retriever)

    print("\nRunning meaning retrieval on test set...")
    rows: List[Dict[str, Any]] = []

    total = len(df_test)
    for i, (_, row) in enumerate(df_test.iterrows(), start=1):
        arabic_input = str(row[CFG.input_col])
        gold_idiom = str(row[CFG.target_col])
        gold_meaning_en = str(row.get(CFG.meaning_en_col, ""))

        generated_meaning = generate_meaning(meaning_model, tokenizer, arabic_input)

        pred_idioms, pred_meanings, pred_scores = retrieve_top_k(
            query_meaning=generated_meaning,
            bank_df=bank_df,
            bank_embeddings=bank_embeddings,
            retriever=retriever,
            top_k=CFG.top_k,
        )

        pred_top1 = pred_idioms[0]
        pred_top1_meaning = pred_meanings[0]
        pred_top1_score = pred_scores[0]

        row_dict = {
            "input_text": arabic_input,
            "gold_idiom": gold_idiom,
            "gold_meaning_en": gold_meaning_en,
            "generated_meaning": generated_meaning,
            "pred_top1": pred_top1,
            "pred_top1_meaning": pred_top1_meaning,
            "pred_top1_score": pred_top1_score,
            "top_k_idioms": pred_idioms,
            "top_k_meanings": pred_meanings,
            "top_k_scores": pred_scores,
            "top1_correct": int(topk_hit(pred_idioms, gold_idiom, 1)),
            "top3_correct": int(topk_hit(pred_idioms, gold_idiom, 3)),
            "top5_correct": int(topk_hit(pred_idioms, gold_idiom, 5)),
            "mrr_at_5": mrr_at_k(pred_idioms, gold_idiom, 5),
            "meaning_token_overlap": token_overlap_score(generated_meaning, gold_meaning_en),
        }
        rows.append(row_dict)

        if i % 200 == 0 or i == total:
            print(f"{i}/{total} samples processed...")

    pred_df = pd.DataFrame(rows)
    metrics = evaluate_predictions(pred_df)

    # save
    pred_csv = CFG.output_dir / "test_predictions.csv"
    metrics_json = CFG.output_dir / "metrics.json"
    metrics_csv = CFG.output_dir / "metrics.csv"
    report_txt = CFG.output_dir / "report.txt"

    pred_df.to_csv(pred_csv, index=False, encoding="utf-8-sig")
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    pd.DataFrame([metrics]).to_csv(metrics_csv, index=False)
    save_report(metrics, pred_df, report_txt)

    # desktop copies
    pred_df.to_csv(CFG.results_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")
    with open(CFG.results_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    pd.DataFrame([metrics]).to_csv(CFG.results_dir / "metrics.csv", index=False)
    save_report(metrics, pred_df, CFG.results_dir / "report.txt")

    # charts
    chart_df = pd.DataFrame(
        {
            "Metric": ["Top-1", "Top-3", "Top-5", "MRR@5"],
            "Score": [
                metrics["top1_accuracy"],
                metrics["top3_accuracy"],
                metrics["top5_accuracy"],
                metrics["mrr_at_5"],
            ],
        }
    )
    save_bar_chart(
        chart_df,
        x_col="Metric",
        y_col="Score",
        title="Task 3 Meaning Retrieval Performance",
        out_path=CFG.results_dir / "task3_meaning_retrieval_metrics.png",
    )

    print("\nSaved outputs to:")
    print(CFG.output_dir)
    print(CFG.results_dir)

    print("\nFinal metrics:")
    print(metrics)

    return {
        "metrics": metrics,
        "predictions": pred_df,
    }


if __name__ == "__main__":
    run_task3_meaning_retrieval()