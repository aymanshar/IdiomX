from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sentence_transformers import InputExample, SentenceTransformer, losses, util
from torch.utils.data import DataLoader


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
    model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

    train_csv: Path = PROJECT_ROOT / "deep_learning" / "datasets" / "arabic_context_to_idiom" / "train.csv"
    val_csv: Path = PROJECT_ROOT / "deep_learning" / "datasets" / "arabic_context_to_idiom" / "validation.csv"
    test_csv: Path = PROJECT_ROOT / "deep_learning" / "datasets" / "arabic_context_to_idiom" / "test.csv"

    output_dir: Path = PROJECT_ROOT / "deep_learning" / "models" / "task3_arabic_bank_finetuned_retrieval"
    results_dir: Path = Path.home() / "Desktop" / "IdiomX_Task3_Results" / "task3_arabic_bank_finetuned_retrieval"

    input_col: str = "input_text"
    target_col: str = "target_text"
    idiom_flag_col: str = "is_example_idiom"

    # Arabic-side columns to use for the semantic bank.
    # Missing columns will be skipped automatically.
    arabic_bank_columns: Tuple[str, ...] = (
        "input_text",
        "meaning_ar",
        "idiom_canonical_ar",
        "idiom_in_example_ar",
        "idiom_canonical_meaning_ar",
        "idiom_in_example_meaning_ar",
    )

    idiomatic_only: bool = True

    train_batch_size: int = 32
    num_epochs: int = 2
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1

    top_k: int = 5
    seed: int = 42

    # how many bank positives per query to sample during training
    positives_per_query: int = 3

    # strict evaluation: bank built from train+validation only
    include_test_in_bank: bool = False


CFG = Config()


# =========================================================
# UTILS
# =========================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dirs() -> None:
    CFG.output_dir.mkdir(parents=True, exist_ok=True)
    CFG.results_dir.mkdir(parents=True, exist_ok=True)


def normalize_text(x: Any) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip().lower()


def exact_match(a: str, b: str) -> bool:
    return normalize_text(a) == normalize_text(b)


def token_overlap_score(a: str, b: str) -> float:
    a_tokens = set(normalize_text(a).split())
    b_tokens = set(normalize_text(b).split())
    if not a_tokens and not b_tokens:
        return 1.0
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


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
        for col in set(CFG.arabic_bank_columns + (CFG.input_col, CFG.target_col)):
            if col in df.columns:
                df[col] = df[col].fillna("").astype(str)

    print("Train shape:", df_train.shape)
    print("Validation shape:", df_val.shape)
    print("Test shape:", df_test.shape)

    return df_train, df_val, df_test


# =========================================================
# BUILD ARABIC SEMANTIC BANK
# =========================================================
def build_arabic_bank(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
) -> pd.DataFrame:
    parts = [df_train, df_val]
    if CFG.include_test_in_bank:
        parts.append(df_test)

    source_df = pd.concat(parts, axis=0, ignore_index=True)

    bank_rows: List[Dict[str, str]] = []

    available_cols = [c for c in CFG.arabic_bank_columns if c in source_df.columns]
    print("Arabic bank columns used:", available_cols)

    for _, row in source_df.iterrows():
        target_idiom = str(row[CFG.target_col]).strip()
        if not target_idiom:
            continue

        for col in available_cols:
            text_val = str(row[col]).strip()
            if not text_val:
                continue

            bank_rows.append(
                {
                    "bank_text_ar": text_val,
                    "bank_text_type": col,
                    "english_idiom": target_idiom,
                }
            )

    bank_df = pd.DataFrame(bank_rows).drop_duplicates().reset_index(drop=True)

    print("Arabic bank rows:", len(bank_df))
    print("Unique English idioms in bank:", bank_df["english_idiom"].nunique())

    return bank_df


# =========================================================
# BUILD TRAINING PAIRS
# =========================================================
def build_positive_pairs(df_train: pd.DataFrame, bank_df: pd.DataFrame) -> List[InputExample]:
    bank_grouped = (
        bank_df.groupby("english_idiom")["bank_text_ar"]
        .apply(list)
        .to_dict()
    )

    examples: List[InputExample] = []

    for _, row in df_train.iterrows():
        query = str(row[CFG.input_col]).strip()
        gold_idiom = str(row[CFG.target_col]).strip()

        if not query or not gold_idiom:
            continue

        positive_texts = bank_grouped.get(gold_idiom, [])
        positive_texts = [x for x in positive_texts if normalize_text(x) != normalize_text(query)]

        if not positive_texts:
            continue

        sampled = positive_texts[:CFG.positives_per_query]
        if len(positive_texts) > CFG.positives_per_query:
            sampled = random.sample(positive_texts, CFG.positives_per_query)

        for pos_text in sampled:
            examples.append(InputExample(texts=[query, pos_text]))

    print("Training pairs built:", len(examples))
    return examples


# =========================================================
# TRAIN RETRIEVER
# =========================================================
def train_retriever(train_examples: List[InputExample]) -> SentenceTransformer:
    model = SentenceTransformer(CFG.model_name)

    train_loader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=CFG.train_batch_size,
    )

    train_loss = losses.MultipleNegativesRankingLoss(model)

    warmup_steps = int(len(train_loader) * CFG.num_epochs * CFG.warmup_ratio)

    print("Starting fine-tuning...")
    print("Warmup steps:", warmup_steps)

    model.fit(
        train_objectives=[(train_loader, train_loss)],
        epochs=CFG.num_epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": CFG.learning_rate},
        output_path=str(CFG.output_dir),
        show_progress_bar=True,
    )

    print("Fine-tuned model saved to:", CFG.output_dir)
    return model


# =========================================================
# RETRIEVAL EVAL
# =========================================================
def encode_bank(bank_df: pd.DataFrame, model: SentenceTransformer) -> np.ndarray:
    texts = bank_df["bank_text_ar"].tolist()
    emb = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    return emb


def rank_idioms_for_query(
    query_text: str,
    bank_df: pd.DataFrame,
    bank_embeddings: np.ndarray,
    model: SentenceTransformer,
    top_k: int = 5,
) -> Tuple[List[str], List[float], pd.DataFrame]:
    query_emb = model.encode([query_text], convert_to_numpy=True)
    sims = util.cos_sim(query_emb, bank_embeddings)[0].cpu().numpy() if hasattr(util.cos_sim(query_emb, bank_embeddings)[0], "cpu") else cosine_sim_numpy(query_emb[0], bank_embeddings)

    scored = bank_df.copy()
    scored["similarity"] = sims

    # aggregate by idiom using max score over all Arabic bank entries
    idiom_scores = (
        scored.groupby("english_idiom")["similarity"]
        .max()
        .sort_values(ascending=False)
        .reset_index()
    )

    top_idioms = idiom_scores["english_idiom"].head(top_k).tolist()
    top_scores = idiom_scores["similarity"].head(top_k).tolist()

    return top_idioms, top_scores, idiom_scores


def cosine_sim_numpy(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    q = query_vec / (np.linalg.norm(query_vec) + 1e-12)
    m = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12)
    return m @ q


def evaluate_test(
    df_test: pd.DataFrame,
    bank_df: pd.DataFrame,
    bank_embeddings: np.ndarray,
    model: SentenceTransformer,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    total = len(df_test)

    for i, (_, row) in enumerate(df_test.iterrows(), start=1):
        query = str(row[CFG.input_col]).strip()
        gold = str(row[CFG.target_col]).strip()

        top_idioms, top_scores, idiom_scores_df = rank_idioms_for_query(
            query_text=query,
            bank_df=bank_df,
            bank_embeddings=bank_embeddings,
            model=model,
            top_k=CFG.top_k,
        )

        rows.append(
            {
                "input_text": query,
                "gold_idiom": gold,
                "pred_top1": top_idioms[0] if len(top_idioms) > 0 else "",
                "pred_top3": top_idioms[:3],
                "pred_top5": top_idioms[:5],
                "top_scores": top_scores,
                "top1_correct": int(exact_match(top_idioms[0], gold)) if len(top_idioms) > 0 else 0,
                "top3_correct": int(any(exact_match(x, gold) for x in top_idioms[:3])),
                "top5_correct": int(any(exact_match(x, gold) for x in top_idioms[:5])),
                "mrr_at_5": next((1.0 / (idx + 1) for idx, x in enumerate(top_idioms[:5]) if exact_match(x, gold)), 0.0),
            }
        )

        if i % 200 == 0 or i == total:
            print(f"{i}/{total} samples processed...")

    return pd.DataFrame(rows)


def compute_metrics(pred_df: pd.DataFrame) -> Dict[str, Any]:
    metrics = {
        "top1_accuracy": float(pred_df["top1_correct"].mean()),
        "top3_accuracy": float(pred_df["top3_correct"].mean()),
        "top5_accuracy": float(pred_df["top5_correct"].mean()),
        "mrr_at_5": float(pred_df["mrr_at_5"].mean()),
        "num_test_samples": int(len(pred_df)),
        "model_name": CFG.model_name,
        "idiomatic_only": bool(CFG.idiomatic_only),
        "include_test_in_bank": bool(CFG.include_test_in_bank),
    }
    return metrics


# =========================================================
# REPORT
# =========================================================
def save_report(metrics: Dict[str, Any], pred_df: pd.DataFrame, out_path: Path) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Task 3 - Arabic Bank Fine-Tuned Retrieval\n")
        f.write("=" * 70 + "\n\n")

        f.write("Metrics\n")
        f.write("-" * 30 + "\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

        f.write("\nCorrect top-1 examples\n")
        f.write("-" * 30 + "\n")
        for _, row in pred_df[pred_df["top1_correct"] == 1].head(15).iterrows():
            f.write(f"Arabic input: {row['input_text']}\n")
            f.write(f"Gold idiom: {row['gold_idiom']}\n")
            f.write(f"Pred top1 : {row['pred_top1']}\n")
            f.write(f"Pred top3 : {row['pred_top3']}\n\n")

        f.write("\nTop-5 hit but top-1 miss examples\n")
        f.write("-" * 30 + "\n")
        subset = pred_df[(pred_df["top1_correct"] == 0) & (pred_df["top5_correct"] == 1)]
        for _, row in subset.head(15).iterrows():
            f.write(f"Arabic input: {row['input_text']}\n")
            f.write(f"Gold idiom: {row['gold_idiom']}\n")
            f.write(f"Pred top5 : {row['pred_top5']}\n\n")

        f.write("\nHard wrong cases\n")
        f.write("-" * 30 + "\n")
        subset = pred_df[pred_df["top5_correct"] == 0]
        for _, row in subset.head(15).iterrows():
            f.write(f"Arabic input: {row['input_text']}\n")
            f.write(f"Gold idiom: {row['gold_idiom']}\n")
            f.write(f"Pred top5 : {row['pred_top5']}\n\n")


# =========================================================
# MAIN
# =========================================================
def run_task3_arabic_bank_finetuned_retrieval() -> Dict[str, Any]:
    set_seed(CFG.seed)
    ensure_dirs()

    print("Loading data...")
    df_train, df_val, df_test = load_splits()

    print("\nBuilding Arabic semantic bank...")
    bank_df = build_arabic_bank(df_train, df_val, df_test)

    print("\nBuilding positive training pairs...")
    train_examples = build_positive_pairs(df_train, bank_df)

    print("\nTraining fine-tuned retriever...")
    model = train_retriever(train_examples)

    print("\nEncoding Arabic semantic bank...")
    bank_embeddings = encode_bank(bank_df, model)

    print("\nEvaluating on test set...")
    pred_df = evaluate_test(df_test, bank_df, bank_embeddings, model)

    metrics = compute_metrics(pred_df)

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
        title="Task 3 Arabic Bank Fine-Tuned Retrieval",
        out_path=CFG.results_dir / "task3_arabic_bank_finetuned_metrics.png",
    )

    print("\nSaved outputs to:")
    print(CFG.output_dir)
    print(CFG.results_dir)

    print("\nFinal metrics:")
    print(metrics)

    return {
        "metrics": metrics,
        "predictions": pred_df,
        "bank": bank_df,
    }


if __name__ == "__main__":
    run_task3_arabic_bank_finetuned_retrieval()