from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report
from transformers import (
    MT5ForConditionalGeneration,
    MT5Tokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)


# =========================================================
# Project root detection
# =========================================================
def find_project_root(start_path: Path, target_folder: str = "deep_learning") -> Path:
    start_path = start_path.resolve()
    for candidate in [start_path] + list(start_path.parents):
        if (candidate / target_folder).exists():
            return candidate
    raise RuntimeError(f"Project root not found from: {start_path}")


PROJECT_ROOT = find_project_root(Path.cwd())
sys.path.append(str(PROJECT_ROOT / "deep_learning" / "training"))


# =========================================================
# Configuration
# =========================================================
@dataclass
class Config:
    model_name: str = "google/mt5-small"

    task_dir: Path = PROJECT_ROOT / "datasets" / "arabic_context_to_idiom"
    train_csv: Path = task_dir / "train.csv"
    val_csv: Path = task_dir / "validation.csv"
    test_csv: Path = task_dir / "test.csv"

    output_dir: Path = PROJECT_ROOT / "models" / "arabic_context_to_idiom_mt5_baseline"
    results_dir: Path = PROJECT_ROOT / "models" / "arabic_context_to_idiom_mt5_baseline" / "IdiomX_Task3_Results" / "task3_mt5_baseline"

    input_col: str = "input_text"
    target_col: str = "target_text"
    idiom_flag_col: str = "is_example_idiom"

    idiomatic_only: bool = True

    task_prefix: str = "Arabic to English idiom: "

    max_source_length: int = 128
    max_target_length: int = 16

    num_train_epochs: int = 3
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    train_batch_size: int = 4
    eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4

    logging_steps: int = 100
    save_total_limit: int = 2
    seed: int = 42

    generation_max_length: int = 16
    num_beams: int = 4

    save_predictions: bool = True


CFG = Config()


# =========================================================
# Utility functions
# =========================================================
def set_seed(seed: int = 42) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dirs() -> None:
    CFG.output_dir.mkdir(parents=True, exist_ok=True)
    CFG.results_dir.mkdir(parents=True, exist_ok=True)


def normalize_text(x: Any) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip().lower()


def fuzzy_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()


def exact_match_accuracy(preds: List[str], labels: List[str]) -> float:
    correct = sum(normalize_text(p) == normalize_text(y) for p, y in zip(preds, labels))
    return correct / len(labels) if labels else 0.0


def fuzzy_match_accuracy(preds: List[str], labels: List[str], threshold: float = 0.85) -> float:
    correct = sum(fuzzy_ratio(p, y) >= threshold for p, y in zip(preds, labels))
    return correct / len(labels) if labels else 0.0


def token_overlap_score(pred: str, gold: str) -> float:
    pred_tokens = set(normalize_text(pred).split())
    gold_tokens = set(normalize_text(gold).split())
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    return len(pred_tokens & gold_tokens) / len(pred_tokens | gold_tokens)


def average_token_overlap(preds: List[str], labels: List[str]) -> float:
    scores = [token_overlap_score(p, y) for p, y in zip(preds, labels)]
    return float(np.mean(scores)) if scores else 0.0


# =========================================================
# Data loading
# =========================================================
def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("Loading datasets...")

    df_train = pd.read_csv(CFG.train_csv)
    df_val = pd.read_csv(CFG.val_csv)
    df_test = pd.read_csv(CFG.test_csv)

    print("Original shapes:")
    print("Train:", df_train.shape)
    print("Validation:", df_val.shape)
    print("Test:", df_test.shape)

    if CFG.idiomatic_only:
        df_train = df_train[df_train[CFG.idiom_flag_col] == 1].copy()
        df_val = df_val[df_val[CFG.idiom_flag_col] == 1].copy()
        df_test = df_test[df_test[CFG.idiom_flag_col] == 1].copy()

        print("\nFiltered idiomatic-only shapes:")
        print("Train:", df_train.shape)
        print("Validation:", df_val.shape)
        print("Test:", df_test.shape)

    for df in (df_train, df_val, df_test):
        df[CFG.input_col] = df[CFG.input_col].astype(str)
        df[CFG.target_col] = df[CFG.target_col].astype(str)

    print("\nSample rows:")
    print(df_train[[CFG.input_col, CFG.target_col]].head(5))

    return df_train, df_val, df_test


# =========================================================
# Prepare seq2seq text
# =========================================================
def prepare_mt5_frames(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_train = df_train.copy()
    df_val = df_val.copy()
    df_test = df_test.copy()

    df_train["source_text"] = CFG.task_prefix + df_train[CFG.input_col]
    df_val["source_text"] = CFG.task_prefix + df_val[CFG.input_col]
    df_test["source_text"] = CFG.task_prefix + df_test[CFG.input_col]

    df_train["target_text"] = df_train[CFG.target_col]
    df_val["target_text"] = df_val[CFG.target_col]
    df_test["target_text"] = df_test[CFG.target_col]

    return df_train, df_val, df_test


# =========================================================
# Dataset class
# =========================================================
class IdiomSeq2SeqDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        max_source_length: int,
        max_target_length: int,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        source_text = str(row["source_text"])
        target_text = str(row["target_text"])

        model_inputs = self.tokenizer(
            source_text,
            max_length=self.max_source_length,
            truncation=True,
            padding=False,
        )

        labels = self.tokenizer(
            text_target=target_text,
            max_length=self.max_target_length,
            truncation=True,
            padding=False,
        )

        label_ids = labels["input_ids"]

        # replace pad tokens with -100 only if any exist
        label_ids = [
            token_id if token_id != self.tokenizer.pad_token_id else -100
            for token_id in label_ids
        ]

        item = {
            "input_ids": model_inputs["input_ids"],
            "attention_mask": model_inputs["attention_mask"],
            "labels": label_ids,
        }

        return item


# =========================================================
# Metrics for trainer
# =========================================================
def build_compute_metrics(tokenizer: MT5Tokenizer):
    def compute_metrics(eval_pred) -> Dict[str, float]:
        predictions, labels = eval_pred

        if isinstance(predictions, tuple):
            predictions = predictions[0]

        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = [normalize_text(x) for x in decoded_preds]
        decoded_labels = [normalize_text(x) for x in decoded_labels]

        metrics = {
            "exact_match_accuracy": exact_match_accuracy(decoded_preds, decoded_labels),
            "fuzzy_match_accuracy_085": fuzzy_match_accuracy(decoded_preds, decoded_labels, 0.85),
            "avg_token_overlap": average_token_overlap(decoded_preds, decoded_labels),
        }
        return metrics

    return compute_metrics


# =========================================================
# Plot helpers
# =========================================================
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


def save_error_breakdown_chart(pred_df: pd.DataFrame, out_path: Path) -> None:
    pred_df = pred_df.copy()
    pred_df["case"] = np.where(
        pred_df["exact_match"] == 1,
        "Exact match",
        np.where(pred_df["fuzzy_match_085"] == 1, "Fuzzy match only", "Wrong"),
    )
    chart_df = pred_df["case"].value_counts(normalize=True).reset_index()
    chart_df.columns = ["Case", "Proportion"]

    plt.figure(figsize=(7, 5))
    plt.bar(chart_df["Case"], chart_df["Proportion"])
    plt.title("Task 3 mT5 Prediction Breakdown")
    plt.ylabel("Proportion")
    plt.ylim(0, 1)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# =========================================================
# Prediction + evaluation
# =========================================================
def generate_predictions(
    model: MT5ForConditionalGeneration,
    tokenizer: MT5Tokenizer,
    df_test_mt5: pd.DataFrame,
) -> pd.DataFrame:
    model.eval()
    device = next(model.parameters()).device

    preds: List[str] = []
    golds: List[str] = []
    inputs: List[str] = []

    for _, row in df_test_mt5.iterrows():
        source_text = str(row["source_text"])
        gold_text = str(row["target_text"])

        encoded = tokenizer(
            source_text,
            max_length=CFG.max_source_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                max_length=CFG.generation_max_length,
                num_beams=CFG.num_beams,
                early_stopping=True,
            )

        pred = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        inputs.append(str(row[CFG.input_col]))
        golds.append(gold_text)
        preds.append(pred)

    pred_df = pd.DataFrame(
        {
            "input_text": inputs,
            "gold_idiom": golds,
            "prediction": preds,
        }
    )

    pred_df["gold_norm"] = pred_df["gold_idiom"].apply(normalize_text)
    pred_df["pred_norm"] = pred_df["prediction"].apply(normalize_text)
    pred_df["exact_match"] = (pred_df["gold_norm"] == pred_df["pred_norm"]).astype(int)
    pred_df["fuzzy_score"] = pred_df.apply(lambda r: fuzzy_ratio(r["prediction"], r["gold_idiom"]), axis=1)
    pred_df["fuzzy_match_085"] = (pred_df["fuzzy_score"] >= 0.85).astype(int)
    pred_df["token_overlap"] = pred_df.apply(lambda r: token_overlap_score(r["prediction"], r["gold_idiom"]), axis=1)

    return pred_df


def build_metrics(pred_df: pd.DataFrame) -> Dict[str, Any]:
    metrics = {
        "exact_match_accuracy": float(pred_df["exact_match"].mean()),
        "fuzzy_match_accuracy_085": float(pred_df["fuzzy_match_085"].mean()),
        "avg_token_overlap": float(pred_df["token_overlap"].mean()),
        "num_test_samples": int(len(pred_df)),
        "model_name": CFG.model_name,
        "idiomatic_only": bool(CFG.idiomatic_only),
    }
    return metrics


def save_text_report(metrics: Dict[str, Any], pred_df: pd.DataFrame, out_path: Path) -> None:
    exact_wrong = pred_df[pred_df["exact_match"] == 0].copy()
    hard_cases = pred_df[(pred_df["exact_match"] == 0) & (pred_df["fuzzy_match_085"] == 0)].copy()

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Task 3 - Arabic Context to Idiom - mT5 Baseline Report\n")
        f.write("=" * 70 + "\n\n")

        f.write("Metrics\n")
        f.write("-" * 30 + "\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

        f.write("\nSample exact-match correct predictions\n")
        f.write("-" * 30 + "\n")
        for _, row in pred_df[pred_df["exact_match"] == 1].head(10).iterrows():
            f.write(f"Input: {row['input_text']}\n")
            f.write(f"Gold : {row['gold_idiom']}\n")
            f.write(f"Pred : {row['prediction']}\n")
            f.write("\n")

        f.write("\nSample hard error cases\n")
        f.write("-" * 30 + "\n")
        for _, row in hard_cases.head(10).iterrows():
            f.write(f"Input: {row['input_text']}\n")
            f.write(f"Gold : {row['gold_idiom']}\n")
            f.write(f"Pred : {row['prediction']}\n")
            f.write(f"Fuzzy score: {row['fuzzy_score']:.4f}\n")
            f.write("\n")

        f.write("\nPaper-ready insight\n")
        f.write("-" * 30 + "\n")
        f.write(
            "The mT5 baseline models the task as direct sequence generation from Arabic context "
            "to canonical English idiom. Unlike retrieval systems, the model is not restricted "
            "to a fixed candidate set and can generate the target idiom directly from contextual semantics.\n"
        )


# =========================================================
# Main training function
# =========================================================
def run_task3_mt5_baseline() -> Dict[str, Any]:
    set_seed(CFG.seed)
    ensure_dirs()

    print("Project root:", PROJECT_ROOT)
    print("Model:", CFG.model_name)
    print("Output dir:", CFG.output_dir)
    print("Results dir:", CFG.results_dir)

    df_train, df_val, df_test = load_data()
    df_train_mt5, df_val_mt5, df_test_mt5 = prepare_mt5_frames(df_train, df_val, df_test)

    from transformers import AutoTokenizer, MT5ForConditionalGeneration

    tokenizer = AutoTokenizer.from_pretrained(CFG.model_name, use_fast=False)
    model = MT5ForConditionalGeneration.from_pretrained(CFG.model_name)

    train_dataset = IdiomSeq2SeqDataset(
        df_train_mt5, tokenizer, CFG.max_source_length, CFG.max_target_length
    )
    val_dataset = IdiomSeq2SeqDataset(
        df_val_mt5, tokenizer, CFG.max_source_length, CFG.max_target_length
    )
    test_dataset = IdiomSeq2SeqDataset(
        df_test_mt5, tokenizer, CFG.max_source_length, CFG.max_target_length
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=None,
    )

    sample_item = train_dataset[0]
    print("Sample input_ids:", sample_item["input_ids"][:20])
    print("Sample labels:", sample_item["labels"][:20])

    valid_label_tokens = [x for x in sample_item["labels"] if x != -100]
    print("Non-masked label token count:", len(valid_label_tokens))

    assert len(valid_label_tokens) > 0, "All label tokens are masked! Training will fail."

    # Newer transformers may use eval_strategy instead of evaluation_strategy
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(CFG.output_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=CFG.logging_steps,
        per_device_train_batch_size=CFG.train_batch_size,
        per_device_eval_batch_size=CFG.eval_batch_size,
        gradient_accumulation_steps=CFG.gradient_accumulation_steps,
        num_train_epochs=CFG.num_train_epochs,
        learning_rate=CFG.learning_rate,
        weight_decay=CFG.weight_decay,
        predict_with_generate=True,
        generation_max_length=CFG.generation_max_length,
        #fp16=torch.cuda.is_available(),
        fp16=False,
        save_total_limit=CFG.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        max_grad_norm=1.0,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(tokenizer),
    )

    print("\nStarting training...")
    trainer.train()

    print("\nSaving final model...")
    trainer.save_model(str(CFG.output_dir))
    tokenizer.save_pretrained(str(CFG.output_dir))

    print("\nGenerating test predictions...")
    pred_df = generate_predictions(model, tokenizer, df_test_mt5)

    metrics = build_metrics(pred_df)

    metrics_path = CFG.output_dir / "metrics.json"
    metrics_csv_path = CFG.output_dir / "metrics.csv"
    pred_path = CFG.output_dir / "test_predictions.csv"
    report_path = CFG.output_dir / "report.txt"

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    pd.DataFrame([metrics]).to_csv(metrics_csv_path, index=False)
    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")
    save_text_report(metrics, pred_df, report_path)

    # copy to desktop results folder too
    desktop_metrics_csv = CFG.results_dir / "metrics.csv"
    desktop_pred_csv = CFG.results_dir / "test_predictions.csv"
    desktop_report = CFG.results_dir / "report.txt"

    pd.DataFrame([metrics]).to_csv(desktop_metrics_csv, index=False)
    pred_df.to_csv(desktop_pred_csv, index=False, encoding="utf-8-sig")
    save_text_report(metrics, pred_df, desktop_report)

    # charts
    chart_df = pd.DataFrame(
        {
            "Metric": ["Exact Match", "Fuzzy@0.85", "Avg Token Overlap"],
            "Score": [
                metrics["exact_match_accuracy"],
                metrics["fuzzy_match_accuracy_085"],
                metrics["avg_token_overlap"],
            ],
        }
    )
    save_bar_chart(
        chart_df,
        x_col="Metric",
        y_col="Score",
        title="Task 3 mT5 Baseline Performance",
        out_path=CFG.results_dir / "task3_mt5_baseline_metrics.png",
    )
    save_error_breakdown_chart(
        pred_df,
        out_path=CFG.results_dir / "task3_mt5_error_breakdown.png",
    )

    # confusion-style label
    pred_df["case_label"] = np.where(
        pred_df["exact_match"] == 1,
        "Exact match",
        np.where(pred_df["fuzzy_match_085"] == 1, "Near match", "Wrong"),
    )
    case_counts = pred_df["case_label"].value_counts().reset_index()
    case_counts.columns = ["Case", "Count"]
    case_counts["Percent"] = case_counts["Count"] / case_counts["Count"].sum()
    case_counts.to_csv(CFG.results_dir / "task3_mt5_case_breakdown.csv", index=False)

    print("\nSaved outputs to:")
    print(CFG.output_dir)
    print(CFG.results_dir)

    print("\nFinal metrics:")
    print(metrics)

    return {
        "metrics": metrics,
        "predictions": pred_df,
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "test_size": len(test_dataset),
    }


if __name__ == "__main__":
    run_task3_mt5_baseline()