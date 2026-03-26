from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    MT5ForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)


# =========================================================
# Project root
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
# Config
# =========================================================
@dataclass
class Config:
    model_name: str = "google/mt5-base"

    task_dir: Path = PROJECT_ROOT / "datasets" / "arabic_context_to_idiom"
    train_csv: Path = task_dir / "train.csv"
    val_csv: Path = task_dir / "validation.csv"
    test_csv: Path = task_dir / "test.csv"

    output_dir: Path = PROJECT_ROOT / "models" / "arabic_context_to_idiom_mt5_base_normalized"
    results_dir: Path = PROJECT_ROOT / "models" / "arabic_context_to_idiom_mt5_base_normalized" / "task3_mt5_base_normalized"

    input_col: str = "input_text"
    target_col: str = "target_text"
    idiom_flag_col: str = "is_example_idiom"

    idiomatic_only: bool = True

    task_prefix: str = "Predict the correct English idiom for this Arabic sentence: "

    max_source_length: int = 128
    max_target_length: int = 20

    num_train_epochs: int = 3
    learning_rate: float = 3e-5
    weight_decay: float = 0.01
    train_batch_size: int = 2
    eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8

    logging_steps: int = 100
    save_total_limit: int = 2
    seed: int = 42

    generation_max_length: int = 20
    num_beams: int = 6
    no_repeat_ngram_size: int = 2
    length_penalty: float = 1.0

    normalization_threshold: float = 0.78


CFG = Config()


# =========================================================
# Utils
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


def token_overlap_score(pred: str, gold: str) -> float:
    pred_tokens = set(normalize_text(pred).split())
    gold_tokens = set(normalize_text(gold).split())
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    return len(pred_tokens & gold_tokens) / len(pred_tokens | gold_tokens)


def exact_match_accuracy(preds: List[str], labels: List[str]) -> float:
    if not labels:
        return 0.0
    return sum(normalize_text(p) == normalize_text(y) for p, y in zip(preds, labels)) / len(labels)


def fuzzy_match_accuracy(preds: List[str], labels: List[str], threshold: float = 0.85) -> float:
    if not labels:
        return 0.0
    return sum(fuzzy_ratio(p, y) >= threshold for p, y in zip(preds, labels)) / len(labels)


def average_token_overlap(preds: List[str], labels: List[str]) -> float:
    if not labels:
        return 0.0
    return float(np.mean([token_overlap_score(p, y) for p, y in zip(preds, labels)]))


def build_idiom_bank(df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame) -> List[str]:
    bank = pd.concat(
        [
            df_train[[CFG.target_col]],
            df_val[[CFG.target_col]],
            df_test[[CFG.target_col]],
        ],
        axis=0,
        ignore_index=True,
    )
    idioms = sorted(bank[CFG.target_col].astype(str).dropna().str.strip().unique().tolist())
    return idioms


def normalize_to_bank(prediction: str, idiom_bank: List[str], threshold: float = 0.78) -> Tuple[str, float]:
    pred_norm = normalize_text(prediction)
    if not pred_norm:
        return prediction, 0.0

    best_idiom = prediction
    best_score = -1.0

    for idiom in idiom_bank:
        score = fuzzy_ratio(pred_norm, idiom)
        if score > best_score:
            best_score = score
            best_idiom = idiom

    if best_score >= threshold:
        return best_idiom, best_score

    return prediction, best_score


# =========================================================
# Data
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


def prepare_frames(
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
# Dataset
# =========================================================
class IdiomSeq2SeqDataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_source_length: int, max_target_length: int):
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

        label_ids = [
            token_id if token_id != self.tokenizer.pad_token_id else -100
            for token_id in labels["input_ids"]
        ]

        return {
            "input_ids": model_inputs["input_ids"],
            "attention_mask": model_inputs["attention_mask"],
            "labels": label_ids,
        }


# =========================================================
# Metrics for trainer
# =========================================================
def build_compute_metrics(tokenizer):
    def compute_metrics(eval_pred) -> Dict[str, float]:
        predictions, labels = eval_pred

        if isinstance(predictions, tuple):
            predictions = predictions[0]

        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = [normalize_text(x) for x in decoded_preds]
        decoded_labels = [normalize_text(x) for x in decoded_labels]

        return {
            "exact_match_accuracy": exact_match_accuracy(decoded_preds, decoded_labels),
            "fuzzy_match_accuracy_085": fuzzy_match_accuracy(decoded_preds, decoded_labels, 0.85),
            "avg_token_overlap": average_token_overlap(decoded_preds, decoded_labels),
        }

    return compute_metrics


# =========================================================
# Charts
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


# =========================================================
# Prediction / evaluation
# =========================================================
def generate_predictions(
    model: MT5ForConditionalGeneration,
    tokenizer,
    df_test_prepared: pd.DataFrame,
    idiom_bank: List[str],
) -> pd.DataFrame:
    model.eval()
    device = next(model.parameters()).device

    rows = []

    for _, row in df_test_prepared.iterrows():
        source_text = str(row["source_text"])
        input_text = str(row[CFG.input_col])
        gold = str(row["target_text"])

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
                no_repeat_ngram_size=CFG.no_repeat_ngram_size,
                length_penalty=CFG.length_penalty,
                early_stopping=True,
            )

        raw_pred = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        normalized_pred, bank_score = normalize_to_bank(
            raw_pred,
            idiom_bank,
            threshold=CFG.normalization_threshold,
        )

        rows.append(
            {
                "input_text": input_text,
                "gold_idiom": gold,
                "raw_prediction": raw_pred,
                "normalized_prediction": normalized_pred,
                "bank_match_score": bank_score,
            }
        )

    pred_df = pd.DataFrame(rows)

    pred_df["gold_norm"] = pred_df["gold_idiom"].apply(normalize_text)
    pred_df["raw_pred_norm"] = pred_df["raw_prediction"].apply(normalize_text)
    pred_df["norm_pred_norm"] = pred_df["normalized_prediction"].apply(normalize_text)

    pred_df["raw_exact_match"] = (pred_df["gold_norm"] == pred_df["raw_pred_norm"]).astype(int)
    pred_df["normalized_exact_match"] = (pred_df["gold_norm"] == pred_df["norm_pred_norm"]).astype(int)

    pred_df["raw_fuzzy_score"] = pred_df.apply(lambda r: fuzzy_ratio(r["raw_prediction"], r["gold_idiom"]), axis=1)
    pred_df["normalized_fuzzy_score"] = pred_df.apply(
        lambda r: fuzzy_ratio(r["normalized_prediction"], r["gold_idiom"]), axis=1
    )

    pred_df["raw_fuzzy_match_085"] = (pred_df["raw_fuzzy_score"] >= 0.85).astype(int)
    pred_df["normalized_fuzzy_match_085"] = (pred_df["normalized_fuzzy_score"] >= 0.85).astype(int)

    pred_df["raw_token_overlap"] = pred_df.apply(
        lambda r: token_overlap_score(r["raw_prediction"], r["gold_idiom"]), axis=1
    )
    pred_df["normalized_token_overlap"] = pred_df.apply(
        lambda r: token_overlap_score(r["normalized_prediction"], r["gold_idiom"]), axis=1
    )

    return pred_df


def build_final_metrics(pred_df: pd.DataFrame) -> Dict[str, Any]:
    metrics = {
        "raw_exact_match_accuracy": float(pred_df["raw_exact_match"].mean()),
        "normalized_exact_match_accuracy": float(pred_df["normalized_exact_match"].mean()),
        "raw_fuzzy_match_accuracy_085": float(pred_df["raw_fuzzy_match_085"].mean()),
        "normalized_fuzzy_match_accuracy_085": float(pred_df["normalized_fuzzy_match_085"].mean()),
        "raw_avg_token_overlap": float(pred_df["raw_token_overlap"].mean()),
        "normalized_avg_token_overlap": float(pred_df["normalized_token_overlap"].mean()),
        "num_test_samples": int(len(pred_df)),
        "model_name": CFG.model_name,
        "normalization_threshold": CFG.normalization_threshold,
        "idiomatic_only": bool(CFG.idiomatic_only),
    }
    return metrics


def save_report(metrics: Dict[str, Any], pred_df: pd.DataFrame, out_path: Path) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Task 3 - mT5-base + idiom-bank normalization\n")
        f.write("=" * 70 + "\n\n")

        f.write("Metrics\n")
        f.write("-" * 30 + "\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

        f.write("\nTop improvement examples after normalization\n")
        f.write("-" * 30 + "\n")
        improved = pred_df[
            (pred_df["raw_exact_match"] == 0) &
            (pred_df["normalized_exact_match"] == 1)
        ].copy()

        for _, row in improved.head(15).iterrows():
            f.write(f"Input: {row['input_text']}\n")
            f.write(f"Gold : {row['gold_idiom']}\n")
            f.write(f"Raw  : {row['raw_prediction']}\n")
            f.write(f"Norm : {row['normalized_prediction']}\n")
            f.write(f"Bank score: {row['bank_match_score']:.4f}\n\n")

        f.write("\nHard wrong cases\n")
        f.write("-" * 30 + "\n")
        hard_wrong = pred_df[pred_df["normalized_exact_match"] == 0].copy()
        for _, row in hard_wrong.head(15).iterrows():
            f.write(f"Input: {row['input_text']}\n")
            f.write(f"Gold : {row['gold_idiom']}\n")
            f.write(f"Raw  : {row['raw_prediction']}\n")
            f.write(f"Norm : {row['normalized_prediction']}\n")
            f.write(f"Bank score: {row['bank_match_score']:.4f}\n\n")


# =========================================================
# Main
# =========================================================
def run_task3_mt5_base_normalized() -> Dict[str, Any]:
    set_seed(CFG.seed)
    ensure_dirs()

    print("Project root:", PROJECT_ROOT)
    print("Model:", CFG.model_name)
    print("Output dir:", CFG.output_dir)
    print("Results dir:", CFG.results_dir)

    df_train, df_val, df_test = load_data()
    idiom_bank = build_idiom_bank(df_train, df_val, df_test)

    print("\nIdiom bank size:", len(idiom_bank))

    df_train_p, df_val_p, df_test_p = prepare_frames(df_train, df_val, df_test)

    tokenizer = AutoTokenizer.from_pretrained(CFG.model_name, use_fast=False)
    model = MT5ForConditionalGeneration.from_pretrained(CFG.model_name)

    train_dataset = IdiomSeq2SeqDataset(df_train_p, tokenizer, CFG.max_source_length, CFG.max_target_length)
    val_dataset = IdiomSeq2SeqDataset(df_val_p, tokenizer, CFG.max_source_length, CFG.max_target_length)

    sample_item = train_dataset[0]
    print("\nSample input_ids:", sample_item["input_ids"][:20])
    print("Sample labels:", sample_item["labels"][:20])
    valid_label_tokens = [x for x in sample_item["labels"] if x != -100]
    print("Non-masked label token count:", len(valid_label_tokens))
    assert len(valid_label_tokens) > 0, "All label tokens are masked"

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=None,
    )

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
        fp16=False,
        max_grad_norm=1.0,
        save_total_limit=CFG.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
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
    pred_df = generate_predictions(model, tokenizer, df_test_p, idiom_bank)
    metrics = build_final_metrics(pred_df)

    metrics_json_path = CFG.output_dir / "metrics.json"
    metrics_csv_path = CFG.output_dir / "metrics.csv"
    preds_csv_path = CFG.output_dir / "test_predictions.csv"
    report_path = CFG.output_dir / "report.txt"

    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    pd.DataFrame([metrics]).to_csv(metrics_csv_path, index=False)
    pred_df.to_csv(preds_csv_path, index=False, encoding="utf-8-sig")
    save_report(metrics, pred_df, report_path)

    # desktop copies
    pd.DataFrame([metrics]).to_csv(CFG.results_dir / "metrics.csv", index=False)
    pred_df.to_csv(CFG.results_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")
    save_report(metrics, pred_df, CFG.results_dir / "report.txt")

    # charts
    chart_df = pd.DataFrame(
        {
            "Metric": [
                "Raw Exact",
                "Normalized Exact",
                "Raw Fuzzy@0.85",
                "Normalized Fuzzy@0.85",
            ],
            "Score": [
                metrics["raw_exact_match_accuracy"],
                metrics["normalized_exact_match_accuracy"],
                metrics["raw_fuzzy_match_accuracy_085"],
                metrics["normalized_fuzzy_match_accuracy_085"],
            ],
        }
    )
    save_bar_chart(
        chart_df,
        x_col="Metric",
        y_col="Score",
        title="Task 3 mT5-base Normalized Performance",
        out_path=CFG.results_dir / "task3_mt5_base_normalized_metrics.png",
    )

    improvement_df = pd.DataFrame(
        {
            "Metric": ["Exact Match Gain", "Fuzzy Match Gain"],
            "Gain": [
                metrics["normalized_exact_match_accuracy"] - metrics["raw_exact_match_accuracy"],
                metrics["normalized_fuzzy_match_accuracy_085"] - metrics["raw_fuzzy_match_accuracy_085"],
            ],
        }
    )
    save_bar_chart(
        improvement_df,
        x_col="Metric",
        y_col="Gain",
        title="Normalization Gain over Raw Generation",
        out_path=CFG.results_dir / "task3_mt5_base_normalization_gain.png",
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
    run_task3_mt5_base_normalized()