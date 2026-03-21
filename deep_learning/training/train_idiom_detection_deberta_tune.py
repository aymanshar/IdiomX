import json
import shutil
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)

# ============================================================
# Reproducibility
# ============================================================

SEED = 42

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "deep_learning" / "datasets" / "idiom_detection"
MODEL_DIR = BASE_DIR / "deep_learning" / "models" / "idiom_detection_deberta_tuned"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "microsoft/deberta-v3-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# Data
# ============================================================

def load_data():
    train = pd.read_csv(DATA_DIR / "train.csv")
    val = pd.read_csv(DATA_DIR / "validation.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    return train, val, test


def prepare_df(df):
    df = df.copy()
    df["text"] = df["input_text"].astype(str).str.strip()
    df = df[df["label"].isin([0, 1])]
    df = df[df["text"] != ""]
    return df[["text", "label"]].reset_index(drop=True)


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.encodings = tokenizer(
            texts.tolist(),
            truncation=True,
            padding=True,
            max_length=max_len
        )
        self.labels = labels.tolist()

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


# ============================================================
# Metrics
# ============================================================

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    acc = accuracy_score(labels, preds)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# ============================================================
# TrainingArguments
# ============================================================

def build_training_args(
    output_dir: Path,
    learning_rate: float,
    weight_decay: float,
    warmup_ratio: float,
    train_bs: int,
    eval_bs: int,
    num_train_epochs: int = 5
):
    common_kwargs = dict(
        output_dir=str(output_dir),
        learning_rate=learning_rate,
        per_device_train_batch_size=train_bs,
        per_device_eval_batch_size=eval_bs,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        save_strategy="epoch",
        logging_dir=str(output_dir / "logs"),
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=100,
        fp16=torch.cuda.is_available(),
        report_to="none"
    )

    try:
        return TrainingArguments(evaluation_strategy="epoch", **common_kwargs)
    except TypeError:
        try:
            return TrainingArguments(eval_strategy="epoch", **common_kwargs)
        except TypeError:
            return TrainingArguments(do_eval=True, **common_kwargs)


# ============================================================
# FINAL TUNED MODEL (IMPORTANT)
# ============================================================

def train_best_model():
    print("Training final tuned DeBERTa model...")

    set_seed()

    train_df, val_df, test_df = load_data()
    train_df = prepare_df(train_df)
    val_df = prepare_df(val_df)
    test_df = prepare_df(test_df)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

    train_dataset = TextDataset(train_df["text"], train_df["label"], tokenizer)
    val_dataset = TextDataset(val_df["text"], val_df["label"], tokenizer)
    test_dataset = TextDataset(test_df["text"], test_df["label"], tokenizer)

    # ✅ BEST CONFIG (from your tuning)
    training_args = build_training_args(
        output_dir=MODEL_DIR,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.10,
        train_bs=8,
        eval_bs=8,
        num_train_epochs=5
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        use_safetensors=True
    ).to(DEVICE)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]
    )

    trainer.train()

    test_output = trainer.predict(test_dataset)

    preds = np.argmax(test_output.predictions, axis=1)
    probs = torch.softmax(torch.tensor(test_output.predictions), dim=1)[:, 1].numpy()

    precision, recall, f1, _ = precision_recall_fscore_support(
        test_output.label_ids, preds, average="binary", zero_division=0
    )
    acc = accuracy_score(test_output.label_ids, preds)

    metrics_df = pd.DataFrame([{
        "model": "DeBERTa-tuned",
        "test_accuracy": acc,
        "test_precision": precision,
        "test_recall": recall,
        "test_f1": f1
    }])

    metrics_df.to_csv(MODEL_DIR / "metrics.csv", index=False)

    preds_df = test_df.copy()
    preds_df["pred"] = preds
    preds_df["prob_idiom"] = probs
    preds_df.to_csv(MODEL_DIR / "predictions.csv", index=False)

    trainer.save_model(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)

    print("Final tuned model training completed.")

    return {
        "model_dir": MODEL_DIR,
        "metrics_path": MODEL_DIR / "metrics.csv",
        "predictions_path": MODEL_DIR / "predictions.csv"
    }


# ============================================================
# INFERENCE
# ============================================================

_tokenizer = None
_model = None

def load_model():
    global _tokenizer, _model

    if _model is not None:
        return _tokenizer, _model

    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    _model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR.as_posix(),
        local_files_only=True
    )
    _model.eval()

    return _tokenizer, _model


def predict(text: str):
    tokenizer, model = load_model()

    enc = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**enc)
        probs = torch.softmax(outputs.logits, dim=1)[0].numpy()

    pred = int(probs.argmax())

    return {
        "text": text,
        "prediction": "Idiomatic" if pred == 1 else "Literal",
        "confidence_idiom": float(probs[1]),
        "confidence_literal": float(probs[0]),
    }