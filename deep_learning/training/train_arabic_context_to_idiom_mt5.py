# ============================================================
# Arabic Context → Idiom (mT5)
# Full Training + Evaluation Pipeline
# ============================================================

import os
import json
import random
from pathlib import Path

import pandas as pd
import torch

from transformers import (
    AutoTokenizer,
    MT5ForConditionalGeneration,
    Trainer,
    TrainingArguments
)

# ------------------------------------------------------------
# Utils
# ------------------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_csv_checked(path):
    df = pd.read_csv(path)
    if "input_text" not in df.columns or "target_text" not in df.columns:
        raise ValueError("CSV must contain input_text and target_text")
    return df


# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------
class Seq2SeqDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        return {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}


# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------
def compute_exact_match(preds, labels):
    correct = sum(p.strip() == l.strip() for p, l in zip(preds, labels))
    return correct / len(preds)


# ------------------------------------------------------------
# Main Function
# ------------------------------------------------------------
def run_arabic_context_mt5(
    train_csv,
    val_csv,
    test_csv,
    model_name="google/mt5-base",
    output_dir="models/mt5",
    batch_size=8,
    num_epochs=3,
    max_input_length=128,
    max_target_length=32,
    learning_rate=3e-5,
    seed=42,
):

    set_seed(seed)

    print("Running mT5 model for Task 3")

    train_df = load_csv_checked(train_csv)
    val_df = load_csv_checked(val_csv)
    test_df = load_csv_checked(test_csv)

    print(f"Train size: {len(train_df)}")
    print(f"Val size: {len(val_df)}")
    print(f"Test size: {len(test_df)}")

    # ------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------
    print("\nLoading mT5 model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    model = MT5ForConditionalGeneration.from_pretrained(
        model_name,
        use_safetensors=True
    )

    print("Model loaded")

    # ------------------------------------------------------------
    # Prompt
    # ------------------------------------------------------------
    def format_input(x):
        return f"Translate Arabic context to English idiom: {x}"

    train_df["input_fmt"] = train_df["input_text"].apply(format_input)
    val_df["input_fmt"] = val_df["input_text"].apply(format_input)
    test_df["input_fmt"] = test_df["input_text"].apply(format_input)

    # ------------------------------------------------------------
    # Tokenization
    # ------------------------------------------------------------
    def tokenize(texts, targets):
        model_inputs = tokenizer(
            texts,
            max_length=max_input_length,
            truncation=True,
            padding="max_length"
        )

        labels = tokenizer(
            targets,
            max_length=max_target_length,
            truncation=True,
            padding="max_length"
        )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    print("\nTokenizing...")
    train_enc = tokenize(train_df["input_fmt"].tolist(), train_df["target_text"].tolist())
    val_enc = tokenize(val_df["input_fmt"].tolist(), val_df["target_text"].tolist())

    train_dataset = Seq2SeqDataset(train_enc)
    val_dataset = Seq2SeqDataset(val_enc)

    print("Tokenization complete")

    # ------------------------------------------------------------
    # Training
    # ------------------------------------------------------------
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        logging_dir=str(output_dir / "logs"),
        logging_steps=100,
        save_steps=1000,
        fp16=torch.cuda.is_available(),
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    print("\nStarting training...")
    trainer.train()

    # ------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------
    print("\nEvaluating...")

    def generate_predictions(df):
        preds = []
        for text in df["input_fmt"]:
            inputs = tokenizer(text, return_tensors="pt", truncation=True).to(model.device)
            outputs = model.generate(**inputs, max_length=max_target_length)
            pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
            preds.append(pred)
        return preds

    val_preds = generate_predictions(val_df)
    test_preds = generate_predictions(test_df)

    val_em = compute_exact_match(val_preds, val_df["target_text"].tolist())
    test_em = compute_exact_match(test_preds, test_df["target_text"].tolist())

    results = {
        "validation_exact_match": val_em,
        "test_exact_match": test_em
    }

    print("\nResults:")
    print(results)

    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    pd.DataFrame({
        "input": val_df["input_text"],
        "target": val_df["target_text"],
        "prediction": val_preds
    }).to_csv(output_dir / "val_predictions.csv", index=False)

    print("\nSaved results to:", output_dir)

    return results


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def main():
    run_arabic_context_mt5(
        train_csv="data/train.csv",
        val_csv="data/validation.csv",
        test_csv="data/test.csv"
    )


if __name__ == "__main__":
    main()