from pathlib import Path
import pandas as pd
import numpy as np
import torch

from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments
)

from datasets import Dataset
from sklearn.metrics import accuracy_score


BASE_DIR = Path(__file__).resolve().parents[2]

TRAIN_CSV = BASE_DIR / "deep_learning/datasets/surface_to_canonical/train.csv"
VAL_CSV = BASE_DIR / "deep_learning/datasets/surface_to_canonical/validation.csv"
TEST_CSV = BASE_DIR / "deep_learning/datasets/surface_to_canonical/test.csv"

OUTPUT_DIR = BASE_DIR / "deep_learning/models/surface_normalization_t5"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


print("Loading datasets...")

train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)
test_df = pd.read_csv(TEST_CSV)

for df in [train_df, val_df, test_df]:
    df["input_text"] = df["input_text"].astype(str).str.strip()
    df["target_text"] = df["target_text"].astype(str).str.strip()

print("Train rows:", len(train_df))
print("Validation rows:", len(val_df))
print("Test rows:", len(test_df))


MODEL_NAME = "t5-small"

print("Loading tokenizer and model:", MODEL_NAME)

tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)


def prepare_dataset(df):

    inputs = ["normalize idiom: " + x for x in df["input_text"].tolist()]
    targets = df["target_text"].tolist()

    model_inputs = tokenizer(
        inputs,
        padding="max_length",
        truncation=True,
        max_length=32
    )

    labels = tokenizer(
        targets,
        padding="max_length",
        truncation=True,
        max_length=16
    )["input_ids"]

    model_inputs["labels"] = labels

    return Dataset.from_dict(model_inputs)


train_ds = prepare_dataset(train_df)
val_ds = prepare_dataset(val_df)
test_ds = prepare_dataset(test_df)


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Training on device:", device)


training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    learning_rate=3e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    logging_steps=200,
    save_strategy="epoch",
    eval_strategy="epoch",
    fp16=torch.cuda.is_available()
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds
)


print("\nStarting training...\n")

trainer.train()

if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("\nEvaluating on test set...\n")

if torch.cuda.is_available():
    torch.cuda.empty_cache()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

all_preds = []

batch_size = 8

test_inputs = ["normalize idiom: " + x for x in test_df["input_text"].tolist()]
gold_texts = [g.strip() for g in test_df["target_text"].tolist()]

for start in range(0, len(test_inputs), batch_size):
    batch_texts = test_inputs[start:start + batch_size]

    enc = tokenizer(
        batch_texts,
        padding=True,
        truncation=True,
        max_length=32,
        return_tensors="pt"
    )

    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            max_new_tokens=16,
            num_beams=4
        )

    batch_preds = tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )

    all_preds.extend([p.strip() for p in batch_preds])

accuracy = accuracy_score(gold_texts, all_preds)

print("Exact Match Accuracy:", accuracy)

pd.DataFrame({
    "surface": test_df["input_text"],
    "gold": gold_texts,
    "prediction": all_preds
}).to_csv(
    OUTPUT_DIR / "test_predictions.csv",
    index=False,
    encoding="utf-8-sig"
)

pd.DataFrame([{
    "exact_match_accuracy": accuracy,
    "num_test_samples": len(gold_texts)
}]).to_csv(
    OUTPUT_DIR / "metrics.csv",
    index=False
)

print("\nSaved predictions to:", OUTPUT_DIR / "test_predictions.csv")
print("Saved metrics to:", OUTPUT_DIR / "metrics.csv")