from pathlib import Path
import pandas as pd
import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)

from sklearn.metrics import accuracy_score

BASE_DIR = Path(__file__).resolve().parents[2]

TRAIN_CSV = BASE_DIR / "deep_learning/datasets/idiom_to_meaning/train.csv"
VAL_CSV = BASE_DIR / "deep_learning/datasets/idiom_to_meaning/validation.csv"
TEST_CSV = BASE_DIR / "deep_learning/datasets/idiom_to_meaning/test.csv"

OUTPUT_DIR = BASE_DIR / "deep_learning/models/meaning_to_idiom_flan_t5"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading datasets...")

train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)
test_df = pd.read_csv(TEST_CSV)

# Correct task mapping: meaning -> idiom
train_df["input_text"] = train_df["target_meaning_en"].astype(str).str.strip()
val_df["input_text"] = val_df["target_meaning_en"].astype(str).str.strip()
test_df["input_text"] = test_df["target_meaning_en"].astype(str).str.strip()

train_df["target_text"] = train_df["input_text"].astype(str).str.strip()
val_df["target_text"] = val_df["input_text"].astype(str).str.strip()
test_df["target_text"] = test_df["input_text"].astype(str).str.strip()

# Actual idiom label
train_df["target_text"] = train_df["input_text"].astype(str).str.strip()
val_df["target_text"] = val_df["input_text"].astype(str).str.strip()
test_df["target_text"] = test_df["input_text"].astype(str).str.strip()

# The idiom_to_meaning dataset uses:
# input_text = idiom
# target_meaning_en = meaning
# so for meaning -> idiom:
train_df["target_text"] = pd.read_csv(TRAIN_CSV)["input_text"].astype(str).str.strip()
val_df["target_text"] = pd.read_csv(VAL_CSV)["input_text"].astype(str).str.strip()
test_df["target_text"] = pd.read_csv(TEST_CSV)["input_text"].astype(str).str.strip()

print("Train rows:", len(train_df))
print("Validation rows:", len(val_df))
print("Test rows:", len(test_df))

MODEL_NAME = "google/flan-t5-base"

print("Loading tokenizer and model:", MODEL_NAME)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

MAX_INPUT = 128
MAX_TARGET = 16

def preprocess(df):
    inputs = [
        "Given the meaning, generate the correct English idiom: " + x
        for x in df["input_text"].tolist()
    ]

    targets = df["target_text"].tolist()

    model_inputs = tokenizer(
        inputs,
        padding="max_length",
        truncation=True,
        max_length=MAX_INPUT
    )

    labels = tokenizer(
        targets,
        padding="max_length",
        truncation=True,
        max_length=MAX_TARGET
    )["input_ids"]

    labels = [
        [(token if token != tokenizer.pad_token_id else -100) for token in seq]
        for seq in labels
    ]

    model_inputs["labels"] = labels

    return Dataset.from_dict(model_inputs)

train_ds = preprocess(train_df)
val_ds = preprocess(val_df)
test_ds = preprocess(test_df)

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

training_args = Seq2SeqTrainingArguments(
    output_dir=str(OUTPUT_DIR),
    learning_rate=3e-5,
    per_device_train_batch_size=8 if torch.cuda.is_available() else 2,
    per_device_eval_batch_size=4 if torch.cuda.is_available() else 2,
    num_train_epochs=2,
    predict_with_generate=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    fp16=False,
    bf16=True if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else False,
    logging_steps=200,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    processing_class=tokenizer,
    data_collator=data_collator,
)

print("\nStarting training...\n")
print("Sample label ids:", train_ds[0]["labels"][:10])

trainer.train()

print("\nEvaluating on test set...\n")

if torch.cuda.is_available():
    torch.cuda.empty_cache()

model.to(device)
model.eval()

all_preds = []
batch_size = 8 if torch.cuda.is_available() else 2

test_inputs = [
    "Given the meaning, generate the correct English idiom: " + x
    for x in test_df["input_text"].tolist()
]
gold_texts = [g.strip() for g in test_df["target_text"].tolist()]

for start in range(0, len(test_inputs), batch_size):
    batch = test_inputs[start:start + batch_size]

    enc = tokenizer(
        batch,
        padding=True,
        truncation=True,
        max_length=MAX_INPUT,
        return_tensors="pt"
    )

    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        generated = model.generate(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            max_new_tokens=16,
            num_beams=4
        )

    preds = tokenizer.batch_decode(
        generated,
        skip_special_tokens=True
    )

    all_preds.extend([p.strip() for p in preds])

accuracy = accuracy_score(gold_texts, all_preds)

print("Exact Match Accuracy:", accuracy)

pd.DataFrame({
    "meaning": test_df["input_text"],
    "gold_idiom": gold_texts,
    "prediction": all_preds
}).to_csv(
    OUTPUT_DIR / "test_predictions.csv",
    index=False
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