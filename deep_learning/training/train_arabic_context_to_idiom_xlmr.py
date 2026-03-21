from pathlib import Path
import pandas as pd

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

from sklearn.metrics import accuracy_score, f1_score


# Resolve project paths robustly
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
BASE_DIR = PROJECT_ROOT / "datasets" / "arabic_context_to_idiom"


train_df = pd.read_csv(BASE_DIR / "train.csv")
val_df = pd.read_csv(BASE_DIR / "validation.csv")
test_df = pd.read_csv(BASE_DIR / "test.csv")


# Build label mapping from train split idioms
label_list = sorted(train_df["target_text"].astype(str).str.strip().unique())
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for label, i in label2id.items()}

# Keep only rows whose target idiom exists in the train label space
train_df = train_df[train_df["target_text"].isin(label2id)].copy()
val_df = val_df[val_df["target_text"].isin(label2id)].copy()
test_df = test_df[test_df["target_text"].isin(label2id)].copy()

train_df["label"] = train_df["target_text"].map(label2id)
val_df["label"] = val_df["target_text"].map(label2id)
test_df["label"] = test_df["target_text"].map(label2id)

train_ds = Dataset.from_pandas(train_df[["input_text", "label"]])
val_ds = Dataset.from_pandas(val_df[["input_text", "label"]])
test_ds = Dataset.from_pandas(test_df[["input_text", "label"]])

MODEL_NAME = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize(example):
    return tokenizer(
        example["input_text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )


train_ds = train_ds.map(tokenize, batched=True)
val_ds = val_ds.map(tokenize, batched=True)
test_ds = test_ds.map(tokenize, batched=True)

train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
val_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])


model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id
)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro")
    }


training_args = TrainingArguments(
    output_dir=str(PROJECT_ROOT / "models" / "xlmr_arabic_context_to_idiom"),
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    processing_class=tokenizer,
    compute_metrics=compute_metrics
)


print(f"Train rows: {len(train_df):,}")
print(f"Validation rows: {len(val_df):,}")
print(f"Test rows: {len(test_df):,}")
print(f"Number of idiom classes: {len(label_list):,}")

trainer.train()

print("\nEvaluating on test set...\n")
results = trainer.evaluate(test_ds)
print(results)