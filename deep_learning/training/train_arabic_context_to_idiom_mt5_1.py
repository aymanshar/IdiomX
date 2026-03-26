from pathlib import Path
import pandas as pd

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)

from sklearn.metrics import accuracy_score
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

config = AutoConfig.from_pretrained(MODEL_NAME)
config.tie_word_embeddings = False

# Resolve paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
BASE_DIR = PROJECT_ROOT / "datasets" / "arabic_context_to_idiom"


train_df = pd.read_csv(BASE_DIR / "train.csv")
val_df = pd.read_csv(BASE_DIR / "validation.csv")
test_df = pd.read_csv(BASE_DIR / "test.csv")


train_ds = Dataset.from_pandas(train_df[["input_text", "target_text"]])
val_ds = Dataset.from_pandas(val_df[["input_text", "target_text"]])
test_ds = Dataset.from_pandas(test_df[["input_text", "target_text"]])


MODEL_NAME = "google/mt5-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)


MAX_INPUT = 128
MAX_TARGET = 32


def preprocess(example):

    model_inputs = tokenizer(
        example["input_text"],
        max_length=MAX_INPUT,
        truncation=True,
    )

    labels = tokenizer(
        example["target_text"],
        max_length=MAX_TARGET,
        truncation=True,
    )

    model_inputs["labels"] = labels["input_ids"]

    return model_inputs


train_ds = train_ds.map(preprocess, batched=True)
val_ds = val_ds.map(preprocess, batched=True)
test_ds = test_ds.map(preprocess, batched=True)


model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME,config=config)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

training_args = Seq2SeqTrainingArguments(

    output_dir=str(PROJECT_ROOT / "models" / "mt5_arabic_context_to_idiom"),

    learning_rate=3e-5,

    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,

    num_train_epochs=3,

    predict_with_generate=True,

    eval_strategy="epoch",
    save_strategy="epoch",

    load_best_model_at_end=True,
)


trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    processing_class=tokenizer,
    data_collator=data_collator,
)


print(f"Train rows: {len(train_df):,}")
print(f"Validation rows: {len(val_df):,}")
print(f"Test rows: {len(test_df):,}")

trainer.train()


print("\nEvaluating on test set...\n")

predictions = trainer.predict(test_ds)

pred_text = tokenizer.batch_decode(
    predictions.predictions,
    skip_special_tokens=True
)

true_text = test_df["target_text"].tolist()


acc = accuracy_score(true_text, pred_text)

print("Exact match accuracy:", acc)