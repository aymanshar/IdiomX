from pathlib import Path
import sys
import pandas as pd
import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
from sklearn.metrics import accuracy_score

BASE_DIR = Path(__file__).resolve().parents[2]
TRAINING_DIR = BASE_DIR / "deep_learning" / "training"

if str(TRAINING_DIR) not in sys.path:
    sys.path.append(str(TRAINING_DIR))

from tools import load_csv_checked, ensure_text_pair_columns, ensure_dir

TRAIN_CSV = BASE_DIR / "deep_learning" / "datasets" / "context_to_idiom" / "train.csv"
VAL_CSV = BASE_DIR / "deep_learning" / "datasets" / "context_to_idiom" / "validation.csv"
TEST_CSV = BASE_DIR / "deep_learning" / "datasets" / "context_to_idiom" / "test.csv"

OUTPUT_DIR = BASE_DIR / "deep_learning" / "models" / "context_to_idiom_flan_t5"
MODEL_NAME = "google/flan-t5-base"

MAX_INPUT = 128
MAX_TARGET = 32
DEFAULT_PROMPT_PREFIX = "Given the context, generate the best English idiom: "


class ContextToIdiomDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_input: int, max_target: int, prompt_prefix: str):
        self.tokenizer = tokenizer
        self.max_input = max_input
        self.max_target = max_target
        self.prompt_prefix = prompt_prefix

        self.inputs = [self.prompt_prefix + x for x in df["input_text"].astype(str).tolist()]
        self.targets = df["target_text"].astype(str).tolist()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_text = self.inputs[idx]
        target_text = self.targets[idx]

        model_inputs = self.tokenizer(
            input_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_input,
            return_tensors="pt",
        )

        labels = self.tokenizer(
            target_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_target,
            return_tensors="pt",
        )["input_ids"].squeeze(0)

        labels = torch.where(labels == self.tokenizer.pad_token_id, -100, labels)

        return {
            "input_ids": model_inputs["input_ids"].squeeze(0),
            "attention_mask": model_inputs["attention_mask"].squeeze(0),
            "labels": labels,
        }


def run_flan_t5_training(
    train_csv: Path = TRAIN_CSV,
    val_csv: Path = VAL_CSV,
    test_csv: Path = TEST_CSV,
    output_dir: Path = OUTPUT_DIR,
    model_name: str = MODEL_NAME,
    num_epochs: int = 2,
    max_input: int = MAX_INPUT,
    max_target: int = MAX_TARGET,
    prompt_prefix: str = DEFAULT_PROMPT_PREFIX,
    save_outputs: bool = True,
):
    output_dir = ensure_dir(Path(output_dir))

    print("Loading datasets...")

    train_df = load_csv_checked(Path(train_csv), low_memory=False)
    val_df = load_csv_checked(Path(val_csv), low_memory=False)
    test_df = load_csv_checked(Path(test_csv), low_memory=False)

    train_df = ensure_text_pair_columns(train_df, input_col="input_text", target_col="target_text")
    val_df = ensure_text_pair_columns(val_df, input_col="input_text", target_col="target_text")
    test_df = ensure_text_pair_columns(test_df, input_col="input_text", target_col="target_text")

    print("Train:", len(train_df))
    print("Validation:", len(val_df))
    print("Test:", len(test_df))

    print("Loading tokenizer and model:", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    train_ds = ContextToIdiomDataset(train_df, tokenizer, max_input, max_target, prompt_prefix)
    val_ds = ContextToIdiomDataset(val_df, tokenizer, max_input, max_target, prompt_prefix)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        learning_rate=3e-5,
        per_device_train_batch_size=8 if torch.cuda.is_available() else 2,
        per_device_eval_batch_size=4 if torch.cuda.is_available() else 2,
        num_train_epochs=num_epochs,
        predict_with_generate=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_steps=200,
        fp16=False,
        bf16=True if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else False,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("\nStarting training...\n")
    trainer.train()

    print("\nEvaluating on test set...\n")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model.to(device)
    model.eval()

    test_inputs = [prompt_prefix + x for x in test_df["input_text"].tolist()]
    gold_texts = [g.strip() for g in test_df["target_text"].tolist()]

    all_preds = []
    batch_size = 8 if torch.cuda.is_available() else 2

    for start in range(0, len(test_inputs), batch_size):
        batch_texts = test_inputs[start:start + batch_size]

        enc = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_input,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                max_new_tokens=16,
                num_beams=4,
            )

        batch_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        all_preds.extend([p.strip() for p in batch_preds])

    exact_match = accuracy_score(gold_texts, all_preds)

    print("Exact Match Accuracy:", exact_match)

    if save_outputs:
        pd.DataFrame(
            {
                "input_text": test_df["input_text"],
                "gold_idiom": gold_texts,
                "prediction": all_preds,
            }
        ).to_csv(output_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")

        pd.DataFrame(
            [
                {
                    "exact_match_accuracy": exact_match,
                    "num_test_samples": len(gold_texts),
                    "model_name": model_name,
                }
            ]
        ).to_csv(output_dir / "metrics.csv", index=False, encoding="utf-8-sig")

        print("\nSaved predictions to:", output_dir / "test_predictions.csv")
        print("Saved metrics to:", output_dir / "metrics.csv")

    return {
        "exact_match_accuracy": exact_match,
        "predictions": all_preds,
        "num_test_samples": len(gold_texts),
        "model_name": model_name,
    }


def main():
    run_flan_t5_training()


if __name__ == "__main__":
    main()