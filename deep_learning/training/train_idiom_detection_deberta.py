import pandas as pd
import torch
import transformers

from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)

# ==============================
# Paths
# ==============================

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "deep_learning" / "datasets" / "idiom_detection"
MODEL_DIR = BASE_DIR / "deep_learning" / "models" / "idiom_detection_deberta"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)


# ==============================
# Load Data
# ==============================

def load_data():
    train = pd.read_csv(DATA_DIR / "train.csv")
    val = pd.read_csv(DATA_DIR / "validation.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    return train, val, test


# ==============================
# Prepare Data
# ==============================

def prepare_df(df):
    df = df.copy()
    df["text"] = df["input_text"].astype(str)
    df = df[df["label"].isin([0, 1])]
    return df[["text", "label"]].reset_index(drop=True)


# ==============================
# Dataset Class
# ==============================

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


# ==============================
# Metrics
# ==============================

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=1)

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


# ==============================
# TrainingArguments compatibility
# ==============================

def build_training_args(output_dir: Path):
    common_kwargs = dict(
        output_dir=str(output_dir),
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=5,
        weight_decay=0.01,
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
        return TrainingArguments(
            evaluation_strategy="epoch",
            **common_kwargs
        )
    except TypeError:
        pass

    try:
        return TrainingArguments(
            eval_strategy="epoch",
            **common_kwargs
        )
    except TypeError:
        pass

    return TrainingArguments(
        do_eval=True,
        **common_kwargs
    )


# ==============================
# Train Function
# ==============================

def train_model():
    print("Loading data...")
    train_df, val_df, test_df = load_data()

    train_df = prepare_df(train_df)
    val_df = prepare_df(val_df)
    test_df = prepare_df(test_df)

    model_name = "microsoft/deberta-v3-base"

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    print("Building datasets...")
    train_dataset = TextDataset(train_df["text"], train_df["label"], tokenizer)
    val_dataset = TextDataset(val_df["text"], val_df["label"], tokenizer)
    test_dataset = TextDataset(test_df["text"], test_df["label"], tokenizer)

    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        use_safetensors=True
    ).to(DEVICE)

    training_args = build_training_args(MODEL_DIR)
    print("Transformers version:", transformers.__version__)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]
    )

    print("Training DeBERTa...")
    trainer.train()

    print("Evaluating on validation set...")
    trainer.evaluate()

    print("Evaluating on test set...")
    test_preds = trainer.predict(test_dataset)

    preds = test_preds.predictions.argmax(axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        test_preds.label_ids, preds, average="binary", zero_division=0
    )
    acc = accuracy_score(test_preds.label_ids, preds)

    metrics_df = pd.DataFrame([{
        "model": "DeBERTa",
        "test_accuracy": acc,
        "test_precision": precision,
        "test_recall": recall,
        "test_f1": f1
    }])

    metrics_path = MODEL_DIR / "metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    probs = torch.softmax(torch.tensor(test_preds.predictions), dim=1)[:, 1].numpy()

    preds_df = test_df.copy()
    preds_df["pred"] = preds
    preds_df["prob_idiom"] = probs

    preds_path = MODEL_DIR / "predictions.csv"
    preds_df.to_csv(preds_path, index=False)

    # Save final model + tokenizer for inference
    trainer.save_model(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)

    print("Training completed.")

    return {
        "model_dir": MODEL_DIR,
        "metrics_path": metrics_path,
        "predictions_path": preds_path
    }


# ==============================
# Inference helper
# ==============================

_tokenizer = None
_model = None

def load_inference_model():
    global _tokenizer, _model

    if _model is not None:
        return _tokenizer, _model

    model_name = "microsoft/deberta-v3-base"
    _tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    _model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR.as_posix(),
        local_files_only=True
    )
    _model.eval()

    return _tokenizer, _model


def predict(text: str):
    tokenizer, model = load_inference_model()

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


if __name__ == "__main__":
    train_model()