from pathlib import Path
import json
import random
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    confusion_matrix,
    precision_score,
    recall_score
)


# ----------------------------------------------------
# Paths
# ----------------------------------------------------

def get_project_paths():
    current_file = Path(__file__).resolve() if "__file__" in globals() else Path.cwd()

    if current_file.is_file():
        script_dir = current_file.parent
    else:
        script_dir = current_file

    project_root = script_dir.parent.parent
    data_dir = project_root / "deep_learning" / "datasets" / "idiom_detection"
    model_dir = project_root / "deep_learning" / "models" / "idiom_detection_baseline"
    model_dir.mkdir(parents=True, exist_ok=True)

    return {
        "script_dir": script_dir,
        "project_root": project_root,
        "data_dir": data_dir,
        "model_dir": model_dir,
    }


PATHS = get_project_paths()
DATA_DIR = PATHS["data_dir"]
MODEL_DIR = PATHS["model_dir"]


# ----------------------------------------------------
# Reproducibility
# ----------------------------------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)


# ----------------------------------------------------
# Data loading
# ----------------------------------------------------

def load_data(data_dir: Optional[Path] = None):
    data_dir = Path(data_dir) if data_dir is not None else DATA_DIR

    train_path = data_dir / "train.csv"
    val_path = data_dir / "validation.csv"
    test_path = data_dir / "test.csv"

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    required_cols = {"input_text", "label"}
    for name, df in [("train", train_df), ("validation", val_df), ("test", test_df)]:
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"{name} split is missing required columns: {missing}")

    return train_df, val_df, test_df


def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["input_text"] = df["input_text"].astype(str).str.strip()
    df = df[df["label"].isin([0, 1])]
    df = df[df["input_text"] != ""].reset_index(drop=True)
    return df


# ----------------------------------------------------
# Model building
# ----------------------------------------------------

def build_vectorizer():
    return TfidfVectorizer(
        max_features=30000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )


def build_model():
    return LogisticRegression(
        max_iter=1000,
        random_state=42,
        n_jobs=-1
    )


# ----------------------------------------------------
# Evaluation
# ----------------------------------------------------

def evaluate_split(model, X_vec, y_true, split_name: str = "test") -> Dict[str, Any]:
    preds = model.predict(X_vec)

    metrics = {
        f"{split_name}_accuracy": float(accuracy_score(y_true, preds)),
        f"{split_name}_precision": float(precision_score(y_true, preds, zero_division=0)),
        f"{split_name}_recall": float(recall_score(y_true, preds, zero_division=0)),
        f"{split_name}_f1": float(f1_score(y_true, preds, zero_division=0)),
        f"{split_name}_macro_f1": float(f1_score(y_true, preds, average="macro", zero_division=0)),
        f"{split_name}_weighted_f1": float(f1_score(y_true, preds, average="weighted", zero_division=0)),
        f"{split_name}_report": classification_report(y_true, preds, output_dict=True, zero_division=0),
        f"{split_name}_confusion_matrix": confusion_matrix(y_true, preds).tolist()
    }

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_vec)[:, 1]
        metrics[f"{split_name}_prob_idiom"] = probs.tolist()

    return {
        "preds": preds,
        "metrics": metrics
    }


# ----------------------------------------------------
# Save artifacts
# ----------------------------------------------------

def save_artifacts(
    model,
    vectorizer,
    metrics: Dict[str, Any],
    test_df: pd.DataFrame,
    test_preds,
    model_dir: Optional[Path] = None
):
    model_dir = Path(model_dir) if model_dir is not None else MODEL_DIR
    model_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_dir / "model.joblib")
    joblib.dump(vectorizer, model_dir / "vectorizer.joblib")

    metrics_flat = {
        "model": "Logistic Regression",
        "task": "Idiom Detection",
        "validation_accuracy": metrics["validation_accuracy"],
        "validation_precision": metrics["validation_precision"],
        "validation_recall": metrics["validation_recall"],
        "validation_f1": metrics["validation_f1"],
        "validation_macro_f1": metrics["validation_macro_f1"],
        "validation_weighted_f1": metrics["validation_weighted_f1"],
        "test_accuracy": metrics["test_accuracy"],
        "test_precision": metrics["test_precision"],
        "test_recall": metrics["test_recall"],
        "test_f1": metrics["test_f1"],
        "test_macro_f1": metrics["test_macro_f1"],
        "test_weighted_f1": metrics["test_weighted_f1"],
        "support": int(len(test_df))
    }

    pd.DataFrame([metrics_flat]).to_csv(
        model_dir / "metrics.csv",
        index=False,
        encoding="utf-8-sig"
    )

    with open(model_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    output_df = test_df.copy()
    output_df["pred"] = test_preds

    if "test_prob_idiom" in metrics:
        output_df["prob_idiom"] = metrics["test_prob_idiom"]
        output_df["prob_literal"] = 1 - output_df["prob_idiom"]

    output_df.to_csv(
        model_dir / "test_predictions.csv",
        index=False,
        encoding="utf-8-sig"
    )


# ----------------------------------------------------
# Main train function
# ----------------------------------------------------

def train_model(
    run_training: bool = True,
    data_dir: Optional[Path] = None,
    model_dir: Optional[Path] = None,
    seed: int = 42
):
    set_seed(seed)

    data_dir = Path(data_dir) if data_dir is not None else DATA_DIR
    model_dir = Path(model_dir) if model_dir is not None else MODEL_DIR
    model_dir.mkdir(parents=True, exist_ok=True)

    train_df, val_df, test_df = load_data(data_dir=data_dir)

    train_df = prepare_df(train_df)
    val_df = prepare_df(val_df)
    test_df = prepare_df(test_df)

    X_train = train_df["input_text"]
    y_train = train_df["label"].astype(int)

    X_val = val_df["input_text"]
    y_val = val_df["label"].astype(int)

    X_test = test_df["input_text"]
    y_test = test_df["label"].astype(int)

    vectorizer = build_vectorizer()
    model = build_model()

    print("Vectorizing text with TF-IDF...")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    X_test_vec = vectorizer.transform(X_test)

    if run_training:
        print("Training Logistic Regression model...")
        model.fit(X_train_vec, y_train)

    print("\nEvaluating on validation set...")
    val_out = evaluate_split(model, X_val_vec, y_val, split_name="validation")
    print("Validation Accuracy:", round(val_out["metrics"]["validation_accuracy"], 4))
    print(classification_report(y_val, val_out["preds"], zero_division=0))

    print("\nEvaluating on test set...")
    test_out = evaluate_split(model, X_test_vec, y_test, split_name="test")
    print("Test Accuracy:", round(test_out["metrics"]["test_accuracy"], 4))
    print(classification_report(y_test, test_out["preds"], zero_division=0))

    all_metrics = {
        **val_out["metrics"],
        **test_out["metrics"],
    }

    save_artifacts(
        model=model,
        vectorizer=vectorizer,
        metrics=all_metrics,
        test_df=test_df,
        test_preds=test_out["preds"],
        model_dir=model_dir
    )

    return {
        "metrics": all_metrics,
        "metrics_path": model_dir / "metrics.csv",
        "metrics_json_path": model_dir / "metrics.json",
        "predictions_path": model_dir / "test_predictions.csv",
        "model_path": model_dir / "model.joblib",
        "vectorizer_path": model_dir / "vectorizer.joblib",
        "model_dir": model_dir,
        "model": model,
        "vectorizer": vectorizer
    }


def main():
    out = train_model(run_training=True)
    print("\nSaved to:", out["model_dir"])


if __name__ == "__main__":
    main()