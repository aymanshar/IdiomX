from pathlib import Path
import json
import itertools
import random

import joblib
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


# ============================================================
# Paths
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

DATA_DIR = PROJECT_ROOT / "deep_learning" / "datasets" / "idiom_detection"
MODEL_DIR = PROJECT_ROOT / "deep_learning" / "models" / "idiom_detection_baseline_tuned"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Reproducibility
# ============================================================

SEED = 42

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)


# ============================================================
# Data Loading
# ============================================================

def load_data():
    train = pd.read_csv(DATA_DIR / "train.csv")
    val = pd.read_csv(DATA_DIR / "validation.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    return train, val, test


# ============================================================
# Evaluation
# ============================================================

def evaluate_predictions(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


# ============================================================
# Main Tuning
# ============================================================

def train_model():
    set_seed()

    print("Loading dataset...")
    train_df, val_df, test_df = load_data()

    X_train = train_df["input_text"].astype(str)
    y_train = train_df["label"].astype(int)

    X_val = val_df["input_text"].astype(str)
    y_val = val_df["label"].astype(int)

    X_test = test_df["input_text"].astype(str)
    y_test = test_df["label"].astype(int)

    # --------------------------------------------------------
    # Search space
    # --------------------------------------------------------
    search_space = {
        "C": [0.5, 1.0, 2.0, 5.0],
        "ngram_range": [(1, 1), (1, 2), (1, 3)],
        "max_features": [20000, 30000, 50000],
        "class_weight": [None, "balanced"],
    }

    all_configs = list(itertools.product(
        search_space["C"],
        search_space["ngram_range"],
        search_space["max_features"],
        search_space["class_weight"]
    ))

    print(f"Total configs to test: {len(all_configs)}")

    results = []
    best_f1 = -1
    best_run = None
    best_model = None
    best_vectorizer = None

    for i, (C, ngram_range, max_features, class_weight) in enumerate(all_configs, start=1):
        print("\n" + "=" * 80)
        print(f"Run {i}/{len(all_configs)}")
        print(f"C={C}, ngram_range={ngram_range}, max_features={max_features}, class_weight={class_weight}")

        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=2,
            max_df=0.95,
            sublinear_tf=True
        )

        model = LogisticRegression(
            C=C,
            max_iter=1500,
            class_weight=class_weight,
            n_jobs=-1,
            random_state=SEED
        )

        X_train_vec = vectorizer.fit_transform(X_train)
        X_val_vec = vectorizer.transform(X_val)

        model.fit(X_train_vec, y_train)
        val_preds = model.predict(X_val_vec)

        val_metrics = evaluate_predictions(y_val, val_preds)

        row = {
            "run_id": i,
            "C": C,
            "ngram_range": str(ngram_range),
            "max_features": max_features,
            "class_weight": str(class_weight),
            "validation_accuracy": val_metrics["accuracy"],
            "validation_precision": val_metrics["precision"],
            "validation_recall": val_metrics["recall"],
            "validation_f1": val_metrics["f1"],
            "validation_macro_f1": val_metrics["macro_f1"],
            "validation_weighted_f1": val_metrics["weighted_f1"],
        }
        results.append(row)

        print("Validation metrics:")
        print({k: round(v, 4) for k, v in val_metrics.items()})

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_run = row
            best_model = model
            best_vectorizer = vectorizer

    # --------------------------------------------------------
    # Save tuning table
    # --------------------------------------------------------
    results_df = pd.DataFrame(results).sort_values(
        by=["validation_f1", "validation_accuracy"],
        ascending=False
    ).reset_index(drop=True)

    tuning_csv = MODEL_DIR / "tuning_results.csv"
    results_df.to_csv(tuning_csv, index=False, encoding="utf-8-sig")

    print("\nBest validation config:")
    print(best_run)

    # --------------------------------------------------------
    # Final test evaluation using best model
    # --------------------------------------------------------
    X_test_vec = best_vectorizer.transform(X_test)
    test_preds = best_model.predict(X_test_vec)
    test_metrics = evaluate_predictions(y_test, test_preds)

    print("\nTest metrics for best config:")
    print({k: round(v, 4) for k, v in test_metrics.items()})
    print("\nClassification report:")
    print(classification_report(y_test, test_preds, zero_division=0))

    # --------------------------------------------------------
    # Save artifacts
    # --------------------------------------------------------
    joblib.dump(best_model, MODEL_DIR / "best_model.joblib")
    joblib.dump(best_vectorizer, MODEL_DIR / "best_vectorizer.joblib")

    best_summary = {
        **best_run,
        "test_accuracy": test_metrics["accuracy"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_f1": test_metrics["f1"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_weighted_f1": test_metrics["weighted_f1"],
    }

    with open(MODEL_DIR / "best_config.json", "w", encoding="utf-8") as f:
        json.dump(best_summary, f, indent=2)

    pd.DataFrame([best_summary]).to_csv(
        MODEL_DIR / "best_metrics.csv",
        index=False,
        encoding="utf-8-sig"
    )

    preds_df = test_df.copy()
    preds_df["pred"] = test_preds
    preds_df.to_csv(MODEL_DIR / "test_predictions.csv", index=False, encoding="utf-8-sig")

    return {
        "results_df": results_df,
        "best_config": best_summary,
        "tuning_results_path": tuning_csv,
        "best_metrics_path": MODEL_DIR / "best_metrics.csv",
        "best_config_path": MODEL_DIR / "best_config.json",
        "predictions_path": MODEL_DIR / "test_predictions.csv",
        "model_dir": MODEL_DIR,
    }


if __name__ == "__main__":
    train_model()