from pathlib import Path
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

DATASETS_DIR = PROJECT_ROOT / "deep_learning" / "datasets"


def summarize_dataset(name, path):
    df = pd.read_csv(path, low_memory=False)

    summary = {
        "dataset": name,
        "rows": len(df),
        "columns": len(df.columns),
        "unique_inputs": df["input_text"].nunique() if "input_text" in df.columns else None,
        "unique_targets": df["target_text"].nunique() if "target_text" in df.columns else None,
    }

    if "label" in df.columns:
        counts = df["label"].value_counts().to_dict()
        summary["label_0"] = counts.get(0, 0)
        summary["label_1"] = counts.get(1, 0)

    return summary


def build_dataset_inventory():
    print("Building dataset inventory...\n")

    datasets = {
        "idiom_detection_train": DATASETS_DIR / "idiom_detection" / "train.csv",
        "idiom_detection_validation": DATASETS_DIR / "idiom_detection" / "validation.csv",
        "idiom_detection_test": DATASETS_DIR / "idiom_detection" / "test.csv",

        "context_to_idiom_train": DATASETS_DIR / "context_to_idiom" / "train.csv",
        "context_to_idiom_validation": DATASETS_DIR / "context_to_idiom" / "validation.csv",
        "context_to_idiom_test": DATASETS_DIR / "context_to_idiom" / "test.csv",

        "arabic_context_to_idiom_train": DATASETS_DIR / "arabic_context_to_idiom" / "train.csv",
        "arabic_context_to_idiom_validation": DATASETS_DIR / "arabic_context_to_idiom" / "validation.csv",
        "arabic_context_to_idiom_test": DATASETS_DIR / "arabic_context_to_idiom" / "test.csv",

        "surface_normalization_train": DATASETS_DIR / "surface_normalization" / "train.csv",
        "surface_normalization_validation": DATASETS_DIR / "surface_normalization" / "validation.csv",
        "surface_normalization_test": DATASETS_DIR / "surface_normalization" / "test.csv",

        "idiom_to_meaning_train": DATASETS_DIR / "idiom_to_meaning" / "train.csv",
        "idiom_to_meaning_validation": DATASETS_DIR / "idiom_to_meaning" / "validation.csv",
        "idiom_to_meaning_test": DATASETS_DIR / "idiom_to_meaning" / "test.csv",
    }

    rows = []

    for name, path in datasets.items():
        if path.exists():
            summary = summarize_dataset(name, path)
            rows.append(summary)
            print(f"✔ {name}")
        else:
            print(f"✘ Missing: {name}")

    df_inventory = pd.DataFrame(rows)

    output_path = DATASETS_DIR / "dataset_inventory.csv"
    df_inventory.to_csv(output_path, index=False)

    print("\nSaved inventory to:", output_path)

    return df_inventory


if __name__ == "__main__":
    build_dataset_inventory()