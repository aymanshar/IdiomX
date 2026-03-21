from pathlib import Path
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

SPLITS_DIR = PROJECT_ROOT / "data" / "splits" / "base"
OUTPUT_DIR = PROJECT_ROOT / "deep_learning" / "datasets" / "idiom_detection"


def build_idiom_detection_dataset(
    splits_dir=SPLITS_DIR,
    output_dir=OUTPUT_DIR,
    save=True
):
    files = {
        "train": splits_dir / "train.csv",
        "validation": splits_dir / "validation.csv",
        "test": splits_dir / "test.csv",
    }

    output = {}

    if save:
        output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, path in files.items():
        print(f"\nProcessing {split_name}...")

        df = pd.read_csv(path, low_memory=False)

        # ===============================
        # Select relevant columns
        # ===============================
        out = df[[
            "idiom_in_example",
            "is_example_idiom",
            "idiom_canonical",
            "idiom_surface",
            "idiom_canonical_meaning",
            "idiom_canonical_meaning_arabic",
            "example_usage_label",
            "ambiguity_flag",
            "idiom_compositionality_level",
            "idiom_register",
            "idiom_domain",
            "learner_difficulty",
            "contains_profanity",
            "example_length",
            "semantic_consistency",
            "canonical_semantic_score"
        ]].copy()

        # ===============================
        # Rename for ML standard
        # ===============================
        out = out.rename(columns={
            "idiom_in_example": "input_text",
            "is_example_idiom": "label",
            "idiom_canonical_meaning": "meaning_en",
            "idiom_canonical_meaning_arabic": "meaning_ar"
        })

        # ===============================
        # Clean
        # ===============================
        for col in out.columns:
            if out[col].dtype == "object":
                out[col] = out[col].fillna("").astype(str).str.strip()

        out = out[out["input_text"] != ""]
        out["label"] = out["label"].astype(bool).astype(int)

        # ===============================
        # Keep only valid usage labels
        # ===============================
        out = out[out["example_usage_label"].isin(["idiomatic", "literal"])]

        # ===============================
        # Remove duplicates
        # ===============================
        out = out.drop_duplicates(subset=["input_text", "label"]).reset_index(drop=True)

        print(f"{split_name}: {len(out):,} rows")
        print("Label distribution:", out["label"].value_counts().to_dict())

        if save:
            out.to_csv(output_dir / f"{split_name}.csv", index=False, encoding="utf-8-sig")

        output[split_name] = out

    print(f"\nSaved idiom detection dataset to: {output_dir}")

    return {
        "train_df": output["train"],
        "validation_df": output["validation"],
        "test_df": output["test"],
        "output_dir": output_dir
    }


def load_idiom_detection_dataset(output_dir=OUTPUT_DIR):
    return (
        pd.read_csv(output_dir / "train.csv"),
        pd.read_csv(output_dir / "validation.csv"),
        pd.read_csv(output_dir / "test.csv")
    )


if __name__ == "__main__":
    build_idiom_detection_dataset(save=True)