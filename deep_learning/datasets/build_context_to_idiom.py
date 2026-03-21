from pathlib import Path
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

SPLITS_DIR = PROJECT_ROOT / "data" / "splits" / "base"
OUTPUT_DIR = PROJECT_ROOT / "deep_learning" / "datasets" / "context_to_idiom"


def build_context_to_idiom_dataset(
    splits_dir=SPLITS_DIR,
    output_dir=OUTPUT_DIR,
    save=True,
    mode="full"   # options: full, soft, strict
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

        # =========================
        # Core dataset
        # =========================
        out = df[[
            "idiom_in_example",
            "idiom_canonical",

            "idiom_canonical_meaning",
            "idiom_canonical_meaning_arabic",

            "example_usage_label",
            "is_example_idiom",

            "semantic_consistency",
            "canonical_semantic_score",
            "example_length",
            "contains_profanity",
            "ambiguity_flag",
            "idiom_compositionality_level",
            "learner_difficulty",
        ]].copy()

        # =========================
        # Rename for model
        # =========================
        out = out.rename(columns={
            "idiom_in_example": "input_text",
            "idiom_canonical": "target_text",
            "idiom_canonical_meaning": "meaning_en",
            "idiom_canonical_meaning_arabic": "meaning_ar"
        })

        # =========================
        # Clean
        # =========================
        for col in out.columns:
            if out[col].dtype == "object":
                out[col] = out[col].fillna("").astype(str).str.strip()

        out = out[out["input_text"] != ""]
        out = out[out["target_text"] != ""]

        # =========================
        # QUALITY FILTER
        # =========================
        if mode == "strict":
            out = out[out["semantic_consistency"] > 0.5]
            out = out[out["canonical_semantic_score"] > 0.5]

        elif mode == "soft":
            out = out[out["semantic_consistency"] > 0.2]
            out = out[out["canonical_semantic_score"] > 0.2]

        elif mode == "full":
            pass

        # =========================
        # Remove duplicates
        # =========================
        out = out.drop_duplicates(subset=["input_text", "target_text"]).reset_index(drop=True)

        print(f"{split_name}: {len(out):,} rows")

        if save:
            out.to_csv(output_dir / f"{split_name}.csv", index=False, encoding="utf-8-sig")

        output[split_name] = out

    print(f"\nSaved context → idiom dataset to: {output_dir}")

    return {
        "train_df": output["train"],
        "validation_df": output["validation"],
        "test_df": output["test"],
        "output_dir": output_dir
    }


def load_context_to_idiom_dataset(output_dir=OUTPUT_DIR):
    return (
        pd.read_csv(output_dir / "train.csv"),
        pd.read_csv(output_dir / "validation.csv"),
        pd.read_csv(output_dir / "test.csv")
    )


if __name__ == "__main__":
    build_context_to_idiom_dataset(save=True)