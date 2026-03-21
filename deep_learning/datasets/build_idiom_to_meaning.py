from pathlib import Path
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

SPLITS_DIR = PROJECT_ROOT / "data" / "splits" / "base"
OUTPUT_DIR = PROJECT_ROOT / "deep_learning" / "datasets" / "idiom_to_meaning"


def build_idiom_to_meaning_dataset(
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

        # =========================
        # Core dataset
        # =========================
        out = df[[
            "idiom_canonical",
            "idiom_canonical_meaning",
            "idiom_canonical_meaning_arabic",

            "idiom_register",
            "idiom_domain",
            "learner_difficulty",
            "ambiguity_flag",
            "idiom_compositionality_level",
            "contains_profanity",

            "semantic_consistency",
            "canonical_semantic_score"
        ]].copy()

        # =========================
        # Rename
        # =========================
        out = out.rename(columns={
            "idiom_canonical": "input_text",
            "idiom_canonical_meaning": "target_text",
            "idiom_canonical_meaning_arabic": "target_text_ar"
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
        # Quality filtering (light)
        # =========================
        if "semantic_consistency" in out.columns:
            out = out[out["semantic_consistency"] > 0.2]

        if "canonical_semantic_score" in out.columns:
            out = out[out["canonical_semantic_score"] > 0.2]

        # =========================
        # Keep best meaning per idiom
        # =========================
        sort_cols = []
        ascending = []

        if "canonical_semantic_score" in out.columns:
            sort_cols.append("canonical_semantic_score")
            ascending.append(False)

        if "semantic_consistency" in out.columns:
            sort_cols.append("semantic_consistency")
            ascending.append(False)

        if sort_cols:
            out = out.sort_values(sort_cols, ascending=ascending)

        out = out.drop_duplicates(subset=["input_text"], keep="first").reset_index(drop=True)

        print(f"{split_name}: {len(out):,} rows")

        if save:
            out.to_csv(output_dir / f"{split_name}.csv", index=False, encoding="utf-8-sig")

        output[split_name] = out

    print(f"\nSaved idiom → meaning dataset to: {output_dir}")

    return {
        "train_df": output["train"],
        "validation_df": output["validation"],
        "test_df": output["test"],
        "output_dir": output_dir
    }


def load_idiom_to_meaning_dataset(output_dir=OUTPUT_DIR):
    return (
        pd.read_csv(output_dir / "train.csv"),
        pd.read_csv(output_dir / "validation.csv"),
        pd.read_csv(output_dir / "test.csv")
    )


if __name__ == "__main__":
    build_idiom_to_meaning_dataset(save=True)