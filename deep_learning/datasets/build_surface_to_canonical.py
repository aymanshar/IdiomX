from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "idiomx_official.csv"
OUTPUT_DIR = PROJECT_ROOT / "deep_learning" / "datasets" / "surface_normalization"


def build_surface_to_canonical_dataset(
    input_path=INPUT_PATH,
    output_dir=OUTPUT_DIR,
    save=True
):
    print("Loading official IdiomX dataset for surface split...")

    df = pd.read_csv(input_path, low_memory=False)

    # ===============================
    # Clean
    # ===============================
    df["idiom_surface"] = df["idiom_surface"].fillna("").astype(str).str.strip()
    df["idiom_canonical"] = df["idiom_canonical"].fillna(df["idiom"])

    df = df[df["idiom_surface"] != ""]
    df = df[df["idiom_canonical"] != ""]

    # ===============================
    # Split by surface (NOT canonical)
    # ===============================
    surfaces = df["idiom_surface"].unique()

    train_s, temp_s = train_test_split(surfaces, test_size=0.2, random_state=42)
    val_s, test_s = train_test_split(temp_s, test_size=0.5, random_state=42)

    train_df = df[df["idiom_surface"].isin(train_s)].copy()
    val_df = df[df["idiom_surface"].isin(val_s)].copy()
    test_df = df[df["idiom_surface"].isin(test_s)].copy()

    files = {
        "train": train_df,
        "validation": val_df,
        "test": test_df,
    }

    output = {}

    if save:
        output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, df_part in files.items():
        print(f"\nProcessing {split_name}...")

        out = df_part[[
            "idiom_surface",
            "idiom_canonical",
            "idiom_canonical_meaning",
            "idiom_canonical_meaning_arabic",
            "example_usage_label",
            "is_example_idiom",
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

        # Rename to standard ML format
        out = out.rename(columns={
            "idiom_surface": "input_text",
            "idiom_canonical": "target_text",
            "idiom_canonical_meaning": "meaning_en",
            "idiom_canonical_meaning_arabic": "meaning_ar"
        })

        # Clean text
        for col in out.columns:
            if out[col].dtype == "object":
                out[col] = out[col].fillna("").astype(str).str.strip()

        out = out[out["input_text"] != ""]
        out = out[out["target_text"] != ""]

        # ===============================
        # Keep best mapping per surface
        # ===============================
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

        # Keep best canonical per surface
        out = out.drop_duplicates(subset=["input_text"], keep="first").reset_index(drop=True)

        print(f"{split_name}: {len(out):,} rows")

        if save:
            out.to_csv(output_dir / f"{split_name}.csv", index=False, encoding="utf-8-sig")

        output[split_name] = out

    print(f"\nSaved surface dataset to: {output_dir}")

    return {
        "train_df": output["train"],
        "validation_df": output["validation"],
        "test_df": output["test"],
        "output_dir": output_dir
    }


def load_surface_to_canonical_dataset(output_dir=OUTPUT_DIR):
    return (
        pd.read_csv(output_dir / "train.csv"),
        pd.read_csv(output_dir / "validation.csv"),
        pd.read_csv(output_dir / "test.csv")
    )


if __name__ == "__main__":
    build_surface_to_canonical_dataset(save=True)