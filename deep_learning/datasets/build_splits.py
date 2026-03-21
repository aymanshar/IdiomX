from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "idiomx_official.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "splits" / "base"

REQUIRED_COLUMNS = [
    "idiom",
    "idiom_canonical",
    "idiom_surface",
    "idiom_in_example",
    "idiom_in_example_arabic",
    "idiom_in_example_meaning_en",
    "idiom_in_example_meaning_arabic",
    "is_example_idiom",
    "example_usage_label",
]

OPTIONAL_METADATA_COLUMNS = [
    "meaning_en",
    "example",
    "source",
    "source_type",
    "pos",
    "tags",
    "idiom_confidence",
    "source_url",
    "idiom_canonical_meaning",
    "idiom_canonical_meaning_arabic",
    "is_idiom",
    "ambiguity_flag",
    "idiom_compositionality_level",
    "idiom_register",
    "idiom_domain",
    "learner_difficulty",
    "is_generated_example",
    "enrichment_model",
    "enrichment_version",
    "validation_status",
    "example_length",
    "idiom_present",
    "contains_profanity",
    "semantic_consistency",
    "canonical_semantic_score",
]


def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    keep_cols = REQUIRED_COLUMNS + [c for c in OPTIONAL_METADATA_COLUMNS if c in df.columns]
    df = df[keep_cols].copy()

    # Fill canonical key from idiom if missing
    df["idiom_canonical"] = df["idiom_canonical"].fillna(df["idiom"])

    # Clean text-like columns
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna("").astype(str).str.strip()

    # Required fields for base split
    df = df[df["idiom"] != ""]
    df = df[df["idiom_canonical"] != ""]
    df = df[df["idiom_surface"] != ""]
    df = df[df["idiom_in_example"] != ""]

    # Keep only trusted validation rows if the field exists
    if "validation_status" in df.columns:
        allowed_status = {"valid", "verified", "corrected"}
        df = df[df["validation_status"].isin(allowed_status)]

    # Normalize labels
    df["is_example_idiom"] = df["is_example_idiom"].astype(bool)
    df["example_usage_label"] = df["example_usage_label"].astype(str).str.lower().str.strip()

    # Drop exact duplicate context pairs
    df = df.drop_duplicates(subset=["idiom_canonical", "idiom_in_example"]).reset_index(drop=True)

    return df


def build_splits(
    input_path=INPUT_PATH,
    output_dir=OUTPUT_DIR,
    test_size=0.1,
    val_size=0.1,
    random_state=42,
    save=True
):
    input_path = Path(input_path)
    output_dir = Path(output_dir)

    print("Loading official IdiomX dataset...")
    df = pd.read_csv(input_path, low_memory=False)
    df = _clean_dataframe(df)

    print("Total usable rows :", len(df))
    print("Unique idioms     :", df["idiom_canonical"].nunique())

    idioms = sorted(df["idiom_canonical"].dropna().unique())

    train_idioms, temp_idioms = train_test_split(
        idioms,
        test_size=test_size + val_size,
        random_state=random_state
    )

    val_idioms, test_idioms = train_test_split(
        temp_idioms,
        test_size=test_size / (test_size + val_size),
        random_state=random_state
    )

    train_df = df[df["idiom_canonical"].isin(train_idioms)].copy()
    val_df = df[df["idiom_canonical"].isin(val_idioms)].copy()
    test_df = df[df["idiom_canonical"].isin(test_idioms)].copy()

    print("\nSplit sizes:")
    print("Train rows      :", len(train_df), "| idioms:", len(train_idioms))
    print("Validation rows :", len(val_df), "| idioms:", len(val_idioms))
    print("Test rows       :", len(test_df), "| idioms:", len(test_idioms))

    print("\nUsage-label balance:")
    for name, part in [("train", train_df), ("validation", val_df), ("test", test_df)]:
        print(name, part["example_usage_label"].value_counts(dropna=False).to_dict())

    shared_train_val = set(train_df["idiom_canonical"]) & set(val_df["idiom_canonical"])
    shared_train_test = set(train_df["idiom_canonical"]) & set(test_df["idiom_canonical"])
    shared_val_test = set(val_df["idiom_canonical"]) & set(test_df["idiom_canonical"])

    print("\nOverlap checks:")
    print("Train ∩ Validation :", len(shared_train_val))
    print("Train ∩ Test       :", len(shared_train_test))
    print("Validation ∩ Test  :", len(shared_val_test))

    if save:
        output_dir.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(output_dir / "train.csv", index=False, encoding="utf-8-sig")
        val_df.to_csv(output_dir / "validation.csv", index=False, encoding="utf-8-sig")
        test_df.to_csv(output_dir / "test.csv", index=False, encoding="utf-8-sig")
        print("\nSaved base splits to:", output_dir)

    return {
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "train_idioms": len(train_idioms),
        "val_idioms": len(val_idioms),
        "test_idioms": len(test_idioms),
        "output_dir": output_dir
    }


def load_splits(output_dir=OUTPUT_DIR):
    output_dir = Path(output_dir)
    train_df = pd.read_csv(output_dir / "train.csv", low_memory=False)
    val_df = pd.read_csv(output_dir / "validation.csv", low_memory=False)
    test_df = pd.read_csv(output_dir / "test.csv", low_memory=False)
    return train_df, val_df, test_df


if __name__ == "__main__":
    build_splits(save=True)