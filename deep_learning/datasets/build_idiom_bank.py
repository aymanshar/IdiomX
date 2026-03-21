from pathlib import Path
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "idiomx_official.csv"
OUTPUT_DIR = PROJECT_ROOT / "deep_learning" / "datasets" / "idiom_bank"


def build_idiom_bank_dataset(
    input_path=INPUT_PATH,
    output_dir=OUTPUT_DIR,
    save=True
):
    print("Loading official IdiomX dataset...")
    df = pd.read_csv(input_path, low_memory=False)

    print(f"Total rows: {len(df):,}")

    # Clean text columns
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna("").astype(str).str.strip()

    # Fix canonical form if missing
    if "idiom_canonical" in df.columns and "idiom" in df.columns:
        df["idiom_canonical"] = df["idiom_canonical"].replace("", pd.NA)
        df["idiom_canonical"] = df["idiom_canonical"].fillna(df["idiom"])

    df = df[df["idiom_canonical"] != ""].copy()

    print("Grouping by canonical idioms...")

    agg_df = df.groupby("idiom_canonical").agg({
        "idiom_canonical_meaning": "first",
        "idiom_canonical_meaning_arabic": "first",
        "idiom_surface": lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0],
        "ambiguity_flag": "first",
        "idiom_compositionality_level": "first",
        "idiom_register": "first",
        "idiom_domain": "first",
        "learner_difficulty": "first",
        "contains_profanity": "max",
        "example_length": "mean",
        "semantic_consistency": "mean",
        "canonical_semantic_score": "mean"
    }).reset_index()

    agg_df = agg_df.rename(columns={
        "idiom_surface": "representative_surface",
        "example_length": "avg_example_length",
        "semantic_consistency": "avg_semantic_consistency",
        "canonical_semantic_score": "avg_canonical_semantic_score"
    })

    agg_df = agg_df.drop_duplicates(subset=["idiom_canonical"]).reset_index(drop=True)

    print(f"Unique idioms: {len(agg_df):,}")

    if save:
        output_dir.mkdir(parents=True, exist_ok=True)
        agg_df.to_csv(output_dir / "idiom_bank.csv", index=False, encoding="utf-8-sig")
        print(f"Saved idiom bank to: {output_dir}")

    return {
        "idiom_bank": agg_df,
        "output_dir": output_dir
    }


def load_idiom_bank(output_dir=OUTPUT_DIR):
    return pd.read_csv(Path(output_dir) / "idiom_bank.csv")


if __name__ == "__main__":
    build_idiom_bank_dataset(save=True)