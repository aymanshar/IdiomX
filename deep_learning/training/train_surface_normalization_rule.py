from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]

TRAIN_CSV = BASE_DIR / "deep_learning/datasets/surface_to_canonical/train.csv"
VAL_CSV = BASE_DIR / "deep_learning/datasets/surface_to_canonical/validation.csv"
TEST_CSV = BASE_DIR / "deep_learning/datasets/surface_to_canonical/test.csv"

OUTPUT_DIR = BASE_DIR / "deep_learning/models/surface_normalization_rule"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading datasets...")

train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)
test_df = pd.read_csv(TEST_CSV)

# Clean
for df in [train_df, val_df, test_df]:
    df["input_text"] = df["input_text"].astype(str).str.strip().str.lower()
    df["target_text"] = df["target_text"].astype(str).str.strip().str.lower()

print("Building surface → canonical dictionary...")

surface_to_canonical = (
    train_df.groupby("input_text")["target_text"]
    .agg(lambda x: x.value_counts().index[0])
    .to_dict()
)

print("Dictionary size:", len(surface_to_canonical))


def evaluate(df):

    correct = 0
    preds = []

    for _, row in df.iterrows():

        surface = row["input_text"]
        gold = row["target_text"]

        pred = surface_to_canonical.get(surface, None)

        if pred == gold:
            correct += 1

        preds.append({
            "surface": surface,
            "gold": gold,
            "pred": pred
        })

    acc = correct / len(df)

    return acc, preds


print("\nEvaluating validation set...")
val_acc, val_preds = evaluate(val_df)

print("Validation Accuracy:", val_acc)

print("\nEvaluating test set...")
test_acc, test_preds = evaluate(test_df)

print("Test Accuracy:", test_acc)

pd.DataFrame(test_preds).to_csv(
    OUTPUT_DIR / "test_predictions.csv",
    index=False,
    encoding="utf-8-sig"
)

pd.DataFrame([{
    "validation_accuracy": val_acc,
    "test_accuracy": test_acc,
    "dictionary_size": len(surface_to_canonical)
}]).to_csv(
    OUTPUT_DIR / "metrics.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nSaved predictions to:", OUTPUT_DIR / "test_predictions.csv")
print("Saved metrics to:", OUTPUT_DIR / "metrics.csv")