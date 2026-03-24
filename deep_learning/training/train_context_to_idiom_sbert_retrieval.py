from pathlib import Path
import sys
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parents[2]
TRAINING_DIR = BASE_DIR / "deep_learning" / "training"

if str(TRAINING_DIR) not in sys.path:
    sys.path.append(str(TRAINING_DIR))

from tools import (
    ensure_dir,
    load_csv_checked,
    ensure_text_pair_columns,
    ensure_single_text_column,
    compute_topk_accuracy,
    compute_mrr,
)


DEFAULT_TEST_CSV = BASE_DIR / "deep_learning" / "datasets" / "context_to_idiom" / "test.csv"
DEFAULT_BANK_CSV = BASE_DIR / "deep_learning" / "datasets" / "idiom_bank" / "idiom_bank.csv"
DEFAULT_OUTPUT_DIR = BASE_DIR / "deep_learning" / "models" / "sbert_context_to_idiom_retrieval"
DEFAULT_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"


def run_sbert_retrieval(
    test_csv: Path = DEFAULT_TEST_CSV,
    bank_csv: Path = DEFAULT_BANK_CSV,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    model_name: str = DEFAULT_MODEL_NAME,
    top_k: int = 5,
    mrr_cutoff: int = 100,
    batch_size: int = 64,
    save_outputs: bool = True,
):
    """
    Run SBERT retrieval for Context -> Idiom.

    Returns
    -------
    dict
        {
            "metrics": dict,
            "predictions_df": pd.DataFrame
        }
    """

    output_dir = ensure_dir(Path(output_dir))

    print("Loading datasets...")
    test_df = load_csv_checked(Path(test_csv), low_memory=False)
    bank_df = load_csv_checked(Path(bank_csv), low_memory=False)

    # Clean input test data
    test_df = test_df[test_df["target_text"].notna()].copy()
    test_df = ensure_text_pair_columns(test_df, input_col="input_text", target_col="target_text")

    bank_df = bank_df[bank_df["idiom_canonical"].notna()].copy()
    bank_df = ensure_single_text_column(bank_df, col="idiom_canonical")

    # Build retrieval text dynamically from available bank columns
    for col in ["idiom_canonical_meaning", "representative_surface"]:
        if col not in bank_df.columns:
            bank_df[col] = ""

    bank_df["idiom_canonical_meaning"] = bank_df["idiom_canonical_meaning"].fillna("").astype(str).str.strip()
    bank_df["representative_surface"] = bank_df["representative_surface"].fillna("").astype(str).str.strip()

    bank_df["retrieval_text_en"] = (
            "Idiom: " + bank_df["idiom_canonical"].astype(str).str.strip() +
            ". Meaning: " + bank_df["idiom_canonical_meaning"] +
            ". Example form: " + bank_df["representative_surface"]
    ).str.strip()

    bank_df = ensure_single_text_column(bank_df, col="retrieval_text_en")

    # Remove duplicated idioms in bank if any
    bank_df = bank_df.drop_duplicates(subset=["idiom_canonical"]).reset_index(drop=True)

    print(f"Test samples: {len(test_df):,}")
    print(f"Idiom bank size: {len(bank_df):,}")

    print("Loading model:", model_name)
    model = SentenceTransformer(model_name)

    # Encode bank
    bank_texts = bank_df["retrieval_text_en"].tolist()
    bank_idioms = bank_df["idiom_canonical"].tolist()

    print("Encoding idiom bank...")
    bank_emb = model.encode(
        bank_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # Encode test contexts
    test_texts = test_df["input_text"].tolist()
    test_targets = test_df["target_text"].tolist()

    print("Encoding test contexts...")
    test_emb = model.encode(
        test_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    print("Computing similarities...")
    sim = cosine_similarity(test_emb, bank_emb)

    ranked_predictions = []
    rr_scores = []
    pred_rows = []

    for i in range(sim.shape[0]):
        scores = sim[i]
        ranked_idx = np.argsort(scores)[::-1]

        ranked_idioms_full = [bank_idioms[j] for j in ranked_idx]
        ranked_idioms_topk = ranked_idioms_full[:top_k]
        gold = test_targets[i]

        ranked_predictions.append(ranked_idioms_topk)

        rank_pos = None
        for rank, pred_idiom in enumerate(ranked_idioms_full[:mrr_cutoff], start=1):
            if pred_idiom == gold:
                rank_pos = rank
                break

        rr_scores.append(0.0 if rank_pos is None else 1.0 / rank_pos)

        pred_rows.append({
            "input_text": test_texts[i],
            "gold_idiom": gold,
            "pred_top1": ranked_idioms_topk[0] if len(ranked_idioms_topk) >= 1 else "",
            "pred_top3": " ||| ".join(ranked_idioms_topk[:3]),
            "pred_top5": " ||| ".join(ranked_idioms_topk[:5]),
        })

    metrics = {
        "top1_accuracy": compute_topk_accuracy(ranked_predictions, test_targets, k=1),
        "top3_accuracy": compute_topk_accuracy(ranked_predictions, test_targets, k=3),
        "top5_accuracy": compute_topk_accuracy(ranked_predictions, test_targets, k=5),
        "mrr_at_100": float(np.mean(rr_scores)),
        "num_test_samples": int(len(test_targets)),
        "num_bank_idioms": int(len(bank_idioms)),
        "model_name": model_name,
    }

    predictions_df = pd.DataFrame(pred_rows)

    print("\nResults:")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    if save_outputs:
        predictions_path = output_dir / "test_predictions.csv"
        metrics_path = output_dir / "metrics.csv"

        predictions_df.to_csv(predictions_path, index=False, encoding="utf-8-sig")
        pd.DataFrame([metrics]).to_csv(metrics_path, index=False, encoding="utf-8-sig")

        print("\nSaved predictions to:", predictions_path)
        print("Saved metrics to:", metrics_path)

    return {
        "metrics": metrics,
        "predictions_df": predictions_df,
    }

def main():
    run_sbert_retrieval()


if __name__ == "__main__":
    main()