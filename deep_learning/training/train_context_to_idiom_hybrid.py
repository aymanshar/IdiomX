from pathlib import Path
import sys
import pandas as pd
import numpy as np
import torch
from tqdm.auto import tqdm
from typing import Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.metrics import accuracy_score

BASE_DIR = Path(__file__).resolve().parents[2]
TRAINING_DIR = BASE_DIR / "deep_learning" / "training"

if str(TRAINING_DIR) not in sys.path:
    sys.path.append(str(TRAINING_DIR))

from tools import (
    ensure_dir,
    load_csv_checked,
    ensure_text_pair_columns,
    ensure_single_text_column,
)

DEFAULT_TEST_CSV = BASE_DIR / "deep_learning" / "datasets" / "context_to_idiom" / "test.csv"
DEFAULT_BANK_CSV = BASE_DIR / "deep_learning" / "datasets" / "idiom_bank" / "idiom_bank.csv"
DEFAULT_OUTPUT_DIR = BASE_DIR / "deep_learning" / "models" / "context_to_idiom_hybrid"

DEFAULT_SBERT_MODEL = "sentence-transformers/all-mpnet-base-v2"
DEFAULT_FLAN_MODEL = "google/flan-t5-base"


def build_retrieval_text(bank_df: pd.DataFrame, mode: str = "idiom_meaning_surface") -> pd.DataFrame:
    bank_df = bank_df.copy()

    for col in ["idiom_canonical", "idiom_canonical_meaning", "representative_surface"]:
        if col not in bank_df.columns:
            bank_df[col] = ""

    bank_df["idiom_canonical"] = bank_df["idiom_canonical"].fillna("").astype(str).str.strip()
    bank_df["idiom_canonical_meaning"] = bank_df["idiom_canonical_meaning"].fillna("").astype(str).str.strip()
    bank_df["representative_surface"] = bank_df["representative_surface"].fillna("").astype(str).str.strip()

    if mode == "meaning_only":
        bank_df["retrieval_text_en"] = bank_df["idiom_canonical_meaning"]

    elif mode == "idiom_meaning":
        bank_df["retrieval_text_en"] = (
            "Idiom: " + bank_df["idiom_canonical"] +
            ". Meaning: " + bank_df["idiom_canonical_meaning"]
        )

    elif mode == "meaning_surface":
        bank_df["retrieval_text_en"] = (
            "Meaning: " + bank_df["idiom_canonical_meaning"] +
            ". Example form: " + bank_df["representative_surface"]
        )

    elif mode == "idiom_meaning_surface":
        bank_df["retrieval_text_en"] = (
            "Idiom: " + bank_df["idiom_canonical"] +
            ". Meaning: " + bank_df["idiom_canonical_meaning"] +
            ". Example form: " + bank_df["representative_surface"]
        )

    else:
        raise ValueError(
            "Invalid retrieval_text_mode. Choose from: "
            "meaning_only, idiom_meaning, meaning_surface, idiom_meaning_surface"
        )

    bank_df["retrieval_text_en"] = bank_df["retrieval_text_en"].fillna("").astype(str).str.strip()
    return bank_df


def get_yes_probability(model, tokenizer, prompt: str, device: str) -> float:
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=256
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=2,
            return_dict_in_generate=True,
            output_scores=True
        )

    logits = outputs.scores[0]
    probs = torch.softmax(logits, dim=-1)

    yes_token_ids = tokenizer.encode("yes", add_special_tokens=False)
    if not yes_token_ids:
        return 0.0

    yes_token = yes_token_ids[0]
    return probs[0][yes_token].item()


def run_hybrid_model(
    test_csv: Path = DEFAULT_TEST_CSV,
    bank_csv: Path = DEFAULT_BANK_CSV,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    sbert_model_name: str = DEFAULT_SBERT_MODEL,
    flan_model_name: str = DEFAULT_FLAN_MODEL,
    retrieval_text_mode: str = "idiom_meaning_surface",
    top_k: int = 5,
    max_samples: Optional[int] = None,
    save_outputs: bool = True,
):
    output_dir = ensure_dir(Path(output_dir))

    print("Loading datasets...")
    test_df = load_csv_checked(Path(test_csv), low_memory=False)
    bank_df = load_csv_checked(Path(bank_csv), low_memory=False)

    test_df = ensure_text_pair_columns(test_df, input_col="input_text", target_col="target_text")

    bank_df = bank_df[bank_df["idiom_canonical"].notna()].copy()
    bank_df = ensure_single_text_column(bank_df, col="idiom_canonical")
    bank_df = build_retrieval_text(bank_df, mode=retrieval_text_mode)
    bank_df = ensure_single_text_column(bank_df, col="retrieval_text_en")
    bank_df = bank_df.drop_duplicates(subset=["idiom_canonical"]).reset_index(drop=True)

    test_contexts = test_df["input_text"].astype(str).tolist()
    gold_idioms = test_df["target_text"].astype(str).tolist()

    if max_samples is not None:
        test_contexts = test_contexts[:max_samples]
        gold_idioms = gold_idioms[:max_samples]

    bank_idioms = bank_df["idiom_canonical"].astype(str).tolist()
    bank_texts = bank_df["retrieval_text_en"].astype(str).tolist()

    print("Test samples:", len(test_contexts))
    print("Idiom bank size:", len(bank_idioms))
    print("Retrieval text mode:", retrieval_text_mode)

    # SBERT retrieval
    print("\nLoading SBERT retrieval model...")
    retrieval_model = SentenceTransformer(sbert_model_name)

    print("Encoding idiom bank...")
    bank_emb = retrieval_model.encode(
        bank_texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    print("Encoding contexts...")
    context_emb = retrieval_model.encode(
        test_contexts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    # FLAN reranker
    print("\nLoading FLAN-T5 reranker...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(flan_model_name, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(flan_model_name)
    model.to(device)
    model.eval()

    predictions = []
    pred_rows = []

    print("\nRunning hybrid retrieval + reranking...\n")

    for i, context in enumerate(tqdm(test_contexts, desc="Hybrid reranking", total=len(test_contexts))):
        scores = cosine_similarity(
            context_emb[i].reshape(1, -1),
            bank_emb
        )[0]

        top_idx = np.argsort(scores)[::-1][:top_k]

        # if SBERT confidence is high → skip FLAN
        top_score = scores[top_idx[0]]

        if top_score > 0.75:
            pred = bank_idioms[top_idx[0]]
            predictions.append(pred)

            pred_rows.append({
                "input_text": context,
                "gold_idiom": gold_idioms[i],
                "prediction": pred,
                "sbert_candidates": " ||| ".join([bank_idioms[j] for j in top_idx]),
                "reranker_scores": "SKIPPED_HIGH_CONF",
            })
            continue

        # otherwise build candidates and rerank with FLAN
        candidates = [bank_idioms[j] for j in top_idx]

        candidate_scores = []

        for cand in candidates:
            prompt = (
                f"Context: {context}\n\n"
                f"Candidate idiom: {cand}\n\n"
                "Is this idiom the best fit for the context? Answer yes or no."
            )

            yes_prob = get_yes_probability(model, tokenizer, prompt, device)
            candidate_scores.append(yes_prob)

        best_idx = int(np.argmax(candidate_scores))
        pred = candidates[best_idx]
        predictions.append(pred)

        pred_rows.append({
            "input_text": context,
            "gold_idiom": gold_idioms[i],
            "prediction": pred,
            "sbert_candidates": " ||| ".join(candidates),
            "reranker_scores": " ||| ".join([f"{x:.6f}" for x in candidate_scores]),
        })
        if (i + 1) % 200 == 0:
            print(f"Processed {i + 1}/{len(test_contexts)} samples")

    accuracy = accuracy_score(gold_idioms, predictions)

    metrics = {
        "exact_match_accuracy": float(accuracy),
        "num_test_samples": int(len(gold_idioms)),
        "num_bank_idioms": int(len(bank_idioms)),
        "top_k": int(top_k),
        "sbert_model_name": sbert_model_name,
        "flan_model_name": flan_model_name,
        "retrieval_text_mode": retrieval_text_mode,
    }

    print("\nHybrid Exact Match Accuracy:", accuracy)

    predictions_df = pd.DataFrame(pred_rows)

    if save_outputs:
        predictions_df.to_csv(
            output_dir / "test_predictions.csv",
            index=False,
            encoding="utf-8-sig"
        )

        pd.DataFrame([metrics]).to_csv(
            output_dir / "metrics.csv",
            index=False,
            encoding="utf-8-sig"
        )

        print("\nSaved predictions to:", output_dir / "test_predictions.csv")
        print("Saved metrics to:", output_dir / "metrics.csv")

    return {
        "metrics": metrics,
        "predictions_df": predictions_df,
    }


def main():
    run_hybrid_model()


if __name__ == "__main__":
    main()