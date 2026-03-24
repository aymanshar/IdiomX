from pathlib import Path
import sys
import re
from difflib import SequenceMatcher
from typing import Optional, Dict, List

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

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
DEFAULT_OUTPUT_DIR = BASE_DIR / "deep_learning" / "models" / "context_to_idiom_hybrid_flan_first"

DEFAULT_SBERT_MODEL = "sentence-transformers/all-mpnet-base-v2"
DEFAULT_FLAN_MODEL = "google/flan-t5-base"
DEFAULT_PROMPT_PREFIX = "Given the context, generate the best English idiom: "


def normalize_text(text: str) -> str:
    """
    Normalize idiom text for comparison.
    Keeps content readable while reducing punctuation/case variation.
    """
    text = str(text).strip().lower()
    text = text.replace("’", "'").replace("`", "'")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[\"“”]", "", text)
    text = re.sub(r"\s*-\s*", "-", text)
    text = text.strip(" .,!?:;")
    return text


def similarity_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()


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


def generate_flan_predictions(
    contexts: List[str],
    model_name: str,
    prompt_prefix: str = DEFAULT_PROMPT_PREFIX,
    max_input: int = 128,
    max_new_tokens: int = 16,
    num_beams: int = 4,
    batch_size: Optional[int] = None,
) -> List[str]:
    """
    Generate idiom predictions using FLAN in batches.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\nLoading FLAN generator:", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.to(device)
    model.eval()

    if batch_size is None:
        batch_size = 8 if torch.cuda.is_available() else 2

    prompts = [prompt_prefix + x for x in contexts]
    predictions = []

    print("\nGenerating FLAN predictions...")
    for start in tqdm(range(0, len(prompts), batch_size), desc="FLAN generation"):
        batch_texts = prompts[start:start + batch_size]

        enc = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_input,
            return_tensors="pt"
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                max_new_tokens=max_new_tokens,
                num_beams=num_beams
            )

        batch_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        predictions.extend([p.strip() for p in batch_preds])

    return predictions


def run_hybrid_model_flan_first(
    test_csv: Path = DEFAULT_TEST_CSV,
    bank_csv: Path = DEFAULT_BANK_CSV,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    sbert_model_name: str = DEFAULT_SBERT_MODEL,
    flan_model_name: str = DEFAULT_FLAN_MODEL,
    retrieval_text_mode: str = "idiom_meaning_surface",
    top_k: int = 5,
    fuzzy_threshold: float = 0.84,
    flan_predictions_csv: Optional[Path] = None,
    max_samples: Optional[int] = None,
    save_outputs: bool = True,
):
    """
    Improved hybrid:
    1) FLAN generates idiom directly
    2) If FLAN output exactly matches bank idiom -> keep it
    3) Else use SBERT Top-K candidates to canonicalize/correct FLAN output
    4) If no strong match -> fallback to SBERT Top-1

    This is faster and usually stronger than SBERT-first reranking.
    """
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
    print("Top-K:", top_k)
    print("Fuzzy threshold:", fuzzy_threshold)

    # Build normalized bank lookup
    bank_norm_to_canonical: Dict[str, str] = {}
    for idiom in bank_idioms:
        norm = normalize_text(idiom)
        if norm not in bank_norm_to_canonical:
            bank_norm_to_canonical[norm] = idiom

    # ----------------------------
    # SBERT retrieval setup
    # ----------------------------
    print("\nLoading SBERT retrieval model:", sbert_model_name)
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

    # ----------------------------
    # FLAN predictions
    # ----------------------------
    if flan_predictions_csv is not None and Path(flan_predictions_csv).exists():
        print("\nLoading cached FLAN predictions from:", flan_predictions_csv)
        flan_df = pd.read_csv(flan_predictions_csv)

        required_cols = {"input_text", "gold_idiom", "prediction"}
        missing_cols = required_cols - set(flan_df.columns)
        if missing_cols:
            raise ValueError(f"Missing columns in FLAN prediction file: {missing_cols}")

        flan_df = flan_df.iloc[:len(test_contexts)].copy()

        cached_inputs = flan_df["input_text"].astype(str).tolist()
        if cached_inputs != test_contexts:
            print("Warning: cached FLAN input order differs from current test order. Using row order only.")

        flan_predictions = flan_df["prediction"].astype(str).tolist()

    else:
        flan_predictions = generate_flan_predictions(
            contexts=test_contexts,
            model_name=flan_model_name
        )

    # ----------------------------
    # Hybrid decision
    # ----------------------------
    predictions = []
    pred_rows = []

    decision_counts = {
        "flan_exact_bank": 0,
        "flan_fuzzy_to_sbert_candidate": 0,
        "sbert_fallback": 0,
    }

    print("\nRunning FLAN-first hybrid decision...\n")

    sbert_top1_predictions = []

    for i, context in enumerate(tqdm(test_contexts, desc="Hybrid FLAN-first", total=len(test_contexts))):
        gold = gold_idioms[i]
        flan_pred = flan_predictions[i]
        flan_norm = normalize_text(flan_pred)

        # SBERT candidate shortlist from context
        scores = cosine_similarity(
            context_emb[i].reshape(1, -1),
            bank_emb
        )[0]

        top_idx = np.argsort(scores)[::-1][:top_k]
        candidates = [bank_idioms[j] for j in top_idx]
        candidate_norms = [normalize_text(c) for c in candidates]

        sbert_top1_predictions.append(candidates[0])

        # Case 1: FLAN exact normalized match to bank
        if flan_norm in bank_norm_to_canonical:
            final_pred = bank_norm_to_canonical[flan_norm]
            decision = "flan_exact_bank"
            decision_counts[decision] += 1

        else:
            # Case 2: Fuzzy canonicalization against SBERT candidate shortlist
            ratios = [similarity_ratio(flan_norm, cand_norm) for cand_norm in candidate_norms]
            best_idx = int(np.argmax(ratios))
            best_ratio = float(ratios[best_idx])

            if best_ratio >= fuzzy_threshold:
                final_pred = candidates[best_idx]
                decision = "flan_fuzzy_to_sbert_candidate"
                decision_counts[decision] += 1
            else:
                # Case 3: fallback to SBERT Top-1
                final_pred = candidates[0]
                decision = "sbert_fallback"
                decision_counts[decision] += 1

        predictions.append(final_pred)

        pred_rows.append({
            "input_text": context,
            "gold_idiom": gold,
            "flan_prediction": flan_pred,
            "final_prediction": final_pred,
            "decision_type": decision,
            "sbert_top1": candidates[0],
            "sbert_candidates": " ||| ".join(candidates),
        })

        if (i + 1) % 500 == 0:
            print(f"Processed {i + 1}/{len(test_contexts)} samples")

    # ----------------------------
    # Metrics
    # ----------------------------
    hybrid_accuracy = accuracy_score(gold_idioms, predictions)
    flan_accuracy = accuracy_score(gold_idioms, flan_predictions)


    sbert_top1_accuracy = accuracy_score(gold_idioms, sbert_top1_predictions)
    print("\nComputing final metrics...")
    metrics = {
        "hybrid_exact_match_accuracy": float(hybrid_accuracy),
        "flan_exact_match_accuracy": float(flan_accuracy),
        "sbert_top1_accuracy": float(sbert_top1_accuracy),
        "num_test_samples": int(len(gold_idioms)),
        "num_bank_idioms": int(len(bank_idioms)),
        "top_k": int(top_k),
        "fuzzy_threshold": float(fuzzy_threshold),
        "sbert_model_name": sbert_model_name,
        "flan_model_name": flan_model_name,
        "retrieval_text_mode": retrieval_text_mode,
        "decision_flan_exact_bank": int(decision_counts["flan_exact_bank"]),
        "decision_flan_fuzzy_to_sbert_candidate": int(decision_counts["flan_fuzzy_to_sbert_candidate"]),
        "decision_sbert_fallback": int(decision_counts["sbert_fallback"]),
    }

    print("\nResults:")
    for k, v in metrics.items():
        print(f"{k}: {v}")

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
    run_hybrid_model_flan_first()


if __name__ == "__main__":
    main()