from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from difflib import SequenceMatcher
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, MT5ForConditionalGeneration
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# CONFIG
# =========================================================
from pathlib import Path
from dataclasses import dataclass

PROJECT_ROOT = Path(__file__).resolve().parents[2]

@dataclass
class Config:
    model_dir: Path = PROJECT_ROOT / "deep_learning" / "models" / "arabic_context_to_idiom_mt5_base_normalized" / "checkpoint-2108"
    output_dir: Path = PROJECT_ROOT / "deep_learning" / "models" / "task3_hybrid_results"

    train_csv: Path = PROJECT_ROOT / "deep_learning" / "datasets" / "arabic_context_to_idiom" / "train.csv"
    val_csv: Path = PROJECT_ROOT / "deep_learning" / "datasets" / "arabic_context_to_idiom" / "validation.csv"
    test_csv: Path = PROJECT_ROOT / "deep_learning" / "datasets" / "arabic_context_to_idiom" / "test.csv"

    input_col: str = "input_text"
    target_col: str = "target_text"
    idiom_flag_col: str = "is_example_idiom"

    max_source_length: int = 128
    max_target_length: int = 20

    num_beams: int = 6

    # scoring weights
    w_fuzzy: float = 0.4
    w_semantic: float = 0.6

    top_k_candidates: int = 20


CFG = Config()


# =========================================================
# UTILS
# =========================================================
def normalize_text(x):
    return str(x).strip().lower()


def fuzzy_score(a, b):
    return SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()


def exact_match(pred, gold):
    return normalize_text(pred) == normalize_text(gold)


# =========================================================
# LOAD
# =========================================================
def load_data():
    df_train = pd.read_csv(CFG.train_csv)
    df_val = pd.read_csv(CFG.val_csv)
    df_test = pd.read_csv(CFG.test_csv)

    df_train = df_train[df_train[CFG.idiom_flag_col] == 1].copy()
    df_val = df_val[df_val[CFG.idiom_flag_col] == 1].copy()
    df_test = df_test[df_test[CFG.idiom_flag_col] == 1].copy()

    for df in (df_train, df_val, df_test):
        df[CFG.input_col] = df[CFG.input_col].astype(str)
        df[CFG.target_col] = df[CFG.target_col].astype(str)

    return df_train, df_val, df_test


def build_idiom_bank(df_train, df_val, df_test):
    all_targets = pd.concat(
        [
            df_train[CFG.target_col],
            df_val[CFG.target_col],
            df_test[CFG.target_col],
        ],
        axis=0,
        ignore_index=True,
    )

    idioms = sorted(all_targets.dropna().astype(str).str.strip().unique().tolist())
    return idioms


# =========================================================
# GENERATION
# =========================================================
def generate_idiom(model, tokenizer, text):
    device = next(model.parameters()).device

    input_text = "Predict the correct English idiom: " + text

    encoded = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=CFG.max_source_length,
    ).to(device)

    with torch.no_grad():
        output = model.generate(
            **encoded,
            max_length=CFG.max_target_length,
            num_beams=CFG.num_beams,
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)


# =========================================================
# HYBRID MATCHING
# =========================================================
def hybrid_match(prediction: str, idiom_bank: List[str], sbert_model) -> Tuple[str, float]:
    # Step 1: fuzzy shortlist
    scores = [(idiom, fuzzy_score(prediction, idiom)) for idiom in idiom_bank]
    scores.sort(key=lambda x: x[1], reverse=True)

    top_candidates = [idiom for idiom, _ in scores[:CFG.top_k_candidates]]

    # Step 2: semantic scoring
    pred_emb = sbert_model.encode([prediction])
    cand_emb = sbert_model.encode(top_candidates)

    sim_scores = cosine_similarity(pred_emb, cand_emb)[0]

    # Step 3: combine
    best_idiom = prediction
    best_score = -1

    for i, idiom in enumerate(top_candidates):
        f = fuzzy_score(prediction, idiom)
        s = sim_scores[i]

        combined = CFG.w_fuzzy * f + CFG.w_semantic * s

        if combined > best_score:
            best_score = combined
            best_idiom = idiom

    return best_idiom, best_score


# =========================================================
# MAIN
# =========================================================
def run():
    CFG.output_dir.mkdir(parents=True, exist_ok=True)

    model_path = str(CFG.model_dir.resolve())
    test_path = str(CFG.test_csv.resolve())
    print("Model path:", model_path)
    print("Exists:", CFG.model_dir.exists())

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, local_files_only=True)
    model = MT5ForConditionalGeneration.from_pretrained(model_path, local_files_only=True)
    model.eval()

    if torch.cuda.is_available():
        model.to("cuda")

    print("Loading SBERT...")
    sbert = SentenceTransformer("all-MiniLM-L6-v2")

    print("Loading data...")
    df_train, df_val, df_test = load_data()
    idiom_bank = build_idiom_bank(df_train, df_val, df_test)

    df = df_test

    print("Idiom bank size:", len(idiom_bank))

    results = []

    correct_raw = 0
    correct_hybrid = 0

    for i, row in df.iterrows():
        input_text = row[CFG.input_col]
        gold = row[CFG.target_col]

        raw_pred = generate_idiom(model, tokenizer, input_text)
        hybrid_pred, score = hybrid_match(raw_pred, idiom_bank, sbert)

        raw_ok = exact_match(raw_pred, gold)
        hybrid_ok = exact_match(hybrid_pred, gold)

        correct_raw += raw_ok
        correct_hybrid += hybrid_ok

        results.append({
            "input": input_text,
            "gold": gold,
            "raw_prediction": raw_pred,
            "hybrid_prediction": hybrid_pred,
            "score": score,
            "raw_correct": raw_ok,
            "hybrid_correct": hybrid_ok,
        })

        if i % 200 == 0:
            print(f"{i} samples processed...")

    df_out = pd.DataFrame(results)

    raw_acc = correct_raw / len(df_out)
    hybrid_acc = correct_hybrid / len(df_out)

    metrics = {
        "raw_accuracy": raw_acc,
        "hybrid_accuracy": hybrid_acc,
        "num_samples": len(df_out),
    }

    print("\nFINAL RESULTS")
    print(metrics)

    df_out.to_csv(CFG.output_dir / "predictions.csv", index=False)

    with open(CFG.output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    tokenizer.save_pretrained(str(CFG.output_dir))

    print("Saved to:", CFG.output_dir)

# =========================================================
if __name__ == "__main__":
    run()