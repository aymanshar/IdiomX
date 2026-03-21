from pathlib import Path
import pandas as pd
import numpy as np
import torch

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from sklearn.metrics import accuracy_score

BASE_DIR = Path(__file__).resolve().parents[2]

TEST_CSV = BASE_DIR / "deep_learning/datasets/context_to_idiom/test.csv"
BANK_CSV = BASE_DIR / "deep_learning/datasets/idiom_bank/idiom_bank.csv"

OUTPUT_DIR = BASE_DIR / "deep_learning/models/context_to_idiom_hybrid"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading datasets...")

test_df = pd.read_csv(TEST_CSV)
bank_df = pd.read_csv(BANK_CSV)

test_contexts = test_df["input_text"].astype(str).tolist()
gold_idioms = test_df["target_text"].astype(str).tolist()

bank_idioms = bank_df["idiom_canonical"].astype(str).tolist()
bank_texts = bank_df["retrieval_text_en"].astype(str).tolist()

print("Test samples:", len(test_contexts))
print("Idiom bank size:", len(bank_idioms))

# ---------- SBERT retrieval ----------

print("\nLoading SBERT retrieval model...")

retrieval_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

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

# ---------- FLAN-T5 reranker ----------

print("\nLoading FLAN-T5 reranker...")

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base", use_fast=False)
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

model.to(device)
model.eval()

predictions = []

print("\nRunning hybrid retrieval + reranking...\n")

for i in range(len(test_contexts)):

    context = test_contexts[i]

    # SBERT retrieval
    scores = cosine_similarity(
        context_emb[i].reshape(1, -1),
        bank_emb
    )[0]

    top_idx = np.argsort(scores)[::-1][:5]
    candidates = [bank_idioms[j] for j in top_idx]

    candidate_scores = []

    for cand in candidates:

        prompt = (
            "Context: " + context + "\n\n"
            "Candidate idiom: " + cand + "\n\n"
            "Is this idiom the best fit for the context? Answer yes or no."
        )

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

        yes_token = tokenizer.encode("yes", add_special_tokens=False)[0]

        yes_prob = probs[0][yes_token].item()

        candidate_scores.append(yes_prob)

    best_idx = np.argmax(candidate_scores)

    pred = candidates[best_idx]

    predictions.append(pred)

accuracy = accuracy_score(gold_idioms, predictions)

print("\nHybrid Exact Match Accuracy:", accuracy)

pd.DataFrame({
    "context": test_contexts,
    "gold": gold_idioms,
    "prediction": predictions
}).to_csv(
    OUTPUT_DIR / "test_predictions.csv",
    index=False
)

pd.DataFrame([{
    "exact_match_accuracy": accuracy
}]).to_csv(
    OUTPUT_DIR / "metrics.csv",
    index=False
)

print("\nSaved results to:", OUTPUT_DIR)