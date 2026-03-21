from pathlib import Path
import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parents[2]

TEST_CSV = BASE_DIR / "deep_learning/datasets/idiom_to_meaning/test.csv"
BANK_CSV = BASE_DIR / "deep_learning/datasets/idiom_bank/idiom_bank.csv"

OUTPUT_DIR = BASE_DIR / "deep_learning/models/meaning_to_idiom_sbert_retrieval"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading test meanings and idiom bank...")

test_df = pd.read_csv(TEST_CSV, low_memory=False)
bank_df = pd.read_csv(BANK_CSV, low_memory=False)

# Clean test
test_df = test_df[test_df["input_text"].notna()].copy()
test_df["input_text"] = test_df["input_text"].astype(str).str.strip()
test_df["target_meaning_en"] = test_df["target_meaning_en"].astype(str).str.strip()

# In idiom_to_meaning, input_text is the idiom and target_meaning_en is the meaning.
# For meaning -> idiom retrieval, we want:
# query = target_meaning_en
# gold = input_text
test_queries = test_df["target_meaning_en"].tolist()
test_gold = test_df["input_text"].astype(str).str.strip().tolist()

# Clean bank
bank_df = bank_df[bank_df["idiom_canonical"].notna()].copy()
bank_df["idiom_canonical"] = bank_df["idiom_canonical"].astype(str).str.strip()

bank_texts = bank_df["retrieval_text_en"].astype(str).tolist()
bank_idioms = bank_df["idiom_canonical"].astype(str).tolist()

print("Test samples:", len(test_queries))
print("Bank idioms:", len(bank_idioms))

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
print("Loading model:", MODEL_NAME)
model = SentenceTransformer(MODEL_NAME)

print("Encoding idiom bank...")
bank_emb = model.encode(
    bank_texts,
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)

print("Encoding test meaning queries...")
query_emb = model.encode(
    test_queries,
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)

print("Computing cosine similarities...")
sim = cosine_similarity(query_emb, bank_emb)

top1 = 0
top3 = 0
top5 = 0
rr_scores = []

pred_rows = []

for i in range(sim.shape[0]):
    scores = sim[i]
    ranked_idx = np.argsort(scores)[::-1]

    ranked_idioms = [bank_idioms[j] for j in ranked_idx[:5]]
    gold = test_gold[i]

    if gold == ranked_idioms[0]:
        top1 += 1
    if gold in ranked_idioms[:3]:
        top3 += 1
    if gold in ranked_idioms[:5]:
        top5 += 1

    rank_pos = None
    for rank, j in enumerate(ranked_idx[:100], start=1):
        if bank_idioms[j] == gold:
            rank_pos = rank
            break

    rr_scores.append(0 if rank_pos is None else 1.0 / rank_pos)

    pred_rows.append({
        "query_meaning_en": test_queries[i],
        "gold_idiom": gold,
        "pred_top1": ranked_idioms[0],
        "pred_top3": " ||| ".join(ranked_idioms[:3]),
        "pred_top5": " ||| ".join(ranked_idioms[:5]),
    })

n = len(test_queries)

results = {
    "top1_accuracy": top1 / n,
    "top3_accuracy": top3 / n,
    "top5_accuracy": top5 / n,
    "mrr_at_100": float(np.mean(rr_scores)),
    "num_test_samples": int(n),
    "num_bank_idioms": int(len(bank_idioms)),
}

print("\nResults:")
for k, v in results.items():
    print(f"{k}: {v}")

pd.DataFrame(pred_rows).to_csv(
    OUTPUT_DIR / "test_predictions.csv",
    index=False,
    encoding="utf-8-sig"
)

pd.DataFrame([results]).to_csv(
    OUTPUT_DIR / "metrics.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nSaved predictions to:", OUTPUT_DIR / "test_predictions.csv")
print("Saved metrics to:", OUTPUT_DIR / "metrics.csv")