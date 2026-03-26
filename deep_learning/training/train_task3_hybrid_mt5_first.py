from __future__ import annotations

import argparse
from pathlib import Path
from difflib import SequenceMatcher
from typing import Dict, List

import pandas as pd
from sentence_transformers import SentenceTransformer, util
from transformers import MT5ForConditionalGeneration, MT5Tokenizer

from common import (
    build_basic_report,
    compute_exact_match_metrics,
    find_project_root,
    get_desktop_results_root,
    load_task3_data,
    normalize_text,
    save_bar_chart,
    save_demo_samples,
    save_json,
    save_predictions_truth_table,
    save_text,
    save_top_confusions,
)

TASK_PREFIX = 'Arabic to English idiom: '


def _best_bank_match(prediction: str, idiom_bank: List[str], threshold: float) -> str | None:
    pred_norm = normalize_text(prediction)
    best_item = None
    best_score = 0.0
    for item in idiom_bank:
        score = SequenceMatcher(None, pred_norm, normalize_text(item)).ratio()
        if score > best_score:
            best_item = item
            best_score = score
    return best_item if best_score >= threshold else None


def run_hybrid_mt5_first(
    project_root: Path,
    output_dir: Path,
    mt5_model_dir: Path | None = None,
    retrieval_model_name: str = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
    fuzzy_threshold: float = 0.84,
    idiomatic_only: bool = True,
    top_k: int = 5,
) -> Dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_task3_data(project_root, idiomatic_only=idiomatic_only)

    idiom_bank = data.train[['target_text']].drop_duplicates()['target_text'].astype(str).tolist()
    test_inputs = data.test['input_text'].astype(str).tolist()
    gold = data.test['target_text'].astype(str).tolist()

    # 1) Load MT5 model
    if mt5_model_dir is None:
        mt5_model_dir = get_desktop_results_root() / 'task3_mt5_baseline' / 'saved_model'
    tokenizer = MT5Tokenizer.from_pretrained(str(mt5_model_dir))
    model = MT5ForConditionalGeneration.from_pretrained(str(mt5_model_dir))

    prompts = [TASK_PREFIX + x for x in test_inputs]
    batch = tokenizer(prompts, padding=True, truncation=True, max_length=128, return_tensors='pt')
    generated = model.generate(**batch, max_length=24)
    mt5_preds = tokenizer.batch_decode(generated, skip_special_tokens=True)

    # 2) Retrieval fallback prepared once
    retrieval = SentenceTransformer(retrieval_model_name)
    bank_emb = retrieval.encode(idiom_bank, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=True)
    test_emb = retrieval.encode(test_inputs, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=True)
    sims = util.cos_sim(test_emb, bank_emb)
    top_vals, top_idx = sims.topk(k=top_k, dim=1)

    rows = []
    decision_counts = {'mt5_exact_bank': 0, 'mt5_fuzzy_bank': 0, 'retrieval_fallback': 0}
    final_preds: List[str] = []

    norm_bank = {normalize_text(x): x for x in idiom_bank}

    for i, (inp, gold_i, mt5_pred) in enumerate(zip(test_inputs, gold, mt5_preds)):
        norm_pred = normalize_text(mt5_pred)
        if norm_pred in norm_bank:
            final_pred = norm_bank[norm_pred]
            decision = 'mt5_exact_bank'
        else:
            fuzzy = _best_bank_match(mt5_pred, idiom_bank, fuzzy_threshold)
            if fuzzy is not None:
                final_pred = fuzzy
                decision = 'mt5_fuzzy_bank'
            else:
                final_pred = idiom_bank[top_idx[i][0].item()]
                decision = 'retrieval_fallback'

        decision_counts[decision] += 1
        final_preds.append(final_pred)
        rows.append(
            {
                'input_text': inp,
                'gold_idiom': gold_i,
                'mt5_prediction': mt5_pred,
                'retrieval_top1': idiom_bank[top_idx[i][0].item()],
                'final_prediction': final_pred,
                'decision_type': decision,
            }
        )

    predictions_df = pd.DataFrame(rows)
    metrics = compute_exact_match_metrics(gold, final_preds)
    metrics.update(
        {
            'retrieval_model_name': retrieval_model_name,
            'mt5_model_dir': str(mt5_model_dir),
            'fuzzy_threshold': fuzzy_threshold,
            **decision_counts,
        }
    )

    predictions_df.to_csv(output_dir / 'predictions.csv', index=False)
    save_predictions_truth_table(predictions_df.rename(columns={'final_prediction': 'prediction'}), output_dir, pred_col='prediction', model_name='task3_hybrid_mt5_first')
    save_top_confusions(predictions_df.rename(columns={'final_prediction': 'prediction'}), output_dir, pred_col='prediction', model_name='task3_hybrid_mt5_first')
    save_demo_samples(predictions_df.rename(columns={'final_prediction': 'prediction'}), output_dir, pred_col='prediction', model_name='task3_hybrid_mt5_first')

    save_bar_chart(
        labels=['Hybrid Exact Match', 'Error Rate'],
        values=[metrics['exact_match_accuracy'], 1 - metrics['exact_match_accuracy']],
        title='Task 3 Hybrid (mT5-first) Performance',
        ylabel='Score',
        out_path=output_dir / 'chart_hybrid_accuracy.png',
    )
    pd.DataFrame({'Decision': list(decision_counts.keys()), 'Count': list(decision_counts.values())}).to_csv(output_dir / 'decision_breakdown.csv', index=False)
    save_json(output_dir / 'metrics.json', metrics)
    save_text(output_dir / 'report.txt', build_basic_report('Task 3 Hybrid mT5-first', metrics))
    return {'metrics': metrics, 'predictions': predictions_df}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Task 3 Hybrid mT5-first model')
    parser.add_argument('--project_root', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--mt5_model_dir', type=str, default=None)
    parser.add_argument('--retrieval_model_name', type=str, default='sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
    parser.add_argument('--fuzzy_threshold', type=float, default=0.84)
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--all_examples', action='store_true')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve() if args.project_root else find_project_root()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else get_desktop_results_root() / 'task3_hybrid_mt5_first'
    mt5_model_dir = Path(args.mt5_model_dir).resolve() if args.mt5_model_dir else None
    result = run_hybrid_mt5_first(
        project_root=project_root,
        output_dir=output_dir,
        mt5_model_dir=mt5_model_dir,
        retrieval_model_name=args.retrieval_model_name,
        fuzzy_threshold=args.fuzzy_threshold,
        idiomatic_only=not args.all_examples,
        top_k=args.top_k,
    )
    print('\nSaved to:', output_dir)
    print(result['metrics'])


if __name__ == '__main__':
    main()
