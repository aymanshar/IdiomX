from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd
from sentence_transformers import SentenceTransformer, util

from common import (
    build_basic_report,
    compute_retrieval_metrics,
    find_project_root,
    get_desktop_results_root,
    load_task3_data,
    save_bar_chart,
    save_demo_samples,
    save_json,
    save_predictions_truth_table,
    save_text,
    save_top_confusions,
)


def run_retrieval(
    project_root: Path,
    output_dir: Path,
    model_name: str = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
    top_k: int = 5,
    idiomatic_only: bool = True,
) -> Dict:
    data = load_task3_data(project_root, idiomatic_only=idiomatic_only)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_bank = data.train[['target_text']].drop_duplicates().reset_index(drop=True)
    idiom_bank: List[str] = train_bank['target_text'].astype(str).tolist()

    model = SentenceTransformer(model_name)

    bank_embeddings = model.encode(idiom_bank, convert_to_tensor=True, show_progress_bar=True, normalize_embeddings=True)
    test_texts = data.test['input_text'].astype(str).tolist()
    gold = data.test['target_text'].astype(str).tolist()
    test_embeddings = model.encode(test_texts, convert_to_tensor=True, show_progress_bar=True, normalize_embeddings=True)

    similarities = util.cos_sim(test_embeddings, bank_embeddings)
    top_values, top_indices = similarities.topk(k=top_k, dim=1)

    ranked_preds: List[List[str]] = []
    rows = []
    for i in range(len(test_texts)):
        preds = [idiom_bank[idx] for idx in top_indices[i].tolist()]
        ranked_preds.append(preds)
        row = {
            'input_text': test_texts[i],
            'gold_idiom': gold[i],
            'pred_top1': preds[0],
            'score_top1': float(top_values[i][0]),
        }
        for rank in range(1, top_k + 1):
            row[f'pred_top{rank}'] = preds[rank - 1]
            row[f'score_top{rank}'] = float(top_values[i][rank - 1])
        rows.append(row)

    predictions_df = pd.DataFrame(rows)
    metrics = compute_retrieval_metrics(gold=gold, ranked_preds=ranked_preds)
    metrics.update(
        {
            'model_name': model_name,
            'num_bank_idioms': len(idiom_bank),
            'idiomatic_only': idiomatic_only,
        }
    )

    predictions_df.to_csv(output_dir / 'predictions.csv', index=False)
    save_predictions_truth_table(predictions_df, output_dir, pred_col='pred_top1', model_name='task3_retrieval')
    top_conf = save_top_confusions(predictions_df, output_dir, pred_col='pred_top1', model_name='task3_retrieval')
    save_demo_samples(predictions_df, output_dir, pred_col='pred_top1', model_name='task3_retrieval')

    save_bar_chart(
        labels=['Top-1', 'Top-3', 'Top-5'],
        values=[metrics['top1_accuracy'], metrics['top3_accuracy'], metrics['top5_accuracy']],
        title='Task 3 Retrieval Performance',
        ylabel='Score',
        out_path=output_dir / 'chart_topk_accuracy.png',
    )
    if not top_conf.empty:
        top_conf['label'] = top_conf['gold_idiom'].astype(str) + ' -> ' + top_conf['predicted_idiom'].astype(str)
        from common import save_horizontal_bar_from_df
        save_horizontal_bar_from_df(
            top_conf[['label', 'count']].head(15),
            label_col='label',
            value_col='count',
            title='Task 3 Retrieval Top Confusions',
            out_path=output_dir / 'chart_top_confusions.png',
        )

    save_json(output_dir / 'metrics.json', metrics)
    report = build_basic_report(
        'Task 3 Retrieval Baseline',
        metrics,
        extra_lines=[
            f'Test size: {len(test_texts)}',
            f'Unique train idioms used as bank: {len(idiom_bank)}',
        ],
    )
    save_text(output_dir / 'report.txt', report)

    return {'metrics': metrics, 'predictions': predictions_df}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Task 3 Arabic Context -> Idiom retrieval baseline')
    parser.add_argument('--project_root', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--model_name', type=str, default='sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--all_examples', action='store_true', help='Use literal + idiomatic examples together.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve() if args.project_root else find_project_root()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else get_desktop_results_root() / 'task3_retrieval_baseline'
    result = run_retrieval(
        project_root=project_root,
        output_dir=output_dir,
        model_name=args.model_name,
        top_k=args.top_k,
        idiomatic_only=not args.all_examples,
    )
    print('\nSaved to:', output_dir)
    print(result['metrics'])


if __name__ == '__main__':
    main()
