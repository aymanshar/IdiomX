from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd


TASK3_RELATIVE_DIR = Path('deep_learning') / 'datasets' / 'arabic_context_to_idiom'


def find_project_root(start_path: Path | None = None) -> Path:
    start = (start_path or Path.cwd()).resolve()
    candidates = [start, *start.parents]
    for candidate in candidates:
        if (candidate / 'deep_learning').exists() and (candidate / 'notebooks').exists():
            return candidate
    raise RuntimeError('Project root not found. Run from inside the IdiomX repository.')


def get_desktop_results_root(folder_name: str = 'IdiomX_Task3_Results') -> Path:
    root = Path.home() / 'Desktop' / folder_name
    root.mkdir(parents=True, exist_ok=True)
    return root


def normalize_text(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r'\s+', ' ', text)
    return text


def safe_filename(name: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_\-.]+', '_', name)


@dataclass
class Task3Data:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def load_task3_data(project_root: Path, idiomatic_only: bool = True) -> Task3Data:
    task_dir = project_root / TASK3_RELATIVE_DIR
    train = pd.read_csv(task_dir / 'train.csv')
    val = pd.read_csv(task_dir / 'validation.csv')
    test = pd.read_csv(task_dir / 'test.csv')

    if idiomatic_only:
        for df_name, df in [('train', train), ('val', val), ('test', test)]:
            if 'is_example_idiom' not in df.columns:
                raise ValueError(f"Column 'is_example_idiom' not found in {df_name} split.")
        train = train[train['is_example_idiom'].astype(bool)].copy()
        val = val[val['is_example_idiom'].astype(bool)].copy()
        test = test[test['is_example_idiom'].astype(bool)].copy()

    return Task3Data(train=train.reset_index(drop=True), val=val.reset_index(drop=True), test=test.reset_index(drop=True))


def compute_exact_match_metrics(gold: List[str], pred: List[str]) -> Dict[str, float]:
    gold_norm = [normalize_text(x) for x in gold]
    pred_norm = [normalize_text(x) for x in pred]
    correct = [g == p for g, p in zip(gold_norm, pred_norm)]
    accuracy = sum(correct) / len(correct) if correct else 0.0
    return {
        'exact_match_accuracy': accuracy,
        'num_samples': len(gold_norm),
        'num_correct': int(sum(correct)),
        'num_incorrect': int(len(correct) - sum(correct)),
    }


def compute_retrieval_metrics(gold: List[str], ranked_preds: List[List[str]]) -> Dict[str, float]:
    gold_norm = [normalize_text(x) for x in gold]
    ranks: List[int | None] = []
    for g, preds in zip(gold_norm, ranked_preds):
        preds_norm = [normalize_text(p) for p in preds]
        rank = None
        for i, p in enumerate(preds_norm, start=1):
            if p == g:
                rank = i
                break
        ranks.append(rank)

    def topk(k: int) -> float:
        return sum(1 for r in ranks if r is not None and r <= k) / len(ranks) if ranks else 0.0

    mrr = sum((1 / r) if r is not None else 0 for r in ranks) / len(ranks) if ranks else 0.0
    return {
        'top1_accuracy': topk(1),
        'top3_accuracy': topk(3),
        'top5_accuracy': topk(5),
        'mrr_at_all': mrr,
        'num_samples': len(ranks),
    }


def save_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')


def save_predictions_truth_table(
    predictions_df: pd.DataFrame,
    output_dir: Path,
    pred_col: str,
    model_name: str,
) -> None:
    df = predictions_df.copy()
    df['gold_norm'] = df['gold_idiom'].map(normalize_text)
    df['pred_norm'] = df[pred_col].map(normalize_text)
    df['is_correct'] = df['gold_norm'] == df['pred_norm']

    truth_table = pd.DataFrame({
        'Outcome': ['Correct', 'Incorrect'],
        'Count': [int(df['is_correct'].sum()), int((~df['is_correct']).sum())],
    })
    truth_table['Percent'] = truth_table['Count'] / truth_table['Count'].sum()

    truth_table.to_csv(output_dir / f'{safe_filename(model_name)}_truth_table.csv', index=False)
    df.to_csv(output_dir / f'{safe_filename(model_name)}_predictions.csv', index=False)


def save_top_confusions(
    predictions_df: pd.DataFrame,
    output_dir: Path,
    pred_col: str,
    model_name: str,
    top_n: int = 20,
) -> pd.DataFrame:
    df = predictions_df.copy()
    df['gold_norm'] = df['gold_idiom'].map(normalize_text)
    df['pred_norm'] = df[pred_col].map(normalize_text)
    conf = df[df['gold_norm'] != df['pred_norm']].copy()
    if conf.empty:
        out = pd.DataFrame(columns=['gold_idiom', 'predicted_idiom', 'count'])
        out.to_csv(output_dir / f'{safe_filename(model_name)}_top_confusions.csv', index=False)
        return out

    top_conf = (
        conf.groupby(['gold_idiom', pred_col])
        .size()
        .reset_index(name='count')
        .sort_values('count', ascending=False)
        .head(top_n)
        .rename(columns={pred_col: 'predicted_idiom'})
    )
    top_conf.to_csv(output_dir / f'{safe_filename(model_name)}_top_confusions.csv', index=False)
    return top_conf


def save_demo_samples(
    predictions_df: pd.DataFrame,
    output_dir: Path,
    pred_col: str,
    model_name: str,
    n: int = 10,
) -> None:
    df = predictions_df.copy()
    df['gold_norm'] = df['gold_idiom'].map(normalize_text)
    df['pred_norm'] = df[pred_col].map(normalize_text)
    df['is_correct'] = df['gold_norm'] == df['pred_norm']

    correct = df[df['is_correct']].head(n)
    incorrect = df[~df['is_correct']].head(n)

    correct.to_csv(output_dir / f'{safe_filename(model_name)}_demo_correct.csv', index=False)
    incorrect.to_csv(output_dir / f'{safe_filename(model_name)}_demo_incorrect.csv', index=False)


def save_bar_chart(labels: List[str], values: List[float], title: str, ylabel: str, out_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.bar(labels, values)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def save_horizontal_bar_from_df(df: pd.DataFrame, label_col: str, value_col: str, title: str, out_path: Path) -> None:
    if df.empty:
        return
    plot_df = df.iloc[::-1]
    plt.figure(figsize=(10, 6))
    plt.barh(plot_df[label_col].astype(str), plot_df[value_col])
    plt.title(title)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def build_basic_report(model_name: str, metrics: Dict[str, float], extra_lines: List[str] | None = None) -> str:
    lines = [f'Model: {model_name}', '']
    for k, v in metrics.items():
        lines.append(f'- {k}: {v}')
    if extra_lines:
        lines.extend(['', *extra_lines])
    return '\n'.join(lines)
