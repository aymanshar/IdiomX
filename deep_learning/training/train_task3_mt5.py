from __future__ import annotations

import argparse
import inspect
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    DataCollatorForSeq2Seq,
    MT5ForConditionalGeneration,
    MT5Tokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

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


class Seq2SeqIdiomDataset(Dataset):
    def __init__(self, encodings: Dict[str, List[List[int]]], labels: List[List[int]]) -> None:
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item


def _encode_split(tokenizer, inputs: List[str], targets: List[str], max_input_len: int, max_target_len: int):
    model_inputs = tokenizer(
        inputs,
        max_length=max_input_len,
        truncation=True,
        padding='max_length',
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            max_length=max_target_len,
            truncation=True,
            padding='max_length',
        )
    label_ids = []
    for row in labels['input_ids']:
        label_ids.append([token if token != tokenizer.pad_token_id else -100 for token in row])
    return model_inputs, label_ids


def _training_args(output_dir: Path, fp16: bool, epochs: int, lr: float, train_bs: int, eval_bs: int):
    signature = inspect.signature(Seq2SeqTrainingArguments.__init__)
    kwargs = dict(
        output_dir=str(output_dir),
        save_strategy='epoch',
        logging_strategy='steps',
        logging_steps=100,
        per_device_train_batch_size=train_bs,
        per_device_eval_batch_size=eval_bs,
        gradient_accumulation_steps=4,
        num_train_epochs=epochs,
        learning_rate=lr,
        weight_decay=0.01,
        predict_with_generate=True,
        generation_max_length=24,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        report_to='none',
    )
    if fp16:
        kwargs['fp16'] = True
    if 'evaluation_strategy' in signature.parameters:
        kwargs['evaluation_strategy'] = 'epoch'
    else:
        kwargs['eval_strategy'] = 'epoch'
    return Seq2SeqTrainingArguments(**kwargs)


def run_mt5(
    project_root: Path,
    output_dir: Path,
    model_name: str = 'google/mt5-small',
    idiomatic_only: bool = True,
    max_input_len: int = 128,
    max_target_len: int = 24,
    num_epochs: int = 3,
    learning_rate: float = 5e-5,
    train_batch_size: int = 4,
    eval_batch_size: int = 4,
) -> Dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_task3_data(project_root, idiomatic_only=idiomatic_only)

    tokenizer = MT5Tokenizer.from_pretrained(model_name)
    model = MT5ForConditionalGeneration.from_pretrained(model_name)

    train_src = [TASK_PREFIX + str(x) for x in data.train['input_text'].tolist()]
    val_src = [TASK_PREFIX + str(x) for x in data.val['input_text'].tolist()]
    test_src = [TASK_PREFIX + str(x) for x in data.test['input_text'].tolist()]

    train_tgt = data.train['target_text'].astype(str).tolist()
    val_tgt = data.val['target_text'].astype(str).tolist()
    test_tgt = data.test['target_text'].astype(str).tolist()

    train_enc, train_labels = _encode_split(tokenizer, train_src, train_tgt, max_input_len, max_target_len)
    val_enc, val_labels = _encode_split(tokenizer, val_src, val_tgt, max_input_len, max_target_len)
    test_enc, test_labels = _encode_split(tokenizer, test_src, test_tgt, max_input_len, max_target_len)

    train_dataset = Seq2SeqIdiomDataset(train_enc, train_labels)
    val_dataset = Seq2SeqIdiomDataset(val_enc, val_labels)
    test_dataset = Seq2SeqIdiomDataset(test_enc, test_labels)

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        return compute_exact_match_metrics(decoded_labels, decoded_preds)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    training_args = _training_args(
        output_dir=output_dir / 'hf_outputs',
        fp16=torch.cuda.is_available(),
        epochs=num_epochs,
        lr=learning_rate,
        train_bs=train_batch_size,
        eval_bs=eval_batch_size,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(str(output_dir / 'saved_model'))
    tokenizer.save_pretrained(str(output_dir / 'saved_model'))

    predict_output = trainer.predict(test_dataset=test_dataset, max_length=max_target_len)
    pred_ids = predict_output.predictions[0] if isinstance(predict_output.predictions, tuple) else predict_output.predictions
    decoded_preds = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

    predictions_df = pd.DataFrame(
        {
            'input_text': data.test['input_text'].astype(str).tolist(),
            'gold_idiom': test_tgt,
            'prediction': decoded_preds,
        }
    )

    metrics = compute_exact_match_metrics(test_tgt, decoded_preds)
    metrics.update({'model_name': model_name, 'idiomatic_only': idiomatic_only})

    predictions_df.to_csv(output_dir / 'predictions.csv', index=False)
    save_predictions_truth_table(predictions_df, output_dir, pred_col='prediction', model_name='task3_mt5')
    save_top_confusions(predictions_df, output_dir, pred_col='prediction', model_name='task3_mt5')
    save_demo_samples(predictions_df, output_dir, pred_col='prediction', model_name='task3_mt5')
    save_bar_chart(
        labels=['Exact Match', 'Error Rate'],
        values=[metrics['exact_match_accuracy'], 1 - metrics['exact_match_accuracy']],
        title='Task 3 mT5 Performance',
        ylabel='Score',
        out_path=output_dir / 'chart_exact_match.png',
    )
    save_json(output_dir / 'metrics.json', metrics)
    save_text(output_dir / 'report.txt', build_basic_report('Task 3 mT5 Baseline', metrics))
    return {'metrics': metrics, 'predictions': predictions_df, 'trainer': trainer}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Task 3 Arabic Context -> Idiom mT5 baseline')
    parser.add_argument('--project_root', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--model_name', type=str, default='google/mt5-small')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--train_batch_size', type=int, default=4)
    parser.add_argument('--eval_batch_size', type=int, default=4)
    parser.add_argument('--all_examples', action='store_true')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve() if args.project_root else find_project_root()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else get_desktop_results_root() / 'task3_mt5_baseline'
    result = run_mt5(
        project_root=project_root,
        output_dir=output_dir,
        model_name=args.model_name,
        idiomatic_only=not args.all_examples,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
    )
    print('\nSaved to:', output_dir)
    print(result['metrics'])


if __name__ == '__main__':
    main()
