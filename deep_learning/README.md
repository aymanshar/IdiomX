# IdiomX Deep Learning Benchmark

## Overview

The Deep Learning module of IdiomX evaluates how well modern natural language processing models can understand and generate idiomatic language. Using the enriched IdiomX dataset, this module prepares benchmark-ready datasets and trains models across multiple idiom-related tasks.

The goal is to investigate how neural models interpret idioms, retrieve their meanings, predict idiomatic expressions from context, and perform cross-lingual idiom understanding.

The module includes dataset preparation, task construction, model training, and evaluation pipelines.

---

# Purpose

Idioms are difficult for language models because their meanings are often non-compositional. The meaning of an idiom cannot usually be derived directly from the meanings of its individual words.

For example:

spill the beans

does not literally refer to spilling beans, but instead means:

reveal a secret

The Deep Learning module evaluates whether machine learning models can learn this type of semantic mapping using the IdiomX dataset.

---

# Input Dataset

The deep learning experiments use the enriched dataset produced by the LLM enrichment module.

Input file:

data/enriched/idiomx_enriched_final.csv

Dataset statistics:
Raw idioms: 16,107
Examples per idiom: 4
Total rows: 64,428

Each row contains:

- idiom
- canonical idiom form
- contextual sentence
- idiom surface form
- semantic explanation
- Arabic translation
- metadata and validation status

---

# Benchmark Tasks

The IdiomX dataset supports several idiom-related NLP tasks.

---

## 1 Idiom Detection

Determine whether a phrase is used idiomatically or literally within a sentence.

Example:

Sentence: He finally spilled the beans about the plan.
Label: idiomatic

Example literal usage:

Sentence: She spilled the beans on the kitchen counter.
Label: literal

This task evaluates contextual semantic understanding.

---

## 2 Idiom Meaning Retrieval

Predict the semantic meaning of an idiom.

Example:

Input: spill the beans
Output: reveal a secret

This task can be evaluated as:

- text generation
- semantic retrieval
- classification over meaning candidates

Because IdiomX includes Arabic translations, the task can also be evaluated bilingually.

---

## 3 Context-to-Idiom Prediction

Given a contextual sentence describing a situation, the model must predict the most appropriate idiom.

Example:

Input:
He accidentally revealed the secret information.

Output:
spill the beans

This task evaluates idiomatic language generation.

---

## 4 Cross-Lingual Idiom Retrieval

The dataset supports cross-lingual idiom retrieval between Arabic and English.

Example:

Input (Arabic):
كشف السر دون قصد

Output (English idiom):
spill the beans

This task evaluates multilingual semantic alignment.

---

## 5 Idiom Surface Normalization

Idioms appear in many grammatical forms.

Examples:

break the ice
broke the ice
breaking the ice

The task is to map the surface form in context to the canonical idiom.

Example:

Input:
He finally broke the ice during the meeting.

Output:
break the ice


---

# Dataset Splitting Strategy

To prevent data leakage, the IdiomX dataset is split **by idiom rather than by row**.

This is critical because each idiom has multiple contextual examples.

If rows were split randomly, the same idiom would appear in both training and test sets, allowing models to memorize meanings rather than learn generalizable patterns.

Instead:

Train idioms: 80%
Validation idioms: 10%
Test idioms: 10%

All examples belonging to the same idiom remain in the same split.

Output files:

data/splits/train.csv
data/splits/validation.csv
data/splits/test.csv

---

# Folder Structure

	deep_learning/
	│
	├── datasets/
	│ ├── build_splits.py
	│
	├── models/
	│ ├── bert_model.py
	│ ├── roberta_model.py
	│
	├── training/
	│ ├── train_classifier.py
	│ ├── train_generator.py
	│
	├── evaluation/
	│ ├── evaluate_classifier.py
	│ ├── evaluate_generator.py
	│
	└── README.md

---

# Baseline Models

The benchmark includes both classical and transformer-based models.

---

## Classical Baselines

### TF-IDF + Logistic Regression

A simple baseline using TF-IDF sentence features and logistic regression.

### Support Vector Machine

SVM classifiers using TF-IDF features provide a strong classical comparison.

---

## Transformer Models

### BERT

BERT models are used for classification tasks such as idiom detection and meaning retrieval.

### RoBERTa

RoBERTa improves contextual representation quality and often performs well in semantic classification.

### XLM-RoBERTa

XLM-RoBERTa supports multilingual inputs and is used for cross-lingual idiom retrieval.

### FLAN-T5

FLAN-T5 is used for generative tasks such as context-to-idiom prediction.

---

# Evaluation Metrics

Different tasks use different evaluation metrics.

---

## Classification Tasks

Accuracy
Precision
Recall
F1-score

---

## Retrieval Tasks

Top-1 Accuracy
Top-5 Accuracy
Mean Reciprocal Rank (MRR)

---

## Generation Tasks

BLEU
ROUGE
Semantic Similarity

---

# Example Training Workflow

Example experiment pipeline:

1 Prepare dataset splits

python datasets/build_splits.py

2 Train a classifier

python training/train_classifier.py

3 Evaluate model performance

python evaluation/evaluate_classifier.py

---

# Hardware Requirements

Typical hardware used for experiments:

GPU: NVIDIA 12GB VRAM
RAM: 64GB
CPU: 32 cores

The experiments can also run on smaller GPUs with reduced batch sizes.

---

# Reproducibility

All experiments follow reproducible practices:

- fixed random seeds
- idiom-level dataset splitting
- saved model checkpoints
- consistent evaluation metrics

This ensures fair comparison across models.

---

# Next Research Steps

Future work may explore:

- idiom understanding in low-resource languages
- idiom paraphrasing
- idiom generation in conversational agents
- idiom-aware language model pretraining


