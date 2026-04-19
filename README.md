# IdiomX – Idiom Understanding Benchmark (Tasks & Experiments)

---

[![Hugging Face Dataset](https://img.shields.io/badge/HuggingFace-Dataset-yellow?logo=huggingface)](https://huggingface.co/datasets/aymansharara/IdiomX)
[![Task 1 Demo](https://img.shields.io/badge/Task1-Demo-blue)](https://huggingface.co/spaces/aymansharara/idiomX_idiom_detection_demo)
[![Task 2 Demo](https://img.shields.io/badge/Task2-Demo-green)](https://huggingface.co/spaces/aymansharara/idiomx_context_to_idiom_demo)
[![Task 3 Demo](https://img.shields.io/badge/Task3-Demo-orange)](https://huggingface.co/spaces/aymansharara/idiomx_arabic_context_to_idiom_demo)
[![Kaggle Dataset](https://img.shields.io/badge/Kaggle-Dataset-blue?logo=kaggle)](https://www.kaggle.com/datasets/aymansharara/idiomx)
[![Paper](https://img.shields.io/badge/Paper-Research-red)](paper/)
[![Task2 Accuracy](https://img.shields.io/badge/Task2-Top1%20Accuracy%200.867-brightgreen)]
[![Task3 Cross-Lingual](https://img.shields.io/badge/Task3-Cross--Lingual-blueviolet)]
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

Understanding idiomatic language remains a major challenge in NLP due to its non-literal and context-dependent nature.

IdiomX introduces a unified benchmark framework for idiom understanding, spanning classification, contextual prediction, and cross-lingual retrieval.

This repository focuses on:
- evaluating idiom understanding tasks
- building reproducible deep learning pipelines
- demonstrating practical inference systems

> This repository focuses on benchmarking and modeling.  
> Dataset construction is described separately.

---

## Dataset

We use the high-quality final IdiomX dataset, available here:

- 🤗 Hugging Face: https://huggingface.co/datasets/aymansharara/IdiomX  
- 📊 Kaggle: https://www.kaggle.com/datasets/aymansharara/idiomx  

The dataset includes:
- English idioms with contextual examples
- Arabic translations and semantic alignment
- idiomatic vs literal labels
- multiple examples per idiom

Although this work focuses on English–Arabic alignment, the pipeline is language-agnostic and can be extended to other languages.

---

## Repository Structure

```
	IdiomX/
	│
	├── data/
	│
	├── notebooks/
	│   ├── idiomx_dataset_analysis.ipynb
	│   ├── Task1_idiom_detection_Benchmark.ipynb
	│   ├── Task1_idiom_detection_Demo.ipynb
	│   ├── Task2_Context_to_Idiom_Benchmark.ipynb
	│   ├── Task2_Context_to_Idiom_Demo.ipynb
	│   ├── Task3_Arabic_Semantic_Retrieval_Benchmark.ipynb
	│   └── Task3_Arabic_Semantic_Retrieval_Demo.ipynb	
	│
	├── figures/
	│
	├── artifacts/
	│   ├── task1/
	│   ├── task2/
	│   └── task3/
	│
	├── paper/
	│
	└── README.md
```
---
### loading dataset 
loading varient dataset from huggingface
```python
# 1.1 load datasets
from datasets import load_dataset
import pandas as pd

# Full dataset load
HF_DATASET_NAME = "aymansharara/IdiomX"
HF_CONFIG_NAME = "idiomx_full"

dataset = load_dataset(HF_DATASET_NAME, HF_CONFIG_NAME)
df_raw = dataset["full"].to_pandas()

# task2 idiomx retrieval dataset load
HF_DATASET_ID = "aymansharara/IdiomX"
CONFIG_NAME = "task2_idiomx_retrieval_dataset"

dataset = load_dataset(HF_DATASET_ID, CONFIG_NAME)
df = dataset[list(dataset.keys())[0]].to_pandas()

```
---

## Benchmark Tasks

These tasks form a progressive evaluation setup, moving from classification to contextual reasoning and finally to cross-lingual semantic alignment.

### 1. Idiom Detection
- classify idiomatic vs literal usage
- transformer-based models (e.g., DeBERTa)

---

### 2. Context → Idiom (Main Task)

Given a sentence, predict the correct idiom.

Pipeline:
- dense retrieval (MiniLM)
- lexical retrieval (BM25)
- hybrid scoring
- cross-encoder reranking

This task represents the primary contribution of the benchmark.

---

### 3. Arabic → Idiom (Cross-lingual)

Given Arabic input, retrieve the corresponding English idiom.

This task evaluates:
- multilingual understanding
- cross-lingual semantic alignment

---

## Interactive Demos (Hugging Face Spaces)

We provide interactive demos for all tasks via Hugging Face Spaces:
These demos allow users to interactively explore the IdiomX system:
- Task 1 focuses on idiom detection
- Task 2 demonstrates hybrid retrieval with reranking
- Task 3 shows cross-lingual retrieval (Arabic → English)

Each demo exposes model behavior and scoring, enabling qualitative analysis.

### Task 1 — Idiom Detection
Detect whether a sentence contains an idiomatic expression.

🔗 https://huggingface.co/spaces/aymansharara/idiomX_idiom_detection_demo

---

### Task 2 — Context → Idiom Retrieval (Hybrid + Reranker)
Retrieve the most relevant idioms given a sentence using hybrid retrieval (dense + BM25) followed by reranking.

🔗 https://huggingface.co/spaces/aymansharara/idiomx_context_to_idiom_demo

---

### Task 3 — Arabic Context → English Idiom Retrieval
Retrieve English idioms from Arabic input using a fine-tuned multilingual embedding model.

🔗 https://huggingface.co/spaces/aymansharara/idiomx_arabic_context_to_idiom_demo

---

## How to Run

### (Task 2: Context → Idiom)

### 1. Full Benchmark

Run the benchmark notebook:

notebooks/Task2_Context_to_Idiom_Benchmark.ipynb

This will:
- train retrieval and reranking models  
- evaluate performance  
- generate task-specific artifacts  

---

### 2. Quick Demo

Run the demo notebook:

notebooks/Task2_Context_to_Idiom_Demo.ipynb

This will:
- load precomputed artifacts  
- allow testing custom sentences  
- return ranked idiom predictions  

---

### Task 3 — Arabic → Idiom

#### 1. Full Benchmark

Run:
notebooks/Task3_Arabic_Semantic_Retrieval_Benchmark.ipynb

#### 2. Quick Demo

Run:
notebooks/Task3_Arabic_Semantic_Retrieval_Demo.ipynb

---

## Artifacts

Artifacts are organized per task:

- `artifacts/task1/`  
- `artifacts/task2/`  
- `artifacts/task3/`  

Example (Task 2):
- idiom embeddings  
- index mappings  
- retrieval structures  

If artifacts are missing:
- run the corresponding benchmark notebook  

---

## Requirements

Install dependencies:

pip install -r requirements.txt

Minimal requirements:

- sentence-transformers  
- rank-bm25  
- numpy  
- pandas  
- scikit-learn  
- matplotlib  

---

## Key Results

### Task 2 — Context → Idiom

| Model | Top-1 Accuracy |
|------|--------------|
| MiniLM | 0.537 |
| Hybrid | 0.763 |
| Hybrid + Reranker | **0.867** |

**Key insights:**
- retrieval alone is insufficient  
- hybrid retrieval improves performance  
- reranking significantly boosts accuracy  

---

### Task 3 — Arabic → Idiom

- strong semantic alignment across languages  
- performance improves significantly after fine-tuning  
- analysis includes:
  - error distribution  
  - hard negatives  
  - confidence calibration  

---

## Reproducibility

This repository is designed to be:

- fully reproducible  
- notebook-driven  
- easy to experiment with  

Two usage modes:
- full experiment reproduction  
- lightweight inference demos  

---

## Research Paper

The full research paper is available in:
paper/
---

## Limitations

- performance depends on clarity of input context  
- open-ended sentences may return related idioms instead of exact matches  
- reranker operates on top-k candidates (not full search space)  

---

## Status

Current project checkpoint:


* Data collection: completed
* LLM enrichment: completed
* Dataset verification: completed

Dataset repository:
https://github.com/aymanshar/idiomx-dataset

* Deep learning benchmark preparation: in progress

---

## Citation

If you use IdiomX in your research, please cite:

@dataset{idiomx2026,
title={IdiomX: A Large-Scale Bilingual Dataset for Idiomatic Expression Understanding},
author={Sharara, Ayman},
year={2026}
}

---
## Final Note

IdiomX aims to push forward research in:
- figurative language understanding  
- multilingual NLP  
- semantic reasoning  

If you find this project useful, consider starring the repository.
---