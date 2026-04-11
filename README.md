# IdiomX – Idiom Understanding Benchmark (Tasks & Experiments)

---

[![Hugging Face](https://img.shields.io/badge/HuggingFace-Dataset-yellow?logo=huggingface)](https://huggingface.co/datasets/aymansharara/IdiomX)
[![Kaggle](https://img.shields.io/badge/Kaggle-Dataset-blue?logo=kaggle)](https://www.kaggle.com/datasets/aymansharara/idiomx)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

This repository contains the **benchmark experiments and modeling pipeline** for IdiomX.

The focus of this repo is:

- evaluating idiom understanding tasks
- building reproducible deep learning pipelines
- demonstrating practical inference systems

> This repository is NOT for dataset construction.  
> The dataset is already finalized and publicly available.

---

## Dataset

We use the **high-quality final IdiomX dataset**, available here:

- 🤗 Hugging Face: https://huggingface.co/datasets/aymansharara/IdiomX  
- 📊 Kaggle: https://www.kaggle.com/datasets/aymansharara/idiomx  

The dataset includes:
- English idioms with contextual examples
- Arabic translations and semantic alignment
- idiomatic vs literal labels
- multiple examples per idiom

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

## Benchmark Tasks

This repository implements three main tasks:

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

This is the **core contribution of the project**.

---

### 3. Arabic → Idiom (Cross-lingual)

Given Arabic input, retrieve the corresponding English idiom.

This task evaluates:
- multilingual understanding
- cross-lingual semantic alignment

---

## How to Run

### Option 1 — Full Reproduction

Run the benchmark notebook:

notebooks/Task2_Context_to_Idiom_Benchmark.ipynb

This will:
- train models  
- evaluate performance  
- generate artifacts  

---

### Option 2 — Quick Demo (Recommended)

Run the demo notebook:

notebooks/Task2_Context_to_Idiom_Demo.ipynb

This will:
- load precomputed artifacts  
- allow testing custom sentences  
- return ranked idiom predictions  

---

## Artifacts

Artifacts are stored in:
artifacts/task2/

These include:
- idiom embeddings  
- index mappings  
- precomputed retrieval structures  

If artifacts are missing:
- run the benchmark notebook first  

---

## Requirements

Install dependencies:
pip install -r requirements.txt

Minimal requirements:


sentence-transformers
rank-bm25
numpy
pandas
scikit-learn
matplotlib

---

## Key Results (Task 2)

| Model | Top-1 Accuracy |
|------|--------------|
| MiniLM | 0.537 |
| Hybrid | 0.763 |
| Hybrid + Reranker | **0.867** |

**Key insight:**
- retrieval alone is not enough  
- hybrid + reranking significantly improves performance  

---

## Reproducibility

This repository is designed to be:

- fully reproducible  
- notebook-driven  
- easy to test  

Two usage modes are supported:
- full experiment reproduction  
- lightweight inference demo  

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
Dataset repository::
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