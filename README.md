# IdiomX: A Large Bilingual Benchmark for Idiom Understanding, Retrieval, and Generation

## IdiomX  

---

[![Hugging Face](https://img.shields.io/badge/HuggingFace-Dataset-yellow?logo=huggingface)](https://huggingface.co/datasets/aymansharara/IdiomX)
[![Kaggle](https://img.shields.io/badge/Kaggle-Dataset-blue?logo=kaggle)](https://www.kaggle.com/datasets/aymansharara/idiomx)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.19137833-blue)](https://doi.org/10.5281/zenodo.19137833)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dataset Size](https://img.shields.io/badge/Examples-123K+-informational)]()
[![Languages](https://img.shields.io/badge/Languages-EN%20%7C%20AR-blue)]()
[![Tasks](https://img.shields.io/badge/Tasks-NLP%20%7C%20Translation%20%7C%20Classification-purple)]()
[![Status](https://img.shields.io/badge/Status-Active%20Research-orange)]()

---

**A Large-Scale Bilingual Dataset for Idiomatic Expression Understanding**

**Author:** Ayman Ali Sharara  

**Affiliation:**  
MSc Data Science & Machine Learning (SPOC S21)  
DSTI School of Engineering  
https://dsti.school/

**Project Context:**  
Deep Learning with Python  
Supervised by Prof. Hanna Abi Akl  

**Contact:**  
- Academic: ayman.sharara@edu.dsti.institute  
- Personal: aymanshar@gmail.com  

---
## Overview

**IdiomX** is a large-scale bilingual dataset and benchmark designed for **idiomatic expression understanding, retrieval, and generation**.

It provides:
- **123K+ contextualized examples**
- **~15K idioms**
- **English–Arabic semantic alignment**
- **Rich annotations and multiple NLP tasks**

> IdiomX is not just a dataset — it is a **complete research framework** for studying idiomatic language in both monolingual and cross-lingual settings.

---

## Motivation

Idioms are inherently **non-compositional** and **context-dependent**, making them one of the most challenging phenomena in NLP.

Existing resources are:
- small
- monolingual
- weakly contextualized

IdiomX addresses this gap by combining:
- multi-source lexical collection  
- LLM-based enrichment  
- structured validation  
- deep learning benchmarks  

---

## Key Contributions

- Large-scale **bilingual idiom dataset (EN–AR)**
- Context-rich examples (idiomatic + literal usage)
- Canonical and surface-form modeling
- Cross-lingual semantic alignment (Arabic ↔ English)
- Structured LLM-based enrichment and validation pipeline
- Multi-task benchmark for idiom understanding and generation
- Reproducible research framework for dataset + modeling


---

## Dataset Statistics

| Metric | Value |
|------|------|
| Total examples | 123,336 |
| Unique idioms | 14,986 |
| Avg examples / idiom | 8.2 |
| Arabic coverage | 99.99% |
| Label balance | 50 / 50 |

---

## Final Dataset Snapshot

The current LLM enrichment stage has been completed successfully.

- Raw idioms: **16,107**
- Generated examples per idiom: **4**
- Final enriched rows: **64,428**
- Automatically valid rows: **63,286**
- Verified rows: **658**
- Corrected rows: **476**

Final dataset file:

```text
data/enriched/idiomx_enriched_final.csv
```
---
## Repository Structure

```
	IdiomX/
	│
	│
	├── deep_learning/
	│   ├── datasets/
	│   ├── models/
	│   ├── training/
	│   ├── evaluation/
	│   └── README.md
	│
	├── data/
	│   ├── raw/
	│   ├── enriched/
	│   └── splits/
	│
	├── figures/
	├── paper/
	└── README.md
```
---
##Key Contributions

- Large-scale **bilingual idiom dataset (EN–AR)**
- Context-rich examples (idiomatic + literal usage)
- Canonical and surface-form modeling
- Cross-lingual semantic alignment (Arabic ↔ English)
- Structured LLM-based enrichment and validation pipeline
- Multi-task benchmark for idiom understanding and generation
- Reproducible research framework for dataset + modeling

This module prepares task-specific datasets and trains models for multiple benchmark tasks, including:

* idiom detection
* idiom meaning retrieval
* context-to-idiom prediction
* cross-lingual idiom retrieval
* idiom surface-form normalization

A critical design principle is that all train/validation/test splits are created **by idiom, not by row**, to prevent leakage and memorization.

## Benchmark Tasks

IdiomX supports multiple NLP tasks:

- **Idiom Detection**  
  Classify idiomatic vs literal usage

- **Context → Idiom Generation**  
  Predict idiom from context (main contribution)

- **Arabic Context → Idiom**  
  Cross-lingual generation (EN ← AR)

- **Idiom Meaning Retrieval**

- **Surface Normalization**

---

## Key Research Findings

### Idiom Detection
- Transformer models significantly outperform classical baselines
- DeBERTa achieves the best performance (~0.91 F1)
- Idiom detection is fundamentally a **semantic task**

### Context → Idiom (Main Contribution)

| Model | Exact Match |
|------|------------|
| SBERT Retrieval | 0.461 |
| FLAN-T5 | 0.678 |
| **Hybrid (Proposed)** | **0.818** |

The proposed **generation-first hybrid model** improves:
- +35.7% over retrieval
- +14% over generation

> This demonstrates that idiom prediction requires **both generative reasoning and retrieval grounding**.

---

## Reproducibility

This repository is structured for full reproducibility. Each major module has its own dedicated `README.md` with:

* overview
* inputs and outputs
* step-by-step execution instructions
* required files
* code entry points
* recommended run order

For environment setup and detailed execution instructions, refer to the module-specific README files.

---
## Pipeline Overview

1. Data collection (WordNet, Wiktionary, etc.)
2. Cleaning & normalization
3. LLM-based enrichment (examples + meanings + translations)
4. Validation & correction
5. Dataset structuring
6. Model training & evaluation

---

## Deep Learning Setup

- All datasets are split **by idiom (not by row)** to prevent leakage
- Supports reproducible experiments
- Works both in:
  - Jupyter Notebook
  - Terminal execution
  
---

## Dataset Access

- 🤗 Hugging Face:  
  https://huggingface.co/datasets/aymansharara/IdiomX  

- 📊 Kaggle:  
  https://www.kaggle.com/datasets/aymansharara/idiomx  

- 📄 DOI (Zenodo):  
  https://doi.org/10.5281/zenodo.19137833  
  
---

## Research Paper

The full research paper describing IdiomX and experiments is included in the repository.

---

## Limitations

- Partial reliance on LLM-generated data
- Arabic limited to Modern Standard Arabic
- Exact-match evaluation may underestimate semantic correctness
---
## Research Perspective

IdiomX is intended as a benchmark-oriented research artifact rather than only a static dataset. The project is designed to support publication in NLP venues focused on lexical semantics, figurative language, multilingual NLP, and low-resource semantic transfer.

The LLM enrichment stage demonstrates that structured generation and targeted verification can be used to construct a high-quality bilingual idiom resource at scale. The downstream modeling stage is intended to evaluate how well neural architectures can generalize to unseen idioms and cross-lingual contexts.

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

* * *