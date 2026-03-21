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

## Introduction
IdiomX is a research-driven project for building a large-scale bilingual idiom dataset and benchmark for English–Arabic idiom understanding, retrieval, normalization, and generation. The project combines multi-source idiom collection, large language model (LLM)-based enrichment, quality-controlled validation, and downstream deep learning experiments to support idiomatic language research in both monolingual and cross-lingual settings.

The core motivation behind IdiomX is that idioms remain a challenging phenomenon for natural language processing systems because their meanings are often non-literal, context-dependent, and culturally grounded. Existing resources are often small, monolingual, weakly contextualized, or not designed for modern transformer-based learning. IdiomX addresses these limitations by providing a reproducible pipeline that transforms a raw idiom collection into a high-quality bilingual benchmark with contextual examples, semantic annotations, surface-form variation, and evaluation-ready splits.

---

## Research Objective

The objective of IdiomX is to construct a reproducible research pipeline for:

- collecting idioms from multiple linguistic resources
- enriching idiom entries using structured LLM generation
- validating and correcting generated annotations
- building benchmark-ready datasets for multiple NLP tasks
- training and evaluating deep learning models for idiom-related understanding and generation

The project is designed not only as a dataset release but as a complete research framework that supports experimentation, reproducibility, and extension.

---

## Main Contributions

IdiomX is designed around the following contributions:

1. A large bilingual idiom dataset centered on English idioms with Arabic semantic annotations.
2. Contextual expansion of idioms into multiple natural example sentences.
3. Canonical idiom normalization and surface-form modeling.
4. Cross-lingual semantic annotations supporting Arabic-to-English idiom tasks.
5. A structured LLM-based enrichment and verification pipeline.
6. A benchmark design covering multiple downstream idiom understanding tasks.
7. A reproducible repository structure for collection, enrichment, and deep learning.

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

## Project Workflow

```

### Deep Learning

This module prepares task-specific datasets and trains models for multiple benchmark tasks, including:

* idiom detection
* idiom meaning retrieval
* context-to-idiom prediction
* cross-lingual idiom retrieval
* idiom surface-form normalization

A critical design principle is that all train/validation/test splits are created **by idiom, not by row**, to prevent leakage and memorization.

## Benchmark Tasks

IdiomX supports multiple research tasks:

### Idiom Detection

Determine whether a phrase is used idiomatically or literally in context.

### Idiom Meaning Retrieval

Predict the semantic meaning of an idiom in English and Arabic.

### Context-to-Idiom Prediction

Predict the most appropriate idiom from a contextual sentence.

### Cross-Lingual Idiom Retrieval

Retrieve the correct English idiom from an Arabic sentence or semantic context.

### Idiom Surface Normalization

Map contextual idiom surface forms to canonical idiom entries.

## Reproducibility

This repository is structured for full reproducibility. Each major module has its own dedicated `README.md` with:

* overview
* inputs and outputs
* step-by-step execution instructions
* required files
* code entry points
* recommended run order

For environment setup and detailed execution instructions, refer to the module-specific README files.

## Research Perspective

IdiomX is intended as a benchmark-oriented research artifact rather than only a static dataset. The project is designed to support publication in NLP venues focused on lexical semantics, figurative language, multilingual NLP, and low-resource semantic transfer.

The LLM enrichment stage demonstrates that structured generation and targeted verification can be used to construct a high-quality bilingual idiom resource at scale. The downstream modeling stage is intended to evaluate how well neural architectures can generalize to unseen idioms and cross-lingual contexts.

## Status

Current project checkpoint:


* Data collection: completed
* LLM enrichment: completed
* Dataset verification: completed
here:
https://github.com/aymanshar/idiomx-dataset

* Deep learning benchmark preparation: in progress

## Citation

If you use IdiomX in academic work, please cite the associated paper once available.

* * *