# Persian-Sentiment-LLM-Augmentation

This repository contains the official implementation of the paper:  
**"LLM-based synthetic data augmentation for Persian sentiment classification"**

(*This paper is currently under review for publication.)

## Overview

Persian sentiment analysis suffers from limited labeled data, and the effectiveness of LLM-based synthetic data augmentation for this language has not been systematically evaluated. This work addresses this gap by using **DeepSeek-V3** and **GPT-4o mini** to generate synthetic samples on top of three baseline datasets derived from **SentiPers**. Nine Transformer architectures are evaluated across three random seeds under a unified optimization framework.

The empirical utility of synthetic augmentation proves **highly task-dependent**:

- **Binary classification** exhibits an apparent performance ceiling, with only marginal variations.
- **Five-class classification** shows substantial practical benefits, with synthetic expansion improving downstream performance in **up to 8 of 9 model architectures**.

Statistical significance testing and intrinsic lexical audits confirm that targeted synthetic expansion enriches fine-grained lexical diversity, providing an effective low-resource strategy for resolving complex sentiment boundaries.

## Results

Confusion matrices for all models in 5-Class task are available in [`results/confusion_matrices/5_Class`](https://github.com/Persian-NLP-Research/Persian-Sentiment-LLM-Augmentation/tree/main/results/confusion_matrices/5_Class).
Full results will be added after paper acceptance.

## Getting Started
**Requirements:** Python ≥ 3.9, `numpy`, `pandas`, `scipy`. A GPU is recommended for training but not required for the statistical analysis. 

**Repository Structure:**

```
.
├── Demo/                           # Some executed demo notebooks
├── notebooks/                      # Training notebooks (empty outputs)
├── statistical_significance_tests/ # Statistical analysis of results
├── datasets/                       # Datasets and quality audit script
└── results/
    └── confusion_matrices/         # Confusion matrices
```
Each sub-folder contains its own `README.md` with detailed instructions.

**A typical workflow:**

1. Train the models using the notebooks in `notebooks/`.
2. Run the statistical analysis in `statistical_significance_tests/` on the
   results.
3. (Optional) Browse the executed notebooks in `Demo/` to see what outputs
   to expect.

## Models

The experiments use the following pretrained models, all available on Hugging Face. Detailed descriptions of each model are provided on its Hugging Face page and in the paper.

| Model Name | Model Page in Hugging Face |
|----------------|---------------|
| ParsBERT | [`HooshvareLab/bert-base-parsbert-uncased`](https://huggingface.co/HooshvareLab/bert-base-parsbert-uncased) |
| ParsBERT v2.0 | [`HooshvareLab/bert-fa-base-uncased`](https://huggingface.co/HooshvareLab/bert-fa-base-uncased) |
| RoBERTa-fa | [`HooshvareLab/roberta-fa-zwnj-base`](https://huggingface.co/HooshvareLab/roberta-fa-zwnj-base) |
| DistilBERT-fa | [`HooshvareLab/distilbert-fa-zwnj-base`](https://huggingface.co/HooshvareLab/distilbert-fa-zwnj-base) |
| ALBERT-fa | [`HooshvareLab/albert-fa-zwnj-base-v2`](https://huggingface.co/HooshvareLab/albert-fa-zwnj-base-v2) |
| E5-Small | [`agentlans/multilingual-e5-small-aligned-sentiment`](https://huggingface.co/agentlans/multilingual-e5-small-aligned-sentiment) |
| E5-General | [`Ludo33/e5_Sentiment_General_v2`](https://huggingface.co/Ludo33/e5_Sentiment_General_v2) |
| GTE | [`Alibaba-NLP/gte-multilingual-base`](https://huggingface.co/Alibaba-NLP/gte-multilingual-base) |
| GTE-Reranker | [`Alibaba-NLP/gte-multilingual-reranker-base`](https://huggingface.co/Alibaba-NLP/gte-multilingual-reranker-base) |

## Authors
