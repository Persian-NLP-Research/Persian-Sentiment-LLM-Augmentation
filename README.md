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
## Getting started

## Dataset

## Authors
