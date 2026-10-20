This folder contains the datasets used in our experiments, together with a small script to audit their intrinsic quality. See Section 3 of the paper for dataset construction details.

> **Note.** The test set is a separate, fixed file shared across **all**
> experiments. It is never modified or regenerated.

## Source

The baseline datasets are derived from the **SentiPers** corpus and its
**DeepSentiPers** processed release:

- **SentiPers (original corpus):** https://github.com/phosseini/SentiPers
- **DeepSentiPers (processed dataset):** https://github.com/JoyeBright/DeepSentiPers/tree/master/Dataset

Please refer to these repositories for details. The data was normalized and lightly cleaned before use. The version stored in this repository is the post-preprocessing version used in all experiments. The augmented set was generated on top of this baseline using LLM-based data augmentation (see the paper for details).


## Quality Audit

`audit_dataset.py` reports intrinsic quality and diversity statistics for a given CSV, exact-duplicate rate, near-duplicate rates at Jaccard thresholds `0.70` and `0.80`, mean pairwise Jaccard distance, and lexical diversity (TTR / Distinct-1, Distinct-2).
