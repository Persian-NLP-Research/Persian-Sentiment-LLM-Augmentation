# Training Notebooks

This folder contains the training notebooks used to produce the results
reported in the paper. Each notebook trains and evaluates one model family
on one classification task (5-Class or Binary).

## Contents

| Notebook                                | Family        | Task     |
|-----------------------------------------|---------------|----------|
| `BERT_Family_5Class_Training.ipynb`     | BERT-based    | 5-Class  |
| `BERT_Family_Binary_Training.ipynb`     | BERT-based    | Binary   |
| `E5_Family_5Class_Training.ipynb`       | E5-based      | 5-Class  |
| `E5_Family_Binary_Training.ipynb`       | E5-based      | Binary   |
| `GTE_Family_5Class_Training.ipynb`      | GTE-based     | 5-Class  |
| `GTE_Family_Binary_Training.ipynb`      | GTE-based     | Binary   |

All notebooks share the same pipeline. They differ only in the input data
(task regime) and in the underlying pretrained model.

## How to Run

All notebooks are designed to run **directly in Google Colab**. No local
setup is required.

1. Open [Google Colab](https://colab.research.google.com/).
2. Click **File → Upload notebook** and select one of the `*.ipynb` files.
3. Run the cells in order. The notebook will prompt you to upload:
   - the **training** CSV,
   - the **test** CSV. 
   
   You can find the datasets used in our experiments in "datasets" folder.
4. At the end, the following outputs are produced:
   -  Accuray, Weighted F1-score and Macro F1-score
   -  Confusion matrix
   - `model_predictions.csv` (true vs. predicted labels, downloaded automatically)

## Pipeline Overview

Each notebook follows the same procedure:

1. **Setup** : install pinned versions of `transformers` and `accelerate`,
   set all random seeds (`random`, `numpy`, `torch`, Hugging Face).
2. **Data loading** : upload train and test CSVs, clean the text, and map
   labels to the task-specific label space (5-Class or Binary).
3. **Train/validation split** : 85/15 stratified split of the training set,
   the test set remains untouched.
4. **Tokenization** : truncation and padding to `max_length = 128`.
5. **Model definition** : a task-specific classification head on top of the
   pretrained encoder.
6. **Hyperparameter search** : a small grid over learning rate, batch size,
   and epochs, evaluated on the validation set. The best configuration is
   selected by validation accuracy.
7. **Final training** : the best configuration is retrained on the combined
   train + validation set.
8. **Test evaluation** : final metrics (Accuracy, Weighted-F1,and Macro-F1) and predictions on the test set.

## Input Format

All notebooks expect two CSV files:

- one training file,
- one test file.

Each file must contain **two columns**: `text` (the review) and `label`
(the sentiment class).

- For **5-Class** tasks, labels are mapped as:
  `-2 → 0`, `-1 → 1`, `0 → 2`, `1 → 3`, `2 → 4`.
- For **Binary** tasks, neutral reviews (`0`) are removed, and the remaining
  labels are mapped to `0` (negative) and `1` (positive).

## Notes

- **Family-specific differences.** All notebooks share the same pipeline, but
  they differ in two implementation details:
  - **BERT-based models** use `AutoModelForSequenceClassification` (a standard
    sequence-classification head).
  - **E5- and GTE-based models** use `AutoModel` with **mean pooling** followed
    by a linear classification head, since these are embedding models.
- **Instruction prefixes.** E5 and GTE tokenizers prepend a `query: ` prefix
  to each input, as recommended by the model cards.
- **Hyperparameter grids.** All families use the same grid
  (`lr ∈ {2e-5, 3e-5, 5e-5}`, `bs ∈ {8, 16, 32}`, `epochs = 3`), except where
  noted in the notebook. For the larger E5 and GTE models, the two
  `bs = 32` configurations were excluded to fit within memory limits.
- **Reproducibility.** A fixed seed (`SEED = 42` by default) is set at the
  beginning of each notebook. Note that the seed may be adjusted in individual
  notebooks depending on the experiment.
