# ReviewBuster NLP ML Project

## Assignment 2

- [Notebook 1](a2_eda.ipynb) — data exploration and preprocessing
- [Notebook 2](a2_kFoldCV.ipynb) — hotel-grouped cross-validation and model selection
- [Notebook 3](a2_finalTraining_testEvaluation.ipynb) — SVM tuning, final training, and held-out test evaluation
- [Review dataset](data/deceptive-opinion.csv)
- [Assignment specification](a2_specification.pdf)

Open the notebooks from the repository root so their relative dataset paths resolve correctly. Notebook 3 uses the selected linear SVM with non-lemmatised TF-IDF unigrams and bigrams, selects `C=6.0` through training-only validation, and evaluates the final model on four held-out hotels.
