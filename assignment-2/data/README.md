# Preprocessed review dataframes

These eight TF-IDF compressed pandas pickle files were generated from `a2_eda.ipynb`.
Each representation has separate training and validation tables:

| Representation | Training file | Validation file |
|---|---|---|
| Unigrams, no lemmatisation | `unigram_training_df.pkl.gz` | `unigram_validation_df.pkl.gz` |
| Unigrams + bigrams, no lemmatisation | `unigram_bigram_training_df.pkl.gz` | `unigram_bigram_validation_df.pkl.gz` |
| Lemmatised unigrams | `lemma_unigram_training_df.pkl.gz` | `lemma_unigram_validation_df.pkl.gz` |
| Lemmatised unigrams + bigrams | `lemma_unigram_bigram_training_df.pkl.gz` | `lemma_unigram_bigram_validation_df.pkl.gz` |

Load a table in a training notebook with, for example:

```python
import pandas as pd
training_df = pd.read_pickle("data/unigram_training_df.pkl.gz")
```

The two lightly cleaned training files, `lightly_cleaned_training_df.pkl.gz` and `lightly_cleaned_lemmatised_training_df.pkl.gz`, contain the 1,277 training reviews used in `a2_kFoldCV.ipynb`. They retain `text`, `deceptive`, `hotel`, `polarity`, and `source`. Notebook 2 groups folds by `hotel` and fits a fresh TF-IDF vectorizer on each run's training folds. It uses `text` as the prediction input and `deceptive` as the target; `polarity` and `source` are not prediction features.

The eight precomputed TF-IDF tables are earlier Notebook 1 outputs. Each retains `deceptive`, `hotel`, `polarity`, and `source` before the `tfidf_...` feature columns. Training and validation columns match within each representation.
