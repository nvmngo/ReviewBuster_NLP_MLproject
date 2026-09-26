# Preprocessed review dataframes

These eight compressed pandas pickle files were generated from `a2-report.ipynb`.
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

Each table retains `deceptive`, `hotel`, `polarity`, and `source` before the sparse `tfidf_...` feature columns. For model inputs, use the TF-IDF columns and optionally `polarity`. Use `deceptive` as the target; keep `hotel` and `source` for grouping and auditing, not prediction. Training and validation columns match within each representation.
