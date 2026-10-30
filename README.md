# Unveiling Customer Insights: Amazon Customers’ Reviews Analysis with LDA and VADER

Seminar project for **Machine Learning in Economics and Finance** (Jun.-Prof. Dr. Melina Ludolph), Master's programme.
Author: Md Tanjib Mahmud

Topic modeling (**Latent Dirichlet Allocation**) combined with lexicon-based sentiment analysis (**VADER**) to find out *what* customers write about in Amazon reviews and *how positive or negative* each topic is.

## Pipeline

1. **Data** – Amazon Reviews Polarity dataset (Kaggle / Zhang et al. 2015); random sample of 10,000 reviews (`SEED = 42`).
2. **Preprocessing** – lowercase, NLTK tokenization, alphabetic tokens only, stop-word removal (plus a small switchable list of generic words), WordNet lemmatization.
3. **N-grams** – bigrams and trigrams with `gensim.models.phrases`.
4. **Representation** – Gensim dictionary (rare/very common tokens filtered) and raw bag-of-words corpus.
5. **LDA** – 5, 8, 10, 15, 20, 30 topics, with and without n-grams, plus an alpha/eta search; fixed random seed, single-process `LdaModel` so runs are repeatable.
6. **Model selection** – C_v coherence for **every** model; interactive map with pyLDAvis.
7. **Sentiment** – VADER compound score on the original text, aggregated by each review's dominant topic (with review counts per topic).
8. **Validation** – VADER compared with the dataset's own positive/negative label.

## Results (full run on the 3.6M-review file, `SEED = 42`, 10,000-review sample)

The executed notebook with all outputs is `notebooks/lda_vader_amazon_reviews.ipynb`; tables and figures are in `results/`, and the written paper is in `docs/`.

**Sample:** 9,999 reviews after filtering (50.3 % negative / 49.7 % positive); vocabulary 6,410 tokens (6,554 with n-grams).

**Coherence (C_v), alpha = eta = symmetric**

| Topics | Unigrams | + bigrams/trigrams |
|---|---|---|
| 5 | 0.493 | **0.539** |
| 8 | 0.526 | 0.520 |
| 10 | 0.480 | 0.449 |
| 15 | 0.467 | 0.462 |
| 20 | 0.438 | 0.461 |
| 30 | 0.426 | 0.415 |

The highest coherence is the 5-topic model with n-grams (0.539). Coherence falls as topics are added, and n-grams do not help consistently – differences of a few hundredths are within what another seed can change. All 24 models are in `results/coherence_results.csv`. Note that `eta="auto"` is not used: in gensim's `LdaModel` it leaves the topics unchanged.

**Topics of the selected model and their sentiment (VADER)**

| Topic | Reviews | Mean compound | Positive | Negative | Top words |
|---|---|---|---|---|---|
| 3 | 1,523 | 0.558 | 81 % | 16 % | album, cd, song, music, sound, love, best, fan, dvd, band |
| 0 | 3,017 | 0.412 | 73 % | 24 % | book, read, story, author, reading, time, life, character |
| 2 | 968 | 0.359 | 70 % | 25 % | game, use, play, work, better, buy, need, problem, computer |
| 4 | 2,809 | 0.298 | 66 % | 30 % | product, work, bought, use, time, used, amazon, buy, price, quality |
| 1 | 1,682 | 0.286 | 66 % | 32 % | movie, film, time, see, watch, love, dvd, show, bad |

All five topics have positive average sentiment; music is the most positive, movies and general products the least.

**VADER check:** against the dataset's own positive/negative label VADER is neutral on 3.5 % of reviews and correct on **70.4 %** of the rest, so topic-level sentiment is a rough indicator only.

| File | Content |
|---|---|
| `results/coherence_results.csv` | C_v coherence of every trained model |
| `results/topic_sentiment_summary.csv` | per topic: review count, mean sentiment, sentiment shares, top words |
| `results/lda_visualization_best_model.html` | pyLDAvis map of the selected model |
| `results/figures/*.png` | coherence by topic count, sentiment by topic |

The seminar paper is available as a [PDF](docs/seminar_paper_lda_vader.pdf).

## Repository structure

```
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── lda_vader_amazon_reviews.ipynb       # full analysis (executed, with outputs)
├── results/
│   ├── coherence_results.csv
│   ├── topic_sentiment_summary.csv
│   ├── topic_example_reviews.csv
│   ├── lda_visualization_best_model.html
│   └── figures/
├── docs/
│   └── seminar_paper_lda_vader.pdf
└── data/
    ├── README.md                            # how to get the data
    └── DATASET_README.txt
```

## How to run

Python 3.10–3.12 (gensim 4.3.3 needs NumPy < 2, which has no wheels for Python 3.13).

```bash
git clone https://github.com/ray4096/amazon-reviews-lda-vader.git
cd amazon-reviews-lda-vader
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

1. Download the dataset ([`data/README.md`](data/README.md)) and save it as `data/amazon_review.csv`.
2. `jupyter notebook notebooks/lda_vader_amazon_reviews.ipynb` and run all cells (takes a while: 24 LDA models are trained and scored).
3. In Colab: upload the repo or mount Drive; the notebook also checks a Google Drive path (see the configuration cell) and the path in the `AMAZON_CSV` environment variable.

## Limitations

- Only 10,000 of about 3.6 million reviews; results can change with another sample or seed.
- LDA topics on short reviews overlap and topic labels are subjective.
- Coherence differences of a few hundredths are within seed noise – check with other `SEED` values.
- VADER is lexicon-based (sarcasm, domain language); the notebook measures its agreement with the dataset label.
- The dominant topic ignores that reviews mix topics.

## References

- Blei, Ng & Jordan (2003). *Latent Dirichlet Allocation.* JMLR.
- Hutto & Gilbert (2014). *VADER.* ICWSM.
- Röder, Both & Hinneburg (2015). *Exploring the space of topic coherence measures.* WSDM.
- Zhang, Zhao & LeCun (2015). *Character-level Convolutional Networks for Text Classification.* NIPS.
- McAuley & Leskovec (2013). *Hidden factors and hidden topics.* RecSys.

## License

MIT – see [LICENSE](LICENSE). The dataset has its own terms; see the Kaggle page.
