# Sentiment Analysis of Amazon Reviews using PySpark

A PySpark and Spark MLlib workflow that classifies Amazon product reviews as positive, neutral or negative using only the review text. Built for INT 315 (Cluster Computing) at Lovely Professional University.

## Pipeline

1. **Load** a multi-line CSV with Spark DataFrames (`quote`, `escape` and `multiLine` options).
2. **Clean**: cast the rating to double, keep `reviewText` and `overall`, drop null or empty rows (4,915 to 4,914 rows).
3. **Label**: derive sentiment from the star rating (1-2 negative, 3 neutral, 4-5 positive).
4. **Split**: 80/20 train-test split with a fixed seed.
5. **Features**: Spark ML Pipeline of Tokenizer, HashingTF (10,000 features), IDF and StringIndexer, fitted on training data only to avoid leakage.
6. **Model**: Logistic Regression (`maxIter=50`).
7. **Evaluate** with accuracy and weighted F1 on the held-out test set.

## Results

| Metric | Value |
|---|---|
| Accuracy | 0.8539 |
| Weighted F1 | 0.8743 |
| Test rows | 938 |

The dataset is heavily imbalanced (90.5% positive, 6.6% negative, 2.9% neutral), so a model that always predicts "positive" would score about 90% accuracy. Accuracy here is therefore not impressive on its own. The script prints the majority-class baseline and a confusion matrix so per-class behaviour can be inspected.

## Limitations

- Labels come from star ratings, not human annotation.
- Neutral reviews are rare (142 in total).
- One model and one parameter set were tried, with no cross-validation.
- TF-IDF ignores word order and context.
- Run locally in single-machine mode, not on a multi-node cluster.

## Run it

```bash
pip install pyspark matplotlib
python sentiment_analysis_pyspark.py amazon_reviews.csv
```

Dataset: [hugginglearners/amazon-reviews-sentiment-analysis](https://huggingface.co/datasets/hugginglearners/amazon-reviews-sentiment-analysis) (save as `amazon_reviews.csv`).

## Future work

Class weights or resampling, per-class precision and recall, `CrossValidator` tuning, other models (Naive Bayes, Linear SVM), stop-word removal and n-grams.
