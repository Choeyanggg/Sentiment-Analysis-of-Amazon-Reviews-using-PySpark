from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Save graphs without opening windows
import matplotlib.pyplot as plt

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, HashingTF, IDF, StringIndexer, IndexToString
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics


# ------------------------------------------------------------
# 1. Configuration
# ------------------------------------------------------------
CSV_PATH = r"C:\Users\Tenzin Choeyang D\OneDrive\Desktop\spark\amazon_reviews.csv"
OUTPUT_DIR = Path("amazon_sentiment_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 2. Start Spark and load the dataset
# ------------------------------------------------------------
spark = (
    SparkSession.builder
    .appName("AmazonReviewSentimentAnalysis")
    .master("local[*]")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

try:
    print("\n=== 1. LOADING DATA ===")
    df = spark.read.csv(
        CSV_PATH,
        header=True,
        inferSchema=False,
        quote='"',
        escape='"',
        multiLine=True
    )

    print("Original columns:", df.columns)
    print("Rows loaded:", df.count())
    print("\nOriginal schema:")
    df.printSchema()

    required_columns = {"reviewText", "overall"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            "Check that CSV_PATH points to the correct CSV file."
        )

    # Keep only columns needed for this classification task.
    df = df.select("reviewText", "overall")
    df = df.withColumn("overall", col("overall").cast("double"))
    df = df.filter(
        col("reviewText").isNotNull()
        & (col("reviewText") != "")
        & col("overall").isNotNull()
        & col("overall").between(1, 5)
    )

    df = df.withColumn(
        "sentiment",
        when(col("overall") <= 2, "negative")
        .when(col("overall") == 3, "neutral")
        .when(col("overall") >= 4, "positive")
    ).filter(col("sentiment").isNotNull())

    print("\n=== 2. CLEANED DATA ===")
    print("Rows after cleaning:", df.count())
    print("Sentiment distribution:")
    df.groupBy("sentiment").count().orderBy("sentiment").show()

    # Collect only the small aggregation for plotting.
    sentiment_counts = (
        df.groupBy("sentiment").count().orderBy("sentiment").collect()
    )
    labels = [row["sentiment"] for row in sentiment_counts]
    counts = [row["count"] for row in sentiment_counts]

    plt.figure(figsize=(7, 5))
    plt.bar(labels, counts)
    plt.title("Amazon Review Sentiment Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Number of Reviews")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "sentiment_distribution.png", dpi=300)
    plt.close()
    print("Saved:", OUTPUT_DIR / "sentiment_distribution.png")
            outputCol="labels",
        handleInvalid="error"
    )

    tfidf_pipeline = Pipeline(
        stages=[tokenizer, hashing_tf, idf, indexer]
    )
    tfidf_model = tfidf_pipeline.fit(train_df)
    train_processed = tfidf_model.transform(train_df)
    test_processed = tfidf_model.transform(test_df)

    print("Example processed records:")
    train_processed.select(
        "reviewText", "sentiment", "labels"
    ).show(5, truncate=80)

    # ------------------------------------------------------------
    # 4. Train Logistic Regression
    # ------------------------------------------------------------
    print("\n=== 5. TRAINING LOGISTIC REGRESSION ===")
    lr = LogisticRegression(
        featuresCol="features",
        labelCol="labels",
        maxIter=50,
        regParam=0.01
    )
    lr_model = lr.fit(train_processed)
    print("Model training completed.")

    # ------------------------------------------------------------
    # 5. Predict and evaluate
    # ------------------------------------------------------------
    print("\n=== 6. EVALUATION ===")
    predictions = lr_model.transform(test_processed)

    accuracy_eval = MulticlassClassificationEvaluator(
        labelCol="labels",
        predictionCol="prediction",
        metricName="accuracy"
    )
    f1_eval = MulticlassClassificationEvaluator(
        labelCol="labels",
        predictionCol="prediction",
        metricName="f1"
    )

    accuracy = accuracy_eval.evaluate(predictions)
    f1 = f1_eval.evaluate(predictions)

    print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Weighted F1 Score: {f1:.4f}")

    # Convert predicted numeric class back to its sentiment name.
    label_model = tfidf_model.stages[3]
    converter = IndexToString(
        inputCol="prediction",
        outputCol="predicted_sentiment",
        labels=label_model.labels
    )
    final_predictions = converter.transform(predictions)

    print("\nSample predictions (actual vs predicted):")
    final_predictions.select(
        "reviewText", "sentiment", "predicted_sentiment"
    ).show(10, truncate=80)

    # Save predictions to one CSV file for easy viewing in Excel.
    # coalesce(1) creates a single data part file in the output folder.
    (
        final_predictions.select(
            "reviewText", "sentiment", "predicted_sentiment"
        )
        .coalesce(1)
        .write.mode("overwrite")
        .option("header", True)
        .csv(str(OUTPUT_DIR / "predictions_csv"))
    )
    print("Saved prediction CSV folder:", OUTPUT_DIR / "predictions_csv")

    # ------------------------------------------------------------
    # 6. Model performance bar chart
    # ------------------------------------------------------------
    plt.figure(figsize=(7, 5))
    plt.bar(["Accuracy", "Weighted F1 Score"], [accuracy, f1])
    plt.title("Logistic Regression Model Performance")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "model_performance.png", dpi=300)
    plt.close()
    print("Saved:", OUTPUT_DIR / "model_performance.png")

    # ------------------------------------------------------------
    # 7. Confusion matrix
    # ------------------------------------------------------------
    print("\n=== 7. CONFUSION MATRIX ===")
    # Collect only the numeric label/prediction pairs needed for the matrix.
    label_prediction_pairs = (
        predictions.select("labels", "prediction")
        .rdd.map(lambda row: (float(row["labels"]), float(row["prediction"])))
    )
    metrics = MulticlassMetrics(label_prediction_pairs)
    matrix = metrics.confusionMatrix().toArray()

    class_names = list(label_model.labels)
    print("Class order:", class_names)
    print("Confusion matrix rows = actual; columns = predicted")
    print(matrix)

    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, interpolation="nearest")
    fig.colorbar(image, ax=ax)
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted Sentiment")
    ax.set_ylabel("Actual Sentiment")
    ax.set_title("Confusion Matrix - Logistic Regression")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(int(matrix[i, j])), ha="center", va="center")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=300)
    plt.close(fig)
    print("Saved:", OUTPUT_DIR / "confusion_matrix.png")
