from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, expr, when

# Local PySpark version of the cleaning steps in the Databricks notebook.
spark = (
    SparkSession.builder
    .appName("AmazonReviewsDataCleaning")
    .master("local[*]")
    .getOrCreate()
)

# Keep amazon_reviews.csv in the same folder as this script.
input_path = r"C:\Users\Tenzin Choeyang D\OneDrive\Desktop\spark\amazon_reviews.csv"
output_dir = Path("cleaned_data")
output_dir.mkdir(parents=True, exist_ok=True)

# BRONZE: read the raw CSV using the notebook's CSV options.
df_bronze = (
    spark.read.format("csv")
    .option("header", "true")
    .option("inferSchema", "false")
    .option("quote", '"')
    .option("escape", '"')
    .option("multiLine", "true")
    .load(input_path)
)

print("\n--- BRONZE: RAW DATA ---")
print("Raw rows:", df_bronze.count())
df_bronze.printSchema()
df_bronze.select(
    "reviewerName", "overall", "reviewText", "reviewTime", "day_diff"
).show(5, truncate=False)

# SILVER: remove the index column, remove rows without review text,
# and convert numeric columns to numeric types as in the notebook.
df_silver = df_bronze.drop("_c0").dropna(subset=["reviewText"])

numeric_casts = {
    "overall": "double",
    "day_diff": "int",
    "helpful_yes": "int",
    "helpful_no": "int",
    "total_vote": "int",
    "score_pos_neg_diff": "int",
    "score_average_rating": "double",
    "wilson_lower_bound": "double",
}
for name, data_type in numeric_casts.items():
    if name in df_silver.columns:
        df_silver = df_silver.withColumn(
            name, expr(f"try_cast(`{name}` AS {data_type})")
        )

# Keep rows with valid ratings for sentiment classification.
df_silver = df_silver.filter(
    col("reviewText").isNotNull()
    & col("overall").isNotNull()
    & col("overall").between(1, 5)
)

print("\n--- SILVER: CLEANED DATA ---")
print("Silver rows:", df_silver.count())
df_silver.select(
    "reviewerName", "overall", "reviewText", "reviewTime", "day_diff"
).show(5, truncate=False)

silver_path = str(output_dir / "amazon_reviews_silver")
df_silver.write.mode("overwrite").parquet(silver_path)
print("Silver Parquet saved to:", silver_path)

# GOLD: retain the modelling columns and create sentiment labels,
# using the same rating rules as the notebook.
df_gold = df_silver.select("reviewText", "overall").withColumn(
    "sentiment",
    when(col("overall") <= 2, "negative")
    .when(col("overall") == 3, "neutral")
    .when(col("overall") >= 4, "positive")
)
df_gold = df_gold.filter(
    col("reviewText").isNotNull()
    & (col("reviewText") != "")
    & col("overall").isNotNull()
    & col("sentiment").isNotNull()
)

print("\n--- GOLD: DATA FOR SENTIMENT MODEL ---")
print("Gold rows:", df_gold.count())
df_gold.groupBy("sentiment").count().orderBy("sentiment").show()
df_gold.select("reviewText", "overall", "sentiment").show(10, truncate=False)

gold_path = str(output_dir / "amazon_reviews_gold")
df_gold.write.mode("overwrite").parquet(gold_path)
print("Gold Parquet saved to:", gold_path)

# Export a CSV folder too, for easy inspection or reuse.
csv_path = str(output_dir / "amazon_reviews_gold_csv")
df_gold.coalesce(1).write.mode("overwrite").option("header", "true").csv(csv_path)
print("Gold CSV folder saved to:", csv_path)
print("Inside it, look for the file whose name starts with 'part-'.")

spark.stop()