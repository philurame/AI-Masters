from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()
spark.sparkContext.setLogLevel('WARN')
from model import pipeline
from pyspark.sql.types import *
import pyspark.sql.functions as f
import sys
train_path = sys.argv[1]
save_model_path = sys.argv[2]


schema = StructType(fields=[
    StructField("id", IntegerType()),
    StructField("overall", IntegerType()),
    StructField("vote", StringType()),
    StructField("verified", BooleanType()),
    StructField("reviewTime", StringType()),
    StructField("reviewerID", StringType()),
    StructField("asin", StringType()),
    StructField("reviewerName", StringType()),
    StructField("reviewText", StringType()),
    StructField("summary", StringType()),
    StructField("unixReviewTime", LongType()),
])
df_train = spark.read.json(train_path, schema=schema)
# df_train = df_train.withColumn("text", f.concat(f.col("reviewerName"), f.lit(". "), f.col("reviewText"), f.lit(". "), f.col("summary")))
df_train = df_train.dropna(subset=["reviewText"]).select("id", "reviewText", "overall")

pipeline.fit(df_train).write().overwrite().save(save_model_path)
