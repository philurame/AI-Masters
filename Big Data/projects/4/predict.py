'''
Скрипт для предсказания должен принимать следующие аргументы:
путь к сохраненной в HDFS обученной модели
путь к тестовому датасету
путь для cохранения предсказаний в HDFS.
В скрипте для предсказания вам уже не нужно импортировать модель из model.py, но надо загрузить ее с HDFS.
'''
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()
spark.sparkContext.setLogLevel('WARN')
from pyspark.ml import Pipeline, PipelineModel
from pyspark.sql.types import *
import pyspark.sql.functions as f
import sys
model_path = sys.argv[1]
test_path = sys.argv[2]
save_preds_path = sys.argv[3]

model = PipelineModel.load(model_path)

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

df_test = spark.read.json(test_path, schema=schema)
# df_test = df_test.withColumn("text", f.concat(f.col("reviewerName"), f.lit(". "), f.col("reviewText"), f.lit(". "), f.col("summary")))
df_test = df_test.fillna(subset=["reviewText"], value="").select("id", "reviewText")

predictions = model.transform(df_test).select("id", "prediction")
predictions = predictions.withColumn("prediction", f.when(f.col("prediction") > 5, 5).otherwise(f.when(f.col("prediction") < 1, 1).otherwise(f.col("prediction"))))


predictions.write().overwrite().save(save_preds_path)
