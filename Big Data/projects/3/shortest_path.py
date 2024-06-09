#! /opt/conda/envs/dsenv/bin/python3

import sys, os
id_from = sys.argv[1]
id_to = sys.argv[2]
path_dataset = sys.argv[3]
path_output = sys.argv[4]

SPARK_HOME = "/usr/lib/spark3"
PYSPARK_PYTHON = "/opt/conda/envs/dsenv/bin/python"
os.environ["PYSPARK_PYTHON"]= PYSPARK_PYTHON
os.environ["PYSPARK_DRIVER_PYTHON"]= PYSPARK_PYTHON
os.environ["SPARK_HOME"] = SPARK_HOME
PYSPARK_HOME = os.path.join(SPARK_HOME, "python/lib")
sys.path.insert(0, os.path.join(PYSPARK_HOME, "py4j-0.10.9.5-src.zip"))
sys.path.insert(0, os.path.join(PYSPARK_HOME, "pyspark.zip"))

from pyspark import SparkConf
from pyspark.sql import SparkSession
conf = SparkConf()
conf.set("spark.ui.port", "4099")
spark = SparkSession.builder.config(conf=conf).appName("Spark SQL").getOrCreate()
from pyspark.sql.types import *
import pyspark.sql.functions as pf
from pyspark.sql.functions import udf

schema = StructType(fields=[
    StructField("user_id", StringType()),
    StructField("follower_id", StringType())
])
df = spark.read\
          .schema(schema)\
          .format("csv")\
          .option("sep", "\t")\
          .load(path_dataset)


df = df.withColumn("paths", udf(lambda: [], ArrayType(StringType()))())
df.cache()

max_path_length = 100

# run BFS and mark visited from id_from to id_to
queue_ids = [id_from]
while queue_ids:
  first_in_queue = queue_ids.pop(0)
  neighbours = df.filter(df.follower_id == first_in_queue).select('user_id').rdd.flatMap(lambda x: x).collect()
  if not neighbours: continue # leaf, so useless
  if id_to in neighbours: break # target found

  # add new neighbours with paths of length<max_path_length:
  filter_length_udf = udf(lambda paths: not paths or len(paths[0].split(',')) < max_path_length, BooleanType())
  new_neighbours = df.filter(filter_length_udf(df.paths))\
              .filter(df.follower_id.isin(neighbours))\
              .select('follower_id').rdd.flatMap(lambda x: x).collect()
  queue_ids.extend(list(set(new_neighbours)))

  # append node_id for each path
  current_paths = df.filter(df.follower_id == first_in_queue)\
              .select('paths').rdd.flatMap(lambda x: x).first()
  append_path_udf = udf(lambda node: [i+','+node for i in current_paths] if current_paths else [node], ArrayType(StringType()))
  df = df.withColumn("paths", pf.when(df.follower_id.isin(neighbours), append_path_udf(pf.col("follower_id"))).otherwise(pf.col("paths")))  

is_not_empty_udf = udf(lambda x: len(x) != 0, BooleanType())
paths = df.filter(df.user_id == id_to)\
              .filter(is_not_empty_udf(df.paths))\
              .select('paths').rdd.flatMap(lambda x: x[0]).collect()
paths = [id_from+','+p+','+id_to for p in paths]

# write in HDFS output:
spark.sparkContext.parallelize(['\n'.join(paths)]).saveAsTextFile(path_output)
spark.stop()