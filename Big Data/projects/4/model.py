from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline
from pyspark.ml.feature import HashingTF, Tokenizer

tokenizer = Tokenizer(inputCol="reviewText", outputCol="words")
hasher = HashingTF(numFeatures=100, binary=True, inputCol=tokenizer.getOutputCol(), outputCol="features")
lr = LinearRegression(featuresCol=hasher.getOutputCol(), labelCol="overall", regParam=0.3)

pipeline = Pipeline(stages=[
  tokenizer,
  hasher,
  lr
])