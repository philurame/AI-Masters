#!/bin/bash

HADOOP_EXE=/usr/bin/yarn
HADOOP_STREAM_JAR=/usr/lib/hadoop-mapreduce/hadoop-streaming.jar

FILES=$1
INPUT=$2
OUTPUT=$3
MAPPER=$4

$HADOOP_EXE jar $HADOOP_STREAM_JAR -files $FILES -D mapred.reduce.tasks=0 -input $INPUT -output $OUTPUT -mapper $MAPPER


# usage:
# cd ai-masters-bigdata
# hdfs dfs -rm -r -f -skipTrash predicted.csv
# 1st arg - projects/1a/predict.py,1a.joblib,projects/1a/model.py
# 2nd arg - /datasets/criteo/test-with-id-50.txt
# 3rd arg - predicted.csv
# 4th arg - predict.py