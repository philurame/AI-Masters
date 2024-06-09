#!/bin/bash

HADOOP_EXE=/usr/bin/yarn
HADOOP_STREAM_JAR=/usr/lib/hadoop-mapreduce/hadoop-streaming.jar

FILES=$1
INPUT=$2
OUTPUT=$3
MAPPER=$4
REDUCER=$5
$HADOOP_EXE jar $HADOOP_STREAM_JAR -files $FILES -input $INPUT -output $OUTPUT -mapper "$MAPPER" -reducer "$REDUCER"

#usage:
#mapper:  filter.py 
#reducer: predict.py
#files projects/1a/filter.py,projects/1a/predict.py,projects/1a/filter_cond.py,1a.joblib,projects/1a/model.py