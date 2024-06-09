#!/bin/bash

HADOOP_EXE=/usr/bin/yarn
HADOOP_STREAM_JAR=/usr/lib/hadoop-mapreduce/hadoop-streaming.jar

FILES=$1
INPUT=$2
OUTPUT=$3
MAPPER=$4

$HADOOP_EXE jar $HADOOP_STREAM_JAR -files $FILES -D mapred.reduce.tasks=0 -input $INPUT -output $OUTPUT -mapper "$MAPPER"

#usage:
#файлы, которые надо послать вместе с задачей, через запятую
#путь к входному файлу
#путь к выходному файлу
#имя файла с программой маппером, то есть `filter.py`