#!/bin/bash
export HADOOP_USER_NAME="${HADOOP_USER_NAME:-root}"
export HDFS_USER="${HDFS_USER:-$HADOOP_USER_NAME}"
export HDFS_INPUT="hdfs://namenode:9000/user/$HDFS_USER/staged/stg_posts_core"
export NLP_MODEL_PATH=/tmp/phobert_finetuned
export CLICKHOUSE_HOST=clickhouse
export PYSPARK_PYTHON=/opt/bitnami/python/bin/python
export PYSPARK_DRIVER_PYTHON=/opt/bitnami/python/bin/python

docker exec -it -e HDFS_INPUT="$HDFS_INPUT" -e HADOOP_USER_NAME="$HDFS_USER" -e HDFS_USER="$HDFS_USER" -e CLICKHOUSE_HOST="$CLICKHOUSE_HOST" -e PYSPARK_PYTHON="$PYSPARK_PYTHON" spark-master \
/opt/bitnami/spark/bin/spark-submit \
  --master spark://spark-master:7077 \
  --executor-memory 2g \
  --total-executor-cores 4 \
  --conf spark.executorEnv.PYSPARK_PYTHON=/opt/bitnami/python/bin/python \
  --conf spark.executorEnv.HADOOP_USER_NAME="$HDFS_USER" \
  --conf spark.executorEnv.HDFS_USER="$HDFS_USER" \
  --conf spark.executorEnv.NLP_MODEL_PATH=/tmp/phobert_finetuned \
  /opt/spark/work-dir/spark_jobs/sentiment_job.py
