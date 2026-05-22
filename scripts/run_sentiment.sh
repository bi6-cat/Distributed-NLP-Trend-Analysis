#!/bin/bash
export HDFS_INPUT=hdfs://namenode:9000/user/zett/staged/stg_posts_core
export NLP_MODEL_PATH=/tmp/phobert_finetuned
export CLICKHOUSE_HOST=clickhouse
export PYSPARK_PYTHON=/opt/bitnami/python/bin/python
export PYSPARK_DRIVER_PYTHON=/opt/bitnami/python/bin/python

docker exec -it -e HDFS_INPUT="$HDFS_INPUT" -e CLICKHOUSE_HOST="$CLICKHOUSE_HOST" -e PYSPARK_PYTHON="$PYSPARK_PYTHON" spark-master \
/opt/bitnami/spark/bin/spark-submit \
  --master spark://spark-master:7077 \
  --executor-memory 2g \
  --total-executor-cores 4 \
  --conf spark.executorEnv.PYSPARK_PYTHON=/opt/bitnami/python/bin/python \
  --conf spark.executorEnv.NLP_MODEL_PATH=/tmp/phobert_finetuned \
  /opt/spark/work-dir/spark_jobs/sentiment_job.py
