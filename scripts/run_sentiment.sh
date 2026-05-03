#!/bin/bash
export HDFS_INPUT=hdfs://192.168.56.11:9000/user/zett/staged/stg_posts_core
export NLP_MODEL_PATH=/tmp/phobert_finetuned
export CLICKHOUSE_HOST=192.168.56.14
export PYSPARK_PYTHON=/opt/miniconda/envs/nlp-trend/bin/python
export PYSPARK_DRIVER_PYTHON=/opt/miniconda/envs/nlp-trend/bin/python

/opt/spark/bin/spark-submit --master spark://192.168.56.11:7077 --executor-memory 2g --total-executor-cores 4 --conf spark.executorEnv.PYSPARK_PYTHON=/opt/miniconda/envs/nlp-trend/bin/python --conf spark.executorEnv.NLP_MODEL_PATH=/tmp/phobert_finetuned /vagrant/spark_jobs/sentiment_job.py
