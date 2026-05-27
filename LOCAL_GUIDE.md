# Local Guide: Chay Pipeline Bang Docker Compose

Tai lieu nay dung cho moi truong local Windows/Docker Compose.

## 1. Docker mount va vi tri file

`docker-compose.yml` mount ca repo vao Airflow/Spark:

```text
./ -> /opt/airflow
./ -> /opt/spark/work-dir
```

Vi vay cac path local can dat theo convention code hien tai:

```text
crawlers/data/                         -> /opt/airflow/crawlers/data
data/stopwords_vi.txt                  -> /opt/airflow/data/stopwords_vi.txt
data/slang_dict.json                   -> /opt/airflow/data/slang_dict.json
models/phobert_finetuned/final/        -> /opt/airflow/models/phobert_finetuned/final
```

Neu model hien co o `models/phobert_finetuned_v2/final`, co 2 cach:

- Copy/rename thanh `models/phobert_finetuned/final` de dung default env.
- Hoac sua `.env`: `LOCAL_SENTIMENT_MODEL_PATH=/opt/airflow/models/phobert_finetuned_v2/final`.

## 2. Start cluster

```powershell
docker-compose up -d
docker-compose ps
```

Neu can reset sach Docker volume:

```powershell
docker-compose down -v
docker-compose up -d
```

## 3. Chay full pipeline dung Airflow UI

1. Airflow UI: <http://localhost:8081>
2. Login: `admin/admin`
3. Unpause DAG `full_processing_pipeline`
4. Trigger DAG

DAG se tu chay:

```text
validate_runtime_mounts
  -> crawl_sources
  -> upload_reference_files_to_hdfs
  -> spark_cleaning
  -> ClickHouse ingest/dbt tasks
```

## 4. Lenh kiem tra nhanh

Kiem tra mount trong container:

```powershell
docker exec airflow-scheduler ls /opt/airflow/crawlers/data/voz
docker exec airflow-scheduler ls /opt/airflow/data/stopwords_vi.txt
docker exec airflow-scheduler ls /opt/airflow/data/slang_dict.json
docker exec airflow-scheduler ls /opt/airflow/models/phobert_finetuned/final
```

Kiem tra HDFS sau khi DAG chay:

```powershell
docker exec namenode hdfs dfs -ls /user/root/ref
docker exec namenode hdfs dfs -find /user/root/raw_data -type f
docker exec namenode hdfs dfs -find /user/root/staged/stg_posts_core -name "*.parquet"
```

Theo doi log:

```powershell
docker-compose logs -f airflow-scheduler
```

## 5. Luu y Windows

Khong dung `chmod` trong Windows CMD/PowerShell. Lenh `chmod` chi co trong Linux/Git Bash/WSL. Voi flow local hien tai, nen trigger pipeline bang Airflow UI thay vi chay shell script truc tiep.
HDFS Web UI: http://localhost:9870
Spark Master: http://localhost:8080
Airflow UI: http://localhost:8081 (\�dmin\ / \�dmin)
ClickHouse: http://localhost:8123
Dashboard: \streamlit run dashboard/app.py