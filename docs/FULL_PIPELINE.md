# Full Pipeline Airflow

Tai lieu nay mo ta luong Airflow chinh cua project **Tech Trend and Controversy Radar**: tu raw data, Spark processing, HDFS staged Parquet, ClickHouse staging, dbt marts, den dashboard NextJS.

Pipeline chinh nam trong [`dags/processing_dag.py`](../dags/processing_dag.py), DAG id:

```text
full_processing_pipeline
```

Schedule mac dinh:

```text
0 2 * * *  # 02:00 hang ngay
```

Trong local Docker Compose, cach test dung la:

```text
docker-compose up -d
mo Airflow UI tai http://localhost:8081
trigger DAG full_processing_pipeline
```

## Tong Quan Luong

```text
pipeline_start
  -> validate_runtime_mounts
  -> crawl_sources
  -> upload_reference_files_to_hdfs
  -> spark_cleaning
  -> ingest_stg_posts_core
      -> sentiment_analysis
      -> ingest_stg_posts_nlp
      -> cms_keyword_counting
      -> ingest_stg_keyword_freq

      -> lda_topic_modeling
      -> ingest_stg_post_topics
      -> ingest_stg_topics

  -> crisis_to_hdfs_stg_crisis_events
  -> ingest_stg_crisis_events
  -> dbt_transform
  -> pipeline_end
```

Y tuong chinh:

- Spark jobs chi ghi staged data ra HDFS/local Parquet.
- ClickHouse khong doc truc tiep raw data. ClickHouse load 6 staged datasets tu HDFS bang `hdfs()`.
- dbt doc cac staging tables trong ClickHouse de build intermediate/mart tables.
- Dashboard NextJS query cac bang dbt mart trong ClickHouse.

## Path Contract

Root HDFS staged duoc cau hinh trong [`dags/processing_dag.py`](../dags/processing_dag.py):

```text
HDFS_STAGED_ROOT=hdfs://namenode:9000/user/root/staged
```

6 dataset staged chinh:

```text
stg_posts_core      -> hdfs://namenode:9000/user/root/staged/stg_posts_core
stg_posts_nlp       -> hdfs://namenode:9000/user/root/staged/stg_posts_nlp
stg_post_topics     -> hdfs://namenode:9000/user/root/staged/stg_post_topics
stg_topics          -> hdfs://namenode:9000/user/root/staged/stg_topics
stg_keyword_freq    -> hdfs://namenode:9000/user/root/staged/stg_keyword_freq
stg_crisis_events   -> hdfs://namenode:9000/user/root/staged/stg_crisis_events
```

Local Docker mount hien tai mount ca repo vao `/opt/airflow`, nen cac folder quan trong la:

```text
crawlers/data/...                    -> /opt/airflow/crawlers/data/...
data/stopwords_vi.txt                -> /opt/airflow/data/stopwords_vi.txt
data/slang_dict.json                 -> /opt/airflow/data/slang_dict.json
models/...                           -> /opt/airflow/models/...
```

## Step 1: pipeline_start

Code: [`dags/processing_dag.py`](../dags/processing_dag.py)

Task `pipeline_start` la `DummyOperator`, dung lam diem bat dau de Airflow Graph View ro rang hon. Task nay khong xu ly data.

## Step 2: validate_runtime_mounts

Code: function `task_validate_runtime_mounts()` trong [`dags/processing_dag.py`](../dags/processing_dag.py)

Muc dich:

- Fail som neu Docker volume mount chua dung.
- Kiem tra raw crawler data co ton tai trong `/opt/airflow/crawlers/data`.
- Kiem tra reference files cho NLP/topic modeling:
  - `/opt/airflow/data/stopwords_vi.txt`
  - `/opt/airflow/data/slang_dict.json`
- Kiem tra sentiment model theo env `LOCAL_SENTIMENT_MODEL_PATH`.

Raw files toi thieu pipeline dang expect:

```text
/opt/airflow/crawlers/data/voz/comments.csv
/opt/airflow/crawlers/data/voz/posts.csv
/opt/airflow/crawlers/data/vnexpress/comment_vnexpress.csv
/opt/airflow/crawlers/data/vnexpress/post_vnexpress.csv
```

Sentiment model toi thieu can co:

```text
config.json
tokenizer_config.json
model.safetensors hoac pytorch_model.bin
vocab.txt hoac bpe.codes
```

Neu step nay fail thi nen sua cau truc file/mount truoc khi chay tiep, vi cac task sau se fail kho doc hon.

## Step 3: crawl_sources

Code:

- Airflow task trong [`dags/processing_dag.py`](../dags/processing_dag.py)
- Upload script: [`crawlers/upload_to_hdfs.py`](../crawlers/upload_to_hdfs.py)

Task `crawl_sources` hien dang chay:

```bash
python3 -u /opt/airflow/crawlers/upload_to_hdfs.py
```

Trong ban local test hien tai, cac crawler mang that dang duoc comment de tranh phu thuoc network. Task nay upload raw data co san trong:

```text
/opt/airflow/crawlers/data
```

len HDFS raw zone:

```text
hdfs://namenode:9000/user/root/raw_data
```

Day la input cho Spark cleaning job.

## Step 4: upload_reference_files_to_hdfs

Code: function `task_upload_reference_files_to_hdfs()` trong [`dags/processing_dag.py`](../dags/processing_dag.py)

Muc dich:

- Dua NLP reference files tu local mount len HDFS de Spark executor doc duoc trong cluster mode.
- Khong sua logic crawler cu.

Input local:

```text
/opt/airflow/data/stopwords_vi.txt
/opt/airflow/data/slang_dict.json
```

Output HDFS:

```text
hdfs://namenode:9000/user/root/ref/stopwords_vi.txt
hdfs://namenode:9000/user/root/ref/slang_dict.json
```

LDA/topic modeling va mot so logic tokenize dung 2 file nay de clean/tokenize tieng Viet.

## Step 5: spark_cleaning

Code:

- Airflow task trong [`dags/processing_dag.py`](../dags/processing_dag.py)
- Spark job: [`spark_jobs/cleaning_job.py`](../spark_jobs/cleaning_job.py)

Task `spark_cleaning` chay `spark-submit` job cleaning.

Input:

```text
HDFS_INPUT=hdfs://namenode:9000/user/root/raw_data
```

Output:

```text
HDFS_OUTPUT=hdfs://namenode:9000/user/root/staged/stg_posts_core
```

Job nay lam cac viec chinh:

- Doc raw crawler data tu HDFS.
- Chuan hoa schema bai viet/comment ve mot bang core.
- Lam sach text.
- Tao/giu cac truong can cho downstream: `post_id`, `source`, `title`, `body`, `segmented_text`, `parent_id`, `reaction_count`, `comment_count`, `view_count`, `created_at`, `crawled_at`.
- Dedup/noi chuan hoa du lieu.
- Ghi Parquet partition theo source vao `stg_posts_core`.

`stg_posts_core` la dataset trung tam. Sentiment, topic modeling, CMS, crisis detection deu phu thuoc vao output nay.

## Step 6: ingest_stg_posts_core

Code:

- Airflow helper `task_ingest_hdfs_dataset()` trong [`dags/processing_dag.py`](../dags/processing_dag.py)
- Ingest helper: [`scripts/hdfs_to_clickhouse.py`](../scripts/hdfs_to_clickhouse.py)

Task nay load HDFS Parquet vao ClickHouse table:

```text
tech_radar.stg_posts_core
```

Mode:

```text
full_refresh
```

Nghia la task se truncate table cu roi insert lai tu HDFS. Script ingest khong dung `SELECT *`, ma dung explicit column list va cast theo contract trong `TABLE_SPECS`.

ClickHouse doc HDFS bang pattern:

```sql
FROM hdfs('hdfs://namenode:9000/user/root/staged/stg_posts_core/*/*.parquet', 'Parquet')
```

Neu ClickHouse khong doc duoc HDFS hoac count = 0, task fail som de khong build dbt tren du lieu rong.

## Step 7A: sentiment_analysis

Code:

- Airflow task trong [`dags/processing_dag.py`](../dags/processing_dag.py)
- Spark job: [`spark_jobs/sentiment_job.py`](../spark_jobs/sentiment_job.py)

Nhanh sentiment chay sau khi `stg_posts_core` da duoc load vao ClickHouse.

Input:

```text
hdfs://namenode:9000/user/root/staged/stg_posts_core
```

Model:

```text
LOCAL_SENTIMENT_MODEL_PATH=/opt/airflow/models/...
```

Output:

```text
hdfs://namenode:9000/user/root/staged/stg_posts_nlp
```

Job nay lam cac viec chinh:

- Doc `stg_posts_core` Parquet.
- Lay text da clean/segmented.
- Chay PhoBERT sentiment inference.
- Tao output gom:
  - `post_id`
  - `sentiment_label`
  - `sentiment_score`
  - `model_version`
  - `predicted_at`
- Ghi Parquet ra `stg_posts_nlp`.

Trong flow hien tai, direct ClickHouse write cua sentiment duoc tat bang:

```text
WRITE_CLICKHOUSE_DIRECT=false
```

ClickHouse se chi load tu staged Parquet qua task ingest rieng.

## Step 8A: ingest_stg_posts_nlp

Code: [`scripts/hdfs_to_clickhouse.py`](../scripts/hdfs_to_clickhouse.py)

Load:

```text
hdfs://namenode:9000/user/root/staged/stg_posts_nlp
```

vao:

```text
tech_radar.stg_posts_nlp
```

Mode:

```text
replace_latest
```

Bang ClickHouse dung de dbt join sentiment vao bai viet trong `int_posts_enriched` va `int_topic_sentiment_hourly`.

## Step 7B: lda_topic_modeling

Code:

- Airflow task trong [`dags/processing_dag.py`](../dags/processing_dag.py)
- Spark job: [`spark_jobs/lda_job.py`](../spark_jobs/lda_job.py)

Nhanh topic modeling chay song song voi sentiment sau `ingest_stg_posts_core`.

Input:

```text
hdfs://namenode:9000/user/root/staged/stg_posts_core
```

Reference files:

```text
hdfs://namenode:9000/user/root/ref/stopwords_vi.txt
hdfs://namenode:9000/user/root/ref/slang_dict.json
```

Output chinh:

```text
hdfs://namenode:9000/user/root/staged/stg_post_topics
hdfs://namenode:9000/user/root/staged/stg_topics
```

Job nay lam cac viec chinh:

- Tokenize/normalize tieng Viet.
- Train/infer LDA topics.
- Tao topic assignment cho post:
  - `post_id`
  - `topic_id`
  - `topic_probability`
  - `model_type`
  - `predicted_at`
- Tao topic lookup:
  - `topic_id`
  - `label`
  - `top_keywords`
  - `coherence_score`
  - `model_version`
  - `created_at`

LDA khong can pretrained model. No can `stg_posts_core` va 2 reference files tren HDFS.

## Step 8B: ingest_stg_post_topics va ingest_stg_topics

Code: [`scripts/hdfs_to_clickhouse.py`](../scripts/hdfs_to_clickhouse.py)

Load topic assignment:

```text
hdfs://namenode:9000/user/root/staged/stg_post_topics
-> tech_radar.stg_post_topics
```

Load topic dimension:

```text
hdfs://namenode:9000/user/root/staged/stg_topics
-> tech_radar.stg_topics
```

Mode:

```text
replace_latest
```

Hai bang nay la dau vao chinh cho dbt build topic activity va dimension topic.

## Step 9: cms_keyword_counting

Code:

- Airflow function `task_run_cms_daily()` trong [`dags/processing_dag.py`](../dags/processing_dag.py)
- Algorithm Count-Min Sketch: [`algorithms/count_min_sketch.py`](../algorithms/count_min_sketch.py)

Task nay chay sau `ingest_stg_posts_nlp`.

Input:

```text
hdfs://namenode:9000/user/root/staged/stg_posts_core
```

Muc dich:

- Doc window data tu `stg_posts_core`.
- Tokenize keyword bang stopwords/slang dict.
- Cap nhat Count-Min Sketch.
- Lay top keywords theo source/window.
- Ghi staged Parquet cho keyword frequency.

Output:

```text
hdfs://namenode:9000/user/root/staged/stg_keyword_freq
```

Schema output dung cho ClickHouse:

```text
keyword
window_start
window_end
estimated_count
source
```

## Step 10: ingest_stg_keyword_freq

Code: [`scripts/hdfs_to_clickhouse.py`](../scripts/hdfs_to_clickhouse.py)

Load:

```text
hdfs://namenode:9000/user/root/staged/stg_keyword_freq
```

vao:

```text
tech_radar.stg_keyword_freq
```

Mode:

```text
append
```

Bang nay phuc vu phan tich keyword/trend theo time window. Trong dbt/dashboard hien tai, topic trend chinh van dua nhieu vao topic assignment va sentiment hourly.

## Step 11: crisis_to_hdfs_stg_crisis_events

Code:

- Airflow task trong [`dags/processing_dag.py`](../dags/processing_dag.py)
- Spark job: [`spark_jobs/crisis_detection.py`](../spark_jobs/crisis_detection.py)

Task crisis chi chay sau khi 3 nhanh da xong:

```text
ingest_stg_post_topics
ingest_stg_topics
ingest_stg_keyword_freq
```

Input chinh:

```text
hdfs://namenode:9000/user/root/staged/stg_posts_core
tech_radar.stg_posts_nlp
```

Target date:

```text
TARGET_DATE="{{ dag_run.conf.get('target_date', ds) }}"
```

Neu trigger manual co `dag_run.conf.target_date`, job se dung ngay do. Neu khong, job dung `ds` cua Airflow run.

Job crisis lam cac viec chinh:

- Doc comments trong `stg_posts_core` theo `TARGET_DATE`.
- Doc sentiment tu ClickHouse `stg_posts_nlp`.
- Tong hop feature theo gio:
  - `comment_count`
  - `unique_posts`
  - `unique_users`
  - cross-source ratio
  - negative sentiment ratio
  - velocity/acceleration
  - z-score neu co baseline
- Neu co model artifact thi dung Isolation Forest / classifier.
- Neu khong co model artifact, job fallback ve logic rule-based va khong crash.
- Build event rows neu co crisis hour.

Output neu phat hien event:

```text
hdfs://namenode:9000/user/root/staged/stg_crisis_events
```

Schema event:

```text
event_id
detected_at
severity
anomaly_score
trigger_conditions
affected_topics
neg_ratio
mention_velocity
evidence_post_ids
detected_date
```

Luu y quan trong: `stg_crisis_events` co the rong. Neu ngay test khong co comment nao theo `TARGET_DATE`, hoac khong co spike du dieu kien, job se pass nhung khong ghi event.

## Step 12: ingest_stg_crisis_events

Code: [`scripts/hdfs_to_clickhouse.py`](../scripts/hdfs_to_clickhouse.py)

Load:

```text
hdfs://namenode:9000/user/root/staged/stg_crisis_events
```

vao:

```text
tech_radar.stg_crisis_events
```

Mode:

```text
append
```

Dataset nay duoc danh dau `allow_empty=True` trong ingest helper. Neu HDFS chua co Parquet crisis event, task se skip insert va van thanh cong. Ly do: ngay test co the khong phat hien crisis, khong nen lam fail toan bo DAG.

## Step 13: dbt_transform

Code:

- Airflow task trong [`dags/processing_dag.py`](../dags/processing_dag.py)
- dbt project: [`warehouse/dbt_project`](../warehouse/dbt_project)

Task chay:

```bash
cd /opt/airflow/warehouse/dbt_project && dbt run --profiles-dir .
```

dbt doc 6 ClickHouse staging tables va build cac layer:

```text
staging      -> warehouse/dbt_project/models/staging
intermediate -> warehouse/dbt_project/models/intermediate
marts        -> warehouse/dbt_project/models/marts
```

Bang/relations quan trong:

```text
dbt_stg_posts_core
dbt_stg_posts_nlp
dbt_stg_post_topics
dbt_int_posts_enriched
dbt_int_topic_sentiment_hourly
dbt_fct_topic_activity
dbt_dim_topics
dbt_fct_crisis_events
```

Macro [`warehouse/dbt_project/macros/generate_alias_name.sql`](../warehouse/dbt_project/macros/generate_alias_name.sql) them prefix `dbt_` vao model name, vi vay model `fct_topic_activity` se thanh table/view `dbt_fct_topic_activity`.

### dbt intermediate

[`warehouse/dbt_project/models/intermediate/int_posts_enriched.sql`](../warehouse/dbt_project/models/intermediate/int_posts_enriched.sql):

- Join `stg_posts_core` voi `stg_posts_nlp`.
- Join topic assignment tu `stg_post_topics`.
- Tao bang enriched post/comment co sentiment va topic.

[`warehouse/dbt_project/models/intermediate/int_topic_sentiment_hourly.sql`](../warehouse/dbt_project/models/intermediate/int_topic_sentiment_hourly.sql):

- Tong hop hoat dong theo `topic_id` va time bucket.
- Tinh volume, sentiment count, negative ratio, trend score.

### dbt marts

[`warehouse/dbt_project/models/marts/fct_topic_activity.sql`](../warehouse/dbt_project/models/marts/fct_topic_activity.sql):

- Fact table cho trend/topic activity.
- Dashboard overview va trends doc bang nay.

[`warehouse/dbt_project/models/marts/dim_topics.sql`](../warehouse/dbt_project/models/marts/dim_topics.sql):

- Dimension topic, label, keyword, ranking metadata.

[`warehouse/dbt_project/models/marts/fct_crisis_events.sql`](../warehouse/dbt_project/models/marts/fct_crisis_events.sql):

- Mart cho crisis monitoring.
- Doc tu `tech_radar.stg_crisis_events`.
- Co the rong neu pipeline khong phat hien crisis event.

## Step 14: pipeline_end

Code: [`dags/processing_dag.py`](../dags/processing_dag.py)

Task `pipeline_end` la `DummyOperator`, danh dau DAG da hoan tat sau khi dbt build thanh cong.

## ClickHouse Ingest Contract

Ingest helper nam trong [`scripts/hdfs_to_clickhouse.py`](../scripts/hdfs_to_clickhouse.py).

File nay dinh nghia `TABLE_SPECS` cho dung 6 dataset:

```text
stg_posts_core
stg_posts_nlp
stg_post_topics
stg_topics
stg_keyword_freq
stg_crisis_events
```

Moi spec gom:

- dataset name
- ClickHouse table name
- ingest mode
- HDFS env path
- Parquet glob
- column list
- select/cast expression
- co cho phep empty hay khong

Mode hien tai:

```text
stg_posts_core      -> full_refresh
stg_posts_nlp       -> replace_latest
stg_post_topics     -> replace_latest
stg_topics          -> replace_latest
stg_keyword_freq    -> append
stg_crisis_events   -> append, allow_empty
```

Co the test rieng tung dataset bang:

```powershell
docker exec airflow-scheduler python /opt/airflow/scripts/hdfs_to_clickhouse.py stg_posts_core
docker exec airflow-scheduler python /opt/airflow/scripts/hdfs_to_clickhouse.py stg_posts_nlp
docker exec airflow-scheduler python /opt/airflow/scripts/hdfs_to_clickhouse.py stg_post_topics
docker exec airflow-scheduler python /opt/airflow/scripts/hdfs_to_clickhouse.py stg_topics
docker exec airflow-scheduler python /opt/airflow/scripts/hdfs_to_clickhouse.py stg_keyword_freq
docker exec airflow-scheduler python /opt/airflow/scripts/hdfs_to_clickhouse.py stg_crisis_events
```

## Dashboard Consumption

Dashboard NextJS doc ClickHouse qua:

- [`dashboard/src/lib/clickhouse.ts`](../dashboard/src/lib/clickhouse.ts)
- [`dashboard/src/lib/dal/radar.ts`](../dashboard/src/lib/dal/radar.ts)

Dashboard khong query HDFS. Dashboard chi query ClickHouse marts/dbt tables, vi du:

```text
tech_radar.dbt_fct_topic_activity
tech_radar.dbt_dim_topics
tech_radar.dbt_fct_crisis_events
tech_radar.dbt_int_posts_enriched
```

Health endpoint de test connection:

```text
http://localhost:3000/api/health/clickhouse
```

## Kiem Tra Sau Khi Chay DAG

Kiem tra HDFS staged Parquet:

```powershell
docker exec namenode hdfs dfs -find /user/root/staged/stg_posts_core -name "*.parquet"
docker exec namenode hdfs dfs -find /user/root/staged/stg_posts_nlp -name "*.parquet"
docker exec namenode hdfs dfs -find /user/root/staged/stg_post_topics -name "*.parquet"
docker exec namenode hdfs dfs -find /user/root/staged/stg_topics -name "*.parquet"
docker exec namenode hdfs dfs -find /user/root/staged/stg_keyword_freq -name "*.parquet"
docker exec namenode hdfs dfs -find /user/root/staged/stg_crisis_events -name "*.parquet"
```

Kiem tra ClickHouse staging:

```powershell
docker exec clickhouse clickhouse-client -u root --password root --query "SELECT 'stg_posts_core', count() FROM tech_radar.stg_posts_core UNION ALL SELECT 'stg_posts_nlp', count() FROM tech_radar.stg_posts_nlp UNION ALL SELECT 'stg_post_topics', count() FROM tech_radar.stg_post_topics UNION ALL SELECT 'stg_topics', count() FROM tech_radar.stg_topics UNION ALL SELECT 'stg_keyword_freq', count() FROM tech_radar.stg_keyword_freq UNION ALL SELECT 'stg_crisis_events', count() FROM tech_radar.stg_crisis_events"
```

Kiem tra dbt marts:

```powershell
docker exec clickhouse clickhouse-client -u root --password root --query "SELECT count() FROM tech_radar.dbt_fct_topic_activity"
docker exec clickhouse clickhouse-client -u root --password root --query "SELECT count() FROM tech_radar.dbt_dim_topics"
docker exec clickhouse clickhouse-client -u root --password root --query "SELECT count() FROM tech_radar.dbt_fct_crisis_events"
```

## Cac Diem Can Luu Y Khi Debug

Neu `validate_runtime_mounts` fail:

- Kiem tra file local co nam dung `crawlers/data`, `data`, `models`.
- Kiem tra Docker Compose da restart sau khi sua `.env`.

Neu `spark_cleaning` fail:

- Kiem tra raw data da upload vao `/user/root/raw_data`.
- Kiem tra schema raw CSV/JSON co dung format crawler cu.

Neu ingest ClickHouse fail:

- Kiem tra ClickHouse container co resolve duoc hostname `namenode`.
- Kiem tra HDFS folder co Parquet.
- Chay query `hdfs()` truc tiep trong ClickHouse de xem loi schema/path.

Neu `dbt_fct_topic_activity` rong:

- Thuong la do `stg_post_topics` khong join duoc voi posts trong window dbt dang lay.
- Can kiem tra `post_id` giua `stg_posts_core` va `stg_post_topics`, va filter thoi gian trong dbt intermediate.

Neu `stg_crisis_events` rong:

- Day khong nhat thiet la loi.
- Crisis job chi ghi event neu `TARGET_DATE` co comments va co spike du dieu kien.
- Neu Airflow run date lech voi ngay data raw, crisis se khong co event.

## DAG Lien Quan Ngoai Full Pipeline

Ngoai `full_processing_pipeline`, repo con co:

- [`dags/weekly_retrain_dag.py`](../dags/weekly_retrain_dag.py): retrain model cho crisis detection.
- `bertopic_weekly_inference` trong [`dags/processing_dag.py`](../dags/processing_dag.py): weekly BERTopic inference, doc `stg_posts_core` va ghi lai `stg_post_topics`, `stg_topics`.
- [`dags/daily_processing_dag.py`](../dags/daily_processing_dag.py): DAG crisis detection rieng theo huong tach ownership cu. Full pipeline hien tai da co crisis task rieng trong DAG chinh.

Trong test local end-to-end hien tai, DAG nen trigger la:

```text
full_processing_pipeline
```
