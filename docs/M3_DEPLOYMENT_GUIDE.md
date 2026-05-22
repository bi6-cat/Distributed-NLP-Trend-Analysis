# Member 3 — Hướng dẫn Deploy & Vận hành Topic Modeling

**Role:** ML Engineer — Topic Modeling (LDA + BERTopic + Count-Min Sketch)  
**Ngày cập nhật:** 2026-05-21

---

## Tổng quan kiến trúc M3

M3 phụ trách **2 pipeline topic modeling song song** với chu kỳ và stack khác nhau:

```
══════════════════════════════════════════════════════════════════
 LDA (hàng ngày)                BERTopic (hàng tuần)
══════════════════════════════════════════════════════════════════

 HDFS staged/ (Parquet)          stg_posts_core.csv
        │                               │
        ▼                               ▼
  lda_job.py                  bertopic_inference_job.py
  (Spark MLlib)                (PyTorch / PhoBERT)
        │                               │
        ▼                               ▼
  HDFS results/lda/           output/bertopic_inference/
  (Parquet)                   ├── post_topic_assignment.parquet
        │                     └── topics.parquet
        ▼                               │
  dbt_transform                         ▼
  (dbt-clickhouse)            save_topics_to_ch.py
        │                               │
        └──────────────┬────────────────┘
                       ▼
              ClickHouse: tech_radar
              ├── stg_post_topics   ← cả LDA lẫn BERTopic
              └── stg_topics
```

| | LDA | BERTopic |
|---|---|---|
| **Chu kỳ** | Hàng ngày 02:00 AM | Chủ nhật 03:00 AM |
| **DAG** | `daily_processing_pipeline` | `bertopic_weekly_inference` |
| **Stack** | Spark MLlib (distributed) | PyTorch + HDBSCAN (single-node) |
| **Input** | HDFS staged Parquet | `stg_posts_core.csv` (local) |
| **Output → ClickHouse** | Qua **dbt** transform | Qua `save_topics_to_ch.py` |
| **model_type trong CH** | `lda` | `bertopic` |

---

## Cấu trúc file liên quan M3

```
├── models/
│   └── bertopic_model.py                   # VietnameseBERTopicModel wrapper
├── spark_jobs/
│   ├── lda_job.py                          # LDA via Spark MLlib (chạy daily)
│   └── bertopic_inference_job.py           # BERTopic inference (chạy weekly)
├── scripts/
│   ├── save_topics_to_ch.py                # Push BERTopic parquet → ClickHouse
│   └── crisis_detector.py                  # Phát hiện crisis events
├── algorithms/
│   └── count_min_sketch.py                 # CMS streaming (Task 2.3)
├── notebooks/
│   ├── bertopic_tuning.ipynb               # Hyperparameter tuning (Kaggle)
│   └── train_bertopic_2xT4_kaggle.ipynb    # Training notebook (Kaggle 2xT4)
├── dags/
│   └── processing_dag.py                   # 3 DAGs: daily, cms, bertopic_weekly
├── warehouse/
│   ├── clickhouse/init_schema.sql          # DDL tạo bảng ClickHouse
│   └── dbt_project/                        # dbt models (load LDA → ClickHouse)
└── output/
    ├── task2.2_lda_evaluation/
    │   └── lda_topics_final.csv            # LDA evaluation kết quả
    └── task3.1_bertopic/output/bertopic_model/   # BERTopic model đã train
        ├── bertopic_model                  # BERTopic pickle
        ├── config.pkl                      # Hyperparameters
        └── topics.pkl                      # Training results
```

---

## Kết nối hạ tầng

| Service | Host (trong Docker) | Host (ngoài Docker) | Credentials |
|---|---|---|---|
| **ClickHouse** | `clickhouse:8123` | `localhost:8123` | user=`admin` / pass=`clickhouse_secret` |
| **Airflow UI** | — | `localhost:8081` | `admin` / `admin` |
| **HDFS NameNode** | `namenode:9000` | `localhost:9870` (Web UI) | — |
| **Spark Master** | `spark-master:7077` | `localhost:8080` (Web UI) | — |

> ClickHouse playground: http://localhost:8123/play  
> Airflow Web UI: http://localhost:8081

---

## Lần đầu deploy (từ đầu)

### Bước 1 — Build Docker

```bash
# Từ thư mục gốc project
docker-compose build --no-cache airflow-webserver airflow-scheduler airflow-init
docker-compose up -d

# Kiểm tra status — chờ airflow-webserver hiện "healthy"
docker-compose ps
```

### Bước 2 — Khởi tạo ClickHouse schema

Init script tự chạy khi container khởi lần đầu (volume trống).  
Nếu container đã có volume cũ, chạy manual:

```bash
docker exec clickhouse clickhouse-client \
  -u admin --password clickhouse_secret \
  --multiquery < warehouse/clickhouse/init_schema.sql
```

Kiểm tra:
```bash
docker exec clickhouse clickhouse-client \
  -u admin --password clickhouse_secret \
  --query "SHOW TABLES FROM tech_radar"
# Kỳ vọng: stg_crisis_events, stg_keyword_freq, stg_post_topics,
#           stg_posts_core, stg_posts_nlp, stg_topics
```

### Bước 3 — Enable DAGs trong Airflow UI

Mở **http://localhost:8081** → đăng nhập `admin / admin` → bật toggle 3 DAGs:

| DAG | Schedule | Mô tả |
|---|---|---|
| `daily_processing_pipeline` | `0 2 * * *` — 02:00 AM hàng ngày | crawl → Spark → **LDA** + sentiment → dbt |
| `bertopic_weekly_inference` | `0 3 * * 0` — Chủ nhật 03:00 AM | **BERTopic** inference + push ClickHouse |
| `cms_keyword_streaming` | `*/15 * * * *` — mỗi 15 phút | Count-Min Sketch keyword frequency |

> Tất cả DAGs **bắt đầu ở trạng thái paused** — phải bật thủ công lần đầu.

---

## Pipeline 1 — LDA (Spark MLlib, hàng ngày)

### Cách hoạt động

```
HDFS staged/ (Parquet)
    │
    ▼  Spark UDF  — preprocess_udf()
    ① Lowercase  ② Remove HTML/URL  ③ Slang normalize
    ④ underthesea word_tokenize  ⑤ Stopword removal (1,942 từ)
    │
    ▼  Spark MLlib
CountVectorizer (vocab=4000, minDF=3)
    → IDF
    → LDA (k=10, maxIter=60, optimizer="em")
    │
    ▼
HDFS /data/results/lda/
  ├── lda_model/               ← Spark MLlib model
  ├── topics.parquet           ← (topic_id, term, weight)
  └── post_topic_assignment/   ← (post_id, topic_id, probability)
    │
    ▼  dbt_transform (trong daily_processing_pipeline DAG)
tech_radar.stg_post_topics  (model_type = 'lda')
tech_radar.stg_topics
```

LDA **không push trực tiếp** vào ClickHouse. Spark ghi ra HDFS Parquet, sau đó `dbt_transform` task đọc và load vào ClickHouse (`warehouse/dbt_project/`).

### Tham số đã tuned

| Tham số | Giá trị | Lý do |
|---|---|---|
| `k` | **10** | Coherence tốt nhất — xem `output/task2.2_lda_evaluation/lda_topics_final.csv` |
| `maxIter` | 60 | Hội tụ ổn định, không overfit |
| `optimizer` | `em` | Batch EM ổn định hơn `online` cho corpus cố định |
| `vocab_size` | 4000 | Giảm nhiễu từ hiếm |
| `min_df` | 3 | Giữ đủ từ khóa quan trọng |

### Chạy LDA thủ công (test)

```bash
# Local mode — đọc CSV, ghi output/lda/
docker exec spark-master spark-submit \
  /opt/airflow/spark_jobs/lda_job.py \
  --local \
  --input-path /opt/airflow/data/preprocessed/ \
  --output-path /opt/airflow/output/lda/ \
  --k 10 --max-iter 60

# Cluster mode — đọc/ghi HDFS
docker exec spark-master spark-submit \
  --master spark://spark-master:7077 \
  /opt/airflow/spark_jobs/lda_job.py \
  --input-path hdfs://namenode:9000/user/zett/staged/ \
  --output-path hdfs://namenode:9000/user/zett/results/lda/ \
  --k 10 --max-iter 60
```

Trigger DAG từ terminal:
```bash
docker exec airflow-scheduler airflow dags trigger daily_processing_pipeline
```

### Kiểm tra kết quả LDA

```sql
-- Topics LDA
SELECT topic_id, label, top_keywords
FROM tech_radar.stg_topics
WHERE model_version LIKE 'lda%'
ORDER BY topic_id;

-- Phân bố bài viết
SELECT topic_id, COUNT(*) AS n_posts, AVG(topic_probability) AS avg_prob
FROM tech_radar.stg_post_topics
WHERE model_type = 'lda'
GROUP BY topic_id
ORDER BY n_posts DESC;
```

---

## Pipeline 2 — BERTopic (PyTorch + PhoBERT, hàng tuần)

### Cách hoạt động

BERTopic và LDA đọc từ **cùng một nguồn**: HDFS Parquet output của `cleaning_job.py` (M2).  
BERTopic không chạy trên Spark nên dùng **WebHDFS REST API** để đọc/ghi thay vì Hadoop CLI.

```
HDFS /user/zett/staged/stg_posts_core/   ← cùng nguồn với LDA
    │  (WebHDFS HTTP, không cần hdfs CLI)
    ▼
bertopic_inference_job.py
  load VietnameseBERTopicModel từ output/task3.1_bertopic/output/bertopic_model/
    │
    ▼  model.topic_model.transform(documents)
  PhoBERT (vinai/phobert-base) → UMAP (5-dim) → HDBSCAN → c-TF-IDF
    │
    ▼
HDFS /user/zett/results/bertopic/
  ├── post_topic_assignment.parquet
  └── topics.parquet
    │
    ▼
save_topics_to_ch.py
    │
    ▼
tech_radar.stg_post_topics  (model_type = 'bertopic')
tech_radar.stg_topics
```

**Tại sao BERTopic không dùng Spark như LDA?**  
BERTopic dùng PyTorch + HDBSCAN — không thể chạy trên Spark executor (không serializable).  
BERTopic chạy single-node trên Airflow driver, đọc HDFS qua WebHDFS HTTP (port 9870) thay vì HDFS CLI.

### Thông tin model v1

| Thành phần | Giá trị |
|---|---|
| Embedding | `vinai/phobert-base` |
| UMAP n_neighbors | 15 |
| UMAP n_components | 5 |
| UMAP min_dist | 0.0 |
| HDBSCAN min_cluster_size | 15 |
| HDBSCAN min_samples | 10 |
| Số topics (v1) | ~214 topics |
| Trained on | ~162K posts (VOZ, VnExpress, YouTube) |
| Model path | `output/task3.1_bertopic/output/bertopic_model/` |
| Model version tag | `bertopic_v1` |

> Dùng `sklearn.cluster.HDBSCAN` — **không** dùng package `hdbscan` cũ (xung đột numpy 2.x).

### Chạy BERTopic inference thủ công

**Cách 1 — Airflow UI (khuyến nghị):**
1. Vào http://localhost:8081 → DAG `bertopic_weekly_inference` → **▶ Trigger DAG**
2. Theo dõi logs từng task trong Graph View

**Cách 2 — Terminal (trigger qua Airflow CLI):**
```bash
docker exec airflow-scheduler \
  airflow dags trigger bertopic_weekly_inference

# Xem log realtime
docker-compose logs -f airflow-scheduler 2>&1 | grep -i bertopic
```

**Cách 3 — Chạy thẳng Python trong container (debug):**
```bash
# Chạy trong airflow-scheduler container (có network tới HDFS/ClickHouse)
docker exec airflow-scheduler bash -c "
  cd /opt/airflow &&
  python spark_jobs/bertopic_inference_job.py \
    --model-path output/task3.1_bertopic/output/bertopic_model
"

# Sau đó push lên ClickHouse
docker exec airflow-scheduler \
  python /opt/airflow/scripts/save_topics_to_ch.py \
  --host clickhouse
```

**Cách 4 — Local mode (chỉ dùng khi dev ngoài Docker, không cần HDFS):**
```bash
# Đọc từ CSV local thay vì HDFS — dùng flag --local
python spark_jobs/bertopic_inference_job.py --local
python scripts/save_topics_to_ch.py --host localhost
```

### Re-train BERTopic (khi có data mới)

Cần GPU — dùng Kaggle:

1. Upload `notebooks/train_bertopic_2xT4_kaggle.ipynb` lên Kaggle
2. Attach dataset `data/preprocessed/stg_posts_core.csv`
3. Chạy với accelerator **2x T4 GPU**
4. Download folder `output/task3.1_bertopic/output/bertopic_model/`
5. Đặt vào đúng đường dẫn trên server → DAG tự load khi chạy tiếp (không cần rebuild Docker)

Hyperparameter tuning: `notebooks/bertopic_tuning.ipynb` (sample ngẫu nhiên 8K docs, ~8-10 phút với UMAP caching).

### Kiểm tra kết quả BERTopic

```sql
-- Topics BERTopic
SELECT topic_id, label, top_keywords, coherence_score
FROM tech_radar.stg_topics
WHERE model_version LIKE 'bertopic%'
ORDER BY topic_id
LIMIT 20;

-- Phân bố bài viết
SELECT topic_id, COUNT(*) AS n_posts
FROM tech_radar.stg_post_topics
WHERE model_type = 'bertopic'
GROUP BY topic_id
ORDER BY n_posts DESC
LIMIT 15;
```

---

## So sánh LDA vs BERTopic trong ClickHouse

```sql
-- Tổng quan 2 model
SELECT model_type,
       COUNT(*)                    AS n_posts,
       COUNT(DISTINCT topic_id)    AS n_topics,
       AVG(topic_probability)      AS avg_confidence
FROM tech_radar.stg_post_topics
GROUP BY model_type;

-- Topic quality so sánh
SELECT model_version, AVG(coherence_score) AS avg_coherence
FROM tech_radar.stg_topics
WHERE coherence_score IS NOT NULL
GROUP BY model_version;
```

Chạy query từ terminal:
```bash
docker exec clickhouse clickhouse-client \
  -u admin --password clickhouse_secret \
  --query "SELECT model_type, COUNT(*) n, COUNT(DISTINCT topic_id) topics FROM tech_radar.stg_post_topics GROUP BY model_type"
```

---

## DAG Flow chi tiết

### `daily_processing_pipeline` (02:00 AM hàng ngày)

```
pipeline_start
     │
     ▼
crawl_sources              ← upload data lên HDFS
     │
     ▼
spark_cleaning             ← Spark: clean + dedup LSH
     │
    ┌┴──────────────────────┐
    ▼                       ▼
lda_topic_modeling    sentiment_analysis    ← song song
    └──────────────────────┬┘
                           ▼
                     dbt_transform          ← load LDA + sentiment → ClickHouse
                           │
                           ▼
                     pipeline_end
```

### `bertopic_weekly_inference` (Chủ nhật 03:00 AM)

```
bertopic_start
     │
     ▼
check_model_path           ← kiểm tra model path tồn tại
     │
     ▼
run_inference_pipeline     ← load model → transform → ghi parquet (timeout 3h)
     │
     ▼
save_topics_to_clickhouse  ← đọc parquet → insert stg_post_topics + stg_topics
     │
     ▼
bertopic_end
```

### `cms_keyword_streaming` (mỗi 15 phút)

```
start → read_new_data → run_cms_update → export_top_keywords → end
```
Kết quả ghi vào `tech_radar.stg_keyword_freq`.

---

## Xử lý sự cố thường gặp

### LDA fail (task `lda_topic_modeling`)

```bash
# Kiểm tra Spark cluster
docker-compose ps spark-master spark-worker
# Xem logs
docker-compose logs spark-master | tail -50

# Thử chạy local để isolate lỗi
docker exec spark-master spark-submit \
  /opt/airflow/spark_jobs/lda_job.py --local \
  --input-path /opt/airflow/data/preprocessed/ \
  --output-path /opt/airflow/output/lda_test/ --k 5 --max-iter 10
```

### BERTopic fail (task `run_inference_pipeline`)

```bash
# Kiểm tra model tồn tại
ls output/task3.1_bertopic/output/bertopic_model/
# Phải có: bertopic_model  config.pkl  topics.pkl

# Kiểm tra data input
wc -l data/preprocessed/stg_posts_core.csv

# Xem log chi tiết
docker-compose logs airflow-scheduler | grep -A 30 "run_inference_pipeline"
```

### BERTopic fail (task `save_topics_to_clickhouse`)

```bash
# Kiểm tra parquet đã tạo chưa
ls output/bertopic_inference/
# Phải có: post_topic_assignment.parquet  topics.parquet

# Test kết nối ClickHouse
docker exec clickhouse clickhouse-client \
  -u admin --password clickhouse_secret --query "SELECT 1"

# Chạy lại thủ công bên trong container
docker exec airflow-scheduler \
  python /opt/airflow/scripts/save_topics_to_ch.py \
  --input-dir /opt/airflow/output/bertopic_inference \
  --host clickhouse
```

### BERTopic import lỗi (package thiếu)

```bash
docker exec airflow-webserver pip show bertopic sentence-transformers umap-learn

# Nếu thiếu → rebuild image (đã có trong requirements.txt)
docker-compose build --no-cache airflow-webserver airflow-scheduler
docker-compose up -d airflow-webserver airflow-scheduler
```

### ClickHouse table chưa có

```bash
docker exec clickhouse clickhouse-client \
  -u admin --password clickhouse_secret \
  --multiquery < warehouse/clickhouse/init_schema.sql
```

---

## Lịch chạy tóm tắt

| DAG | Cron | Lần kế tiếp | Nhiệm vụ M3 |
|---|---|---|---|
| `daily_processing_pipeline` | `0 2 * * *` | Hàng ngày 02:00 AM | LDA topic modeling |
| `bertopic_weekly_inference` | `0 3 * * 0` | Chủ nhật 2026-05-24 03:00 AM | BERTopic inference |
| `cms_keyword_streaming` | `*/15 * * * *` | Mỗi 15 phút | Keyword frequency |

---

## Liên kết tham khảo nội bộ

- [docs/CLUSTER_INFO.md](CLUSTER_INFO.md) — Thông tin infrastructure Docker
- [docs/TEAM_ASSIGNMENTS.md](TEAM_ASSIGNMENTS.md) — Phân công toàn team
- [docs/data_flow_schema_evolution.md](data_flow_schema_evolution.md) — Schema ClickHouse chi tiết
- [warehouse/clickhouse/init_schema.sql](../warehouse/clickhouse/init_schema.sql) — DDL tạo bảng
- [spark_jobs/lda_job.py](../spark_jobs/lda_job.py) — LDA Spark job
- [spark_jobs/bertopic_inference_job.py](../spark_jobs/bertopic_inference_job.py) — BERTopic inference
- [scripts/save_topics_to_ch.py](../scripts/save_topics_to_ch.py) — Push BERTopic → ClickHouse
