# TEAM ASSIGNMENTS — Vietnamese Tech Trend & Controversy Radar
## Phân công công việc chi tiết theo Phase & Thành viên

---

## Tổng quan phân công

| | Member 1 | Member 2 | Member 3 | Member 4 | Member 5 |
|---|---|---|---|---|---|
| **Role** | Data Engineer | DevOps / Infra | ML Engineer | NLP Engineer | Full-stack |
| **Layer** | Ingestion | Infrastructure + Big Data Core | Topic Modeling | Sentiment & Anomaly | Trend Scoring + Dashboard |
| **Primary** | Crawlers → HDFS + Pydantic | Ansible + Spark + HDFS + ClickHouse + dbt + LSH | LDA + BERTopic + Count-Min Sketch | PhoBERT + Isolation Forest | TrendScore + Streamlit |

---

## Chi tiết theo thành viên

---

### 👤 Member 1 — Data Engineer (Crawling & Direct HDFS Write)

**Trách nhiệm chính:** Thu thập toàn bộ raw data và write trực tiếp vào HDFS với schema validation.

#### Nhiệm vụ theo Phase

**Phase 1: Xây dựng crawler & raw dataset**

| Task | Mô tả | Output |
|---|---|---|
| 1.1 | Viết crawler VOZ (requests + BeautifulSoup4) | `crawlers/voz_crawler.py` |
| 1.2 | Viết crawler VnExpress Technology | `crawlers/vnexpress_crawler.py` |
| 1.3 | Viết crawler YouTube comments (YouTube Data API v3) | `crawlers/youtube_crawler.py` |
| 1.4 | Xử lý Cloudflare trên VOZ bằng `undetected-chromedriver` | `crawlers/voz_stealth.py` |
| 1.5 | **Viết Pydantic models** cho schema validation (VOZ, VnExpress, YouTube) | `schemas/models.py` |
| 1.6 | **Viết HDFS writer** với partition theo date `/data/raw/{source}/date={YYYY-MM-DD}/` | `utils/hdfs_writer.py` |
| 1.7 | Thu thập dataset thô tối thiểu: **100K posts** trên HDFS | HDFS `/data/raw/` |

**Phase 2: Ổn định pipeline & tăng volume**

| Task | Mô tả | Output |
|---|---|---|
| 2.1 | Tích hợp Airflow DAG cho crawl job định kỳ (2:00 AM daily) | `dags/crawl_dag.py` |
| 2.2 | Thêm rate-limiting, rotating proxy, retry logic | Cập nhật crawlers |
| 2.3 | **Viết crawl_logs** vào HDFS `/logs/crawl/date={YYYY-MM-DD}/` (JSONL format) | `crawlers/log_handler.py` |
| 2.4 | Test HDFS write performance: benchmark throughput | `benchmarks/hdfs_write.md` |
| 2.5 | Tăng dataset lên **500K posts** trên tất cả nguồn | Dataset trên HDFS |

**Phase 3: Duy trì & mở rộng**

| Task | Mô tả | Output |
|---|---|---|
| 3.1 | Giám sát crawler, sửa lỗi selector khi trang đổi HTML | Patch updates |
| 3.2 | Bổ sung metadata: timestamp, source_url, author_id | Schema update |
| 3.3 | Hỗ trợ Member 4 chuẩn bị labeled dataset cho fine-tuning PhoBERT | `data/labeled/` |

**Phase 4: Báo cáo & Hoàn thiện**

| Task | Mô tả | Output |
|---|---|---|
| 4.1 | Thống kê dataset cuối (số records, phân bố theo nguồn/thời gian) | `reports/dataset_stats.md` |
| 4.2 | Viết phần báo cáo: Data Ingestion & Collection Methodology | Báo cáo cuối kỳ |

**Deliverables:**
- [ ] 3 scripts crawler hoạt động ổn định
- [ ] Pydantic models cho schema validation
- [ ] HDFS writer với date partitioning
- [ ] HDFS `/data/raw/` với ≥ 500K raw records
- [ ] Airflow DAG crawl chạy daily

---

### 👤 Member 2 — DevOps / Infrastructure + Big Data Core 

**Trách nhiệm chính:** Toàn bộ nhóm phụ thuộc vào bạn. Infra phải sẵn sàng trước khi ai cũng làm việc được.

#### Nhiệm vụ theo Phase

**Phase 1: Dựng toàn bộ infrastructure**

| Task | Mô tả | Output |
|---|---|---|
| 1.1 | Viết Ansible inventory — map IP → role (master, worker x3, storage) | `ansible/inventory.ini` |
| 1.2 | Playbook `setup_java.yml` — cài Java JDK 11 tất cả nodes | `ansible/playbooks/setup_java.yml` |
| 1.3 | Playbook `setup_hdfs.yml` — cài Hadoop 3.3, cấu hình NameNode + DataNodes | `ansible/playbooks/setup_hdfs.yml` |
| 1.4 | Playbook `setup_spark.yml` — cài Spark 3.5, cấu hình standalone cluster | `ansible/playbooks/setup_spark.yml` |
| 1.5 | **Playbook `setup_clickhouse.yml`** — cài ClickHouse trên storage node | `ansible/playbooks/setup_clickhouse.yml` |
| 1.6 | **Playbook `setup_dbt.yml`** — cài dbt-core + dbt-clickhouse trên master | `ansible/playbooks/setup_dbt.yml` |
| 1.7 | Playbook `setup_airflow.yml` — cài Airflow 2.9 trên master node | `ansible/playbooks/setup_airflow.yml` |
| 1.8 | Playbook `setup_conda.yml` — tạo conda env `nlp-trend` trên tất cả nodes | `ansible/playbooks/setup_conda.yml` |
| 1.9 | Verify cluster: test Spark job `spark-submit examples/pi.py` | Cluster health check |
| 1.10 | Viết `CLUSTER_SETUP.md` — hướng dẫn cho cả nhóm | `docs/CLUSTER_SETUP.md` |

**Phase 2: Spark Pipeline + LSH/MinHash**

| Task | Mô tả | Output |
|---|---|---|
| 2.1 | Viết Spark job: đọc từ HDFS (Parquet) + data cleaning pipeline | `spark_jobs/cleaning_job.py` |
| 2.2 | **Cài ClickHouse JDBC driver** cho Spark | Config trong `spark_jobs/` |
| 2.3 | **Triển khai MinHashLSH** trên Spark để dedup bài đăng | `spark_jobs/dedup_lsh.py` |
| 2.4 | Tune Spark: executor memory, cores, shuffle partitions cho cluster | `spark_jobs/spark_config.py` |
| 2.5 | Viết Airflow DAG: `crawl` + `spark_cleaning` + `load_clickhouse` tasks | `dags/processing_dag.py` |
| 2.6 | Benchmark: đo execution time với 1 vs 3 workers (Strong Scaling Exp.1) | `benchmarks/scaling_exp1.md` |

**Phase 3: Integration & Optimization**

| Task | Mô tả | Output |
|---|---|---|
| 3.1 | **Setup dbt project** structure với dbt-clickhouse | `dbt_project/` |
| 3.2 | **Viết dbt staging models** cho ClickHouse tables | `dbt_project/models/staging/` |
| 3.3 | **Viết dbt mart models** (trend aggregations, crisis filtering) | `dbt_project/models/marts/` |
| 3.4 | Hỗ trợ Member 3 deploy BERTopic job trên Spark (`mapPartitions`) | Code review + fix |
| 3.5 | Hỗ trợ Member 4 cấu hình PhoBERT inference trên worker nodes | CUDA env check |
| 3.6 | Tối ưu HDFS: replication factor, block size cho dataset lớn | HDFS config update |
| 3.7 | Weak Scaling Experiment: 100K / 200K / 300K records trên 1/2/3 workers | `benchmarks/scaling_exp2.md` |

**Phase 4: Báo cáo hiệu năng**

| Task | Mô tả | Output |
|---|---|---|
| 4.1 | Chạy full benchmarking pipeline: Spark throughput, LSH accuracy, Speedup ratio | `benchmarks/final_report.md` |
| 4.2 | Viết phần báo cáo: Infrastructure, Distributed Computing, Performance Analysis | Báo cáo cuối kỳ |
| 4.3 | Playbook `deploy_app.yml` — deploy Streamlit dashboard lên master node | `ansible/playbooks/deploy_app.yml` |

**Deliverables:**
- [ ] 8 Ansible playbooks hoạt động trên toàn cluster (bao gồm ClickHouse + dbt)
- [ ] HDFS cluster (1 NameNode + 3 DataNodes) healthy
- [ ] Spark 3.5 standalone cluster chạy được multi-node job
- [ ] ClickHouse + dbt project với staging & mart models
- [ ] LSH dedup pipeline với Jaccard threshold ≥ 0.8
- [ ] Benchmarking report (Strong + Weak Scaling)
- [ ] Docker Compose cho môi trường dev

---

### 👤 Member 3 — ML Engineer (Topic Modeling & Keyword Streaming)

**Trách nhiệm chính:** Phân tích chủ đề từ corpus, đếm keyword streaming, và phân loại nội dung theo topic.

#### Nhiệm vụ theo Phase

**Phase 1: Research & Chuẩn bị**

| Task | Mô tả | Output |
|---|---|---|
| 1.1 | Nghiên cứu BERTopic, LDA, TF-IDF trên dữ liệu tiếng Việt | `research/topic_modeling_notes.md` |
| 1.2 | Chuẩn bị Vietnamese stopwords list (≥ 500 từ) | `data/stopwords_vi.txt` |
| 1.3 | Xây dựng custom slang dictionary (teencode → chuẩn, ≥ 2000 từ) | `data/slang_dict.json` |
| 1.4 | Prototype LDA nhỏ trên 5K mẫu bằng `gensim` (khám phá số topics k) | `notebooks/lda_prototype.ipynb` |

**Phase 2: Triển khai thuật toán**

| Task | Mô tả | Output |
|---|---|---|
| 2.1 | **Triển khai LDA** trên Spark MLlib (k=15–25, maxIter=50) | `spark_jobs/lda_job.py` |
| 2.2 | Đánh giá LDA: Topic Coherence Score (UMass / CV) | `notebooks/lda_evaluation.ipynb` |
| 2.3 | **Triển khai Count-Min Sketch** cho keyword frequency streaming | `algorithms/count_min_sketch.py` |
| 2.4 | Test Count-Min Sketch: so sánh với exact count, đo error rate | `tests/test_cms.py` |
| 2.5 | Tích hợp CMS vào Airflow DAG (chạy mỗi 15 phút) | Cập nhật `dags/processing_dag.py` |

**Phase 3: BERTopic & Integration**

| Task | Mô tả | Output |
|---|---|---|
| 3.1 | **Triển khai BERTopic** với PhoBERT embeddings (`vinai/phobert-base`) | `models/bertopic_model.py` |
| 3.2 | Tune BERTopic: `min_topic_size`, `nr_topics`, `umap_model` | `notebooks/bertopic_tuning.ipynb` |
| 3.3 | So sánh LDA vs BERTopic: coherence, phân bố topic, qualitative | `reports/topic_comparison.md` |
| 3.4 | Lưu topic clusters vào PostgreSQL (`topic_clusters` table) | `scripts/save_topics_to_pg.py` |
| 3.5 | Cung cấp `topic_id` và `topic_label` cho Member 5 dùng trong dashboard | Schema PostgreSQL |

**Phase 4: Hoàn thiện**

| Task | Mô tả | Output |
|---|---|---|
| 4.1 | Đánh giá cuối: Top topics mỗi tuần, so sánh giữa các nguồn | `reports/topic_analysis_final.md` |
| 4.2 | Viết phần báo cáo: Topic Modeling & Count-Min Sketch | Báo cáo cuối kỳ |

**Deliverables:**
- [ ] LDA pipeline chạy trên Spark MLlib
- [ ] BERTopic model với PhoBERT embeddings
- [ ] Count-Min Sketch implementation + unit tests
- [ ] Topic clusters trong ClickHouse
- [ ] So sánh LDA vs BERTopic (coherence scores)

---

### 👤 Member 4 — NLP Engineer (Sentiment + Crisis Detection)

**Trách nhiệm chính:** Fine-tune mô hình PhoBERT cho phân tích cảm xúc tiếng Việt và xây dựng hệ thống phát hiện khủng hoảng.

#### Nhiệm vụ theo Phase

**Phase 1: Preprocessing & Data Preparation**

| Task | Mô tả | Output |
|---|---|---|
| 1.1 | Cài và test VnCoreNLP trên cluster (yêu cầu Java 11) | `preprocessing/vncorenlp_test.py` |
| 1.2 | Viết text preprocessing pipeline hoàn chỉnh | `preprocessing/text_cleaner.py` |
| 1.3 | Gán nhãn dataset thủ công: ~2,000 mẫu (Negative/Neutral/Positive) | `data/labeled/sentiment_2k.csv` |
| 1.4 | Research: UIT-VSFC dataset + cách augment dữ liệu | `research/sentiment_data_notes.md` |

**Phase 2: Fine-tuning PhoBERT**

| Task | Mô tả | Output |
|---|---|---|
| 2.1 | **Fine-tune PhoBERT** (`vinai/phobert-base`) cho 3-class sentiment | `models/phobert_finetuned/` |
| 2.2 | Training config: lr=2e-5, batch_size=32, epochs=5, warmup_steps | `models/train_config.json` |
| 2.3 | Đánh giá model: Accuracy, Macro F1, Confusion Matrix | `reports/sentiment_eval.md` |
| 2.4 | Lưu model checkpoint để dùng trên Spark workers | `models/checkpoints/` |
| 2.5 | Viết inference wrapper: `predict_batch(texts, batch_size=32)` | `models/sentiment_predictor.py` |

**Phase 3: Phân tán + Crisis Detection**

| Task | Mô tả | Output |
|---|---|---|
| 3.1 | **Tích hợp PhoBERT vào Spark** qua `mapPartitions` | `spark_jobs/sentiment_job.py` |
| 3.2 | Test throughput: đo records/giây trên 1 vs 3 workers | `benchmarks/sentiment_throughput.md` |
| 3.3 | **Triển khai Isolation Forest** với features: mention_count, neg_ratio, velocity, engagement | `models/isolation_forest.py` |
| 3.4 | **Triển khai Rolling Mean threshold**: spike = rolling_mean + 2σ | `models/rolling_threshold.py` |
| 3.5 | Định nghĩa Crisis Alert: ≥2/3 điều kiện → ghi vào `crisis_alerts` PostgreSQL | `scripts/crisis_detector.py` |
| 3.6 | Tích hợp vào Airflow DAG: task `sentiment_job` + `crisis_detection` | Cập nhật DAG |

**Phase 4: Đánh giá**

| Task | Mô tả | Output |
|---|---|---|
| 4.1 | So sánh PhoBERT sentence-level vs BERTopic topic-level sentiment | `reports/nlp_comparison.md` |
| 4.2 | Đánh giá crisis detection: precision/recall của Isolation Forest | `reports/anomaly_eval.md` |
| 4.3 | Viết phần báo cáo: Sentiment Analysis & Crisis Detection | Báo cáo cuối kỳ |

**Deliverables:**
- [ ] PhoBERT fine-tuned (Accuracy ≥ 80%, Macro F1 ≥ 0.75)
- [ ] Spark sentiment job chạy phân tán (≥ 500 records/giây)
- [ ] Isolation Forest + Rolling Mean crisis detector
- [ ] `stg_crisis_alerts` table trong ClickHouse
- [ ] Preprocessing pipeline (VnCoreNLP + slang dict)

---

### 👤 Member 5 — Full-Stack (Trend Scoring + Dashboard)

**Trách nhiệm chính:** Tính toán điểm trend và xây dựng toàn bộ giao diện dashboard.

#### Nhiệm vụ theo Phase

**Phase 1: Setup ClickHouse Schema & Prototype Dashboard**

| Task | Mô tả | Output |
|---|---|---|
| 1.1 | Thiết kế schema toàn bộ ClickHouse (staging + mart tables) | `db/clickhouse_schema.sql` |
| 1.2 | Phối hợp Member 2 setup ClickHouse trên storage node | Config file |
| 1.3 | Prototype Streamlit dashboard với mock data | `dashboard/app_prototype.py` |
| 1.4 | Xác định màu sắc, layout, các trang dashboard (Overview, Trend, Crisis) | `dashboard/design_spec.md` |

**Phase 2: Trend Scoring Engine**

| Task | Mô tả | Output |
|---|---|---|
| 2.1 | Triển khai **Trend Score formula**: $\alpha \cdot V + \beta \cdot A + \gamma \cdot E + \delta \cdot I$ | `scoring/trend_scorer.py` |
| 2.2 | Tính **Mention Velocity** V(t): số mentions / giờ theo time window | `scoring/velocity.py` |
| 2.3 | Tính **Acceleration** A(t): đạo hàm bậc 1 của V theo thời gian | `scoring/acceleration.py` |
| 2.4 | Tính **Engagement Weight** E(t): likes + shares + comments với weight | `scoring/engagement.py` |
| 2.5 | Lưu `trend_scores` vào PostgreSQL theo ngày/tuần | `scripts/save_trends.py` |
| 2.6 | Tích hợp TrendScorer vào Airflow DAG | Cập nhật DAG |

**Phase 3: Dashboard hoàn thiện**

| Task | Mô tả | Output |
|---|---|---|
| 3.1 | Hoàn thiện **Streamlit Dashboard** — tất cả 3 trang | `dashboard/app.py` |
| 3.2 | Kết nối dashboard với ClickHouse (`clickhouse-connect`) | `dashboard/db_connector.py` |
| 3.3 | Tích hợp với dbt: dashboard đọc từ mart tables thay vì staging | Cập nhật queries |

**Phase 4: Hoàn thiện & Demo**

| Task | Mô tả | Output |
|---|---|---|
| 4.1 | Demo dashboard với dữ liệu thật từ pipeline | Live demo |
| 4.2 | Thêm `@st.cache_data` để tối ưu dashboard performance | Cập nhật `app.py` |
| 4.3 | Export top trending topics CSV / PDF cho báo cáo | `reports/trend_results.csv` |
| 4.4 | Viết phần báo cáo: Trend Scoring, Visualization | Báo cáo cuối kỳ |

**Deliverables:**
- [ ] Trend Score engine (3 components) hoạt động
- [ ] Streamlit Dashboard (3 trang) kết nối ClickHouse
- [ ] `stg_trend_scores` + dbt marts trong ClickHouse

---

## Timeline tổng thể (4 Phases)

```
Phase │ M1 (Crawl)        │ M2 (Infra)         │ M3 (Topic)        │ M4 (Sentiment)     │ M5 (Dashboard)
─────┼───────────────────┼────────────────────┼───────────────────┼────────────────────┼────────────────
     │ VOZ crawler       │ Ansible inventory  │ Research LDA      │ Research PhoBERT   │ DB schema SQL
     │ VnExpress crawler │ Java + Hadoop setup│ Stopwords dict    │ VnCoreNLP setup    │ Streamlit proto
  1  │ YouTube API       │ Spark cluster      │ Slang dict        │ Label 1K data      │ Layout design
     │ MongoDB setup     │ Airflow + verify   │ LDA prototype     │ Label 1K data      │ Mock dashboard
─────┼───────────────────┼────────────────────┼───────────────────┼────────────────────┼────────────────
     │ HDFS write setup │ Airflow + verify   │ LDA prototype     │ Label 1K data      │ Mock dashboard
─────├───────────────────├────────────────────├───────────────────├────────────────────├────────────────
     │ Crawl → HDFS    │ Spark cleaning job │ LDA on Spark      │ PhoBERT fine-tune  │ Trend velocity
     │ Proxy + retry     │ ClickHouse setup   │ LDA evaluation    │ Training run       │ Trend accel.
  2  │ Pydantic validate │ LSH/MinHash        │ Count-Min Sketch  │ Model evaluation   │ Engagement score
     │ 500K dataset      │ Strong Scaling exp │ CMS integration   │ Inference wrapper  │ Save to ClickHouse
─────┼───────────────────┼────────────────────┼───────────────────┼────────────────────┼────────────────
     │ Giám sát crawler  │ Docker Compose     │ BERTopic setup    │ Spark sentiment    │ Dashboard pages
     │ Metadata update   │ dbt project setup  │ BERTopic tuning   │ Throughput test    │ Connect CH
  3  │ Help M4 labeling  │ dbt models (marts) │ LDA vs BERTopic   │ Isolation Forest   │ dbt integration
     │ Fix selectors     │ Weak Scaling exp   │ Save to ClickHouse│ Crisis detector    │ Polish UI
─────┼───────────────────┼────────────────────┼───────────────────┼────────────────────┼────────────────
     │ Dataset stats     │ Full benchmark     │ Final evaluation  │ Eval crisis detect │ Dashboard demo
  4  │ Write report      │ Write report       │ Write report      │ Write report       │ Export results
     │ Review + submit   │ Review + submit    │ Review + submit   │ Review + submit    │ Write report
```

---

## Dependencies giữa các Member

### Sơ đồ tổng quan

```
                    ┌──────────────────────────────────────────────┐
                    │   M2 (Infra) — CRITICAL PATH                 │
                    │   Tất cả đều phụ thuộc vào M2                │
                    │   Phải xong Spark + HDFS trước cuối Phase 1  │
                    └──────┬───────────┬───────────────┬───────────┘
                           │           │               │
              ┌────────────▼──┐   ┌────▼──────┐  ┌─────▼───────────┐
              │  M3 (Topic)   │   │ M4 (NLP)  │  │ M5 (Dashboard)  │
              └───────┬───────┘   └─────┬─────┘  └─────────────────┘
                      │                 │
              topic_id│                 │sentiment_scores
              topic_lb│                 │crisis_alerts
                      └────────────────►│
                                        │
                                   ┌────▼─────┐
                                   │ClickHouse│
                                   │+ dbt     │
                                   │Dashboard │
                                   └──────────┘

        M1 (Crawler) ──────────────────────────────────────────►
        Cung cấp raw data trên HDFS cho M3, M4 dùng (corpus + labeled set)
```

---

### Bảng Hard Blockers — Không làm được nếu chưa có

> **Hard blocker** = bắt buộc phải chờ, không có cách workaround.

| Người bị block | Chờ ai | Chờ gì cụ thể | Deadline cần có | Nếu trễ thì làm gì |
|---|---|---|---|---|
| M1 | **M2** | HDFS NameNode + DataNode chạy được | Cuối Phase 1 | M1 tạm export ra file local, sync lên HDFS sau |
| M3 | **M2** | Spark cluster multi-node chạy job được | Cuối Phase 1 | M3 chạy LDA bằng `gensim` local, port sang Spark sau |
| M3 | **M1** | ≥ 50K raw text posts trong HDFS | Cuối Phase 1 | M3 dùng dataset mẫu từ GitHub (VLSP, UIT-VSFC) |
| M4 | **M2** | Java 11 cài xong trên tất cả nodes | Giữa Phase 1 | M4 dùng `underthesea` (thuần Python) thay VnCoreNLP tạm |
| M4 | **M2** | Spark cluster để chạy `sentiment_job.py` | Cuối Phase 1 | M4 fine-tune PhoBERT local, chưa cần Spark ngay |
| M4 | **M1** | ≥ 2K bài đăng để gán nhãn sentiment | Cuối Phase 1 | M4 dùng UIT-VSFC public dataset gán sẵn |
| M5 | **M2** | ClickHouse trên storage node + dbt setup | Cuối Phase 1 | M5 dùng ClickHouse local (Docker), sync schema sau |
| M5 | **M3** | `topic_id`, `topic_label` trong ClickHouse | Đầu Phase 3 | M5 dùng mock topic data, thay bằng real data sau |
| M5 | **M4** | `sentiment_scores`, `crisis_alerts` trong ClickHouse | Giữa Phase 3 | M5 build dashboard với mock data trước |


---

### Bảng Soft Dependencies — Nên có nhưng không bắt buộc dừng lại

| Ai | Cần từ ai | Mô tả | Workaround |
|---|---|---|---|
| M3 | M4 | Preprocessing pipeline (text_cleaner.py) để dùng chung | M3 tự viết preprocessing tạm, merge sau |
| M5 | M3 | Coherence score để quyết định số topics k hiển thị trên dashboard | M5 hardcode k=20 tạm |
| M2 | M4 | Biết trước model size của PhoBERT để cấu hình worker RAM đúng | M2 để `executor.memory=8g` mặc định |

---

### Giao thức Handoff (Bàn giao giữa các thành viên)

Mỗi khi một thành viên hoàn thành phần mình để người khác dùng, **phải thực hiện đủ 3 bước**:

```
Bước 1: Tạo PR vào branch develop, tag người nhận vào để review
Bước 2: Ghi vào file HANDOFF_LOG.md:
         - Tên artifact (file, table, model checkpoint)
         - Đường dẫn chính xác (HDFS path / PostgreSQL table / folder)
         - Schema / format (ví dụ: CSV với columns gì, HDFS Parquet với schema gì)
         - Ví dụ đầu vào/đầu ra
Bước 3: Ping người nhận trên nhóm chat, confirm họ đã chạy test được
```

**Các mốc handoff cụ thể:**

| Phase | Từ | Đến | Artifact | Đường dẫn |
|---|---|---|---|---|
| Giửa Phase 1 | M2 | M4 | Java 11 ready trên cluster | — (verify bằng `java -version` trên nodes) |
| Cuối Phase 1 | M2 | M1, M3, M4, M5 | Spark + HDFS + ClickHouse sẵn sàng | `hdfs://master:9000/data/` |
| Cuối Phase 1 | M1 | M3, M4 | 100K raw posts trên HDFS | `hdfs://master:9000/data/raw/` |
| Cuối Phase 2 | M4 | — | PhoBERT checkpoint fine-tuned | `hdfs://master:9000/models/phobert-sentiment/` |
| Cuối Phase 2 | M2 | M5 | dbt project ready | `dbt_project/` với staging & mart models |
| Đầu Phase 3 | M3 | M5 | Topic clusters trong ClickHouse | `nlp_db.stg_topic_clusters` |
| Giữa Phase 3 | M4 | M5 | Sentiment scores + crisis alerts | `nlp_db.stg_processed_posts`, `nlp_db.stg_crisis_alerts` |



---

## Shared Responsibilities (Tất cả thành viên)

| Trách nhiệm chung | Mô tả |
|---|---|
| **Code review** | Mỗi PR cần ít nhất 1 người review trước khi merge vào `develop` |
| **Unit tests** | Mỗi module phải có file `tests/test_*.py` cơ bản |
| **Git workflow** | Làm việc trên `feature/<tên>` branch, không push thẳng vào `main` |
| **Documentation** | Mỗi function phải có docstring, mỗi module có README ngắn |
| **Meeting hàng tuần** | Báo cáo tiến độ, blockers, điều chỉnh kế hoạch |
| **Báo cáo cuối kỳ** | Mỗi người viết section của mình (xem Phase 4 tasks) |

---

## Cấu trúc thư mục dự án (Target)

```
Distributed-NLP-Trend-Analysis/
├── ansible/
│   ├── inventory.ini
│   └── playbooks/
│       ├── setup_java.yml
│       ├── setup_hdfs.yml
│       ├── setup_spark.yml
│       ├── setup_airflow.yml
│       ├── setup_conda.yml
│       └── deploy_app.yml
├── crawlers/
│   ├── voz_crawler.py
│   ├── voz_stealth.py
│   ├── vnexpress_crawler.py
│   ├── youtube_crawler.py
│   └── log_handler.py
├── spark_jobs/
│   ├── cleaning_job.py
│   ├── dedup_lsh.py
│   ├── lda_job.py
│   └── sentiment_job.py
├── preprocessing/
│   ├── text_cleaner.py
│   └── vncorenlp_test.py
├── algorithms/
│   └── count_min_sketch.py
├── models/
│   ├── phobert_finetuned/
│   ├── bertopic_model.py
│   ├── sentiment_predictor.py
│   ├── isolation_forest.py
│   └── rolling_threshold.py
├── scoring/
│   ├── trend_scorer.py
│   ├── velocity.py
│   ├── acceleration.py
│   └── engagement.py
├── graph/
│   └── build_graph.py
├── dags/
│   ├── crawl_dag.py
│   └── processing_dag.py
├── dbt_project/
│   ├── dbt_project.yml
│   ├── models/
│   │   ├── staging/
│   │   │   ├── stg_processed_posts.sql
│   │   │   ├── stg_sentiment_scores.sql
│   │   │   └── stg_topic_clusters.sql
│   │   └── marts/
│   │       ├── mart_trend_scores.sql
│   │       ├── mart_crisis_alerts.sql
│   │       └── mart_topic_summary.sql
│   └── tests/
├── dashboard/
│   ├── app.py
│   └── db_connector.py
├── db/
│   └── clickhouse_schema.sql
├── scripts/
│   ├── save_topics_to_ch.py
│   ├── save_trends_ch.py
│   └── crisis_detector.py
├── data/
│   ├── stopwords_vi.txt
│   ├── slang_dict.json
│   └── labeled/
├── benchmarks/
│   ├── scaling_exp1.md
│   ├── scaling_exp2.md
│   └── sentiment_throughput.md
├── tests/
│   └── test_cms.py
├── reports/
├── docs/
│   └── CLUSTER_SETUP.md
├── docker-compose.yml
├── requirements.txt
├── README.md
└── TECH_STACK.md
```

---

## Checklist Phase Gate

### ✅ Phase 1 Done khi:
- [ ] Cluster chạy được Spark job multi-node (M2)
- [ ] ≥ 100K raw records trong HDFS (M1)
- [ ] ClickHouse + dbt setup xong (M2)
- [ ] VnCoreNLP chạy được trên mọi node (M4)
- [ ] ClickHouse schema khởi tạo xong (M5)

### ✅ Phase 2 Done khi:
- [ ] LSH dedup pipeline ra kết quả, Jaccard ≥ 0.8 (M2)
- [ ] dbt models (staging + marts) chạy được (M2)
- [ ] LDA cho ra 15–25 topics có nghĩa (M3)
- [ ] PhoBERT F1 ≥ 0.75 trên test set (M4)
- [ ] TrendScore tính được cho mọi topic (M5)

### ✅ Phase 3 Done khi:
- [ ] Full pipeline chạy end-to-end: crawl → clean → NLP → score → ClickHouse → dbt → dashboard (tất cả)
- [ ] BERTopic tốt hơn hoặc ngang LDA về coherence (M3)
- [ ] Crisis alerts hoạt động (M4)
- [ ] Dashboard 3 trang kết nối live ClickHouse + đọc từ dbt marts (M5)

### ✅ Phase 4 Done khi:
- [ ] Benchmarking report hoàn chỉnh (M2)
- [ ] Báo cáo cuối kỳ: mỗi người đã viết section của mình
- [ ] Demo dashboard với dữ liệu thật
