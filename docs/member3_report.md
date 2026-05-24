# BÁO CÁO KỸ THUẬT — MEMBER 3: TOPIC MODELING & KEYWORD ANALYTICS

**Dự án:** Vietnamese Tech Trend & Controversy Radar (DNLPA)
**Thành viên:** Member 3 — ML Engineer
**Ngày:** 22/05/2026

---

## 1. TỔNG QUAN NHIỆM VỤ

Member 3 chịu trách nhiệm toàn bộ tầng **Topic Modeling & Keyword Analytics** trong pipeline Big Data, bao gồm:
- Xây dựng 2 mô hình topic modeling: **LDA** (daily) và **BERTopic** (weekly)
- Triển khai thuật toán **Count-Min Sketch** đếm keyword theo thời gian thực
- Tích hợp kết quả vào **ClickHouse** và orchestrate bằng **Apache Airflow**

---

## 2. CÁC THÀNH PHẦN ĐÃ TRIỂN KHAI

### 2.1 LDA Topic Modeling (`spark_jobs/lda_job.py`)

**Kiến trúc:**
- Đọc dữ liệu đầu vào từ `stg_posts_core` (parquet Hive-partitioned, output của cleaning_job Member 2)
- Pipeline tiền xử lý tiếng Việt: lowercase → bỏ HTML/URL/emoji → chuẩn hóa teencode (~5,000 mapping) → tách từ bằng underthesea → bỏ stopwords (1,942 từ)
- **TF-IDF:** CountVectorizer (vocab=4,000) + IDF
- **LDA:** Spark MLlib, k=20 topics, maxIter=60, optimizer="em" (Blei et al., 2003)
- Output: 20 topics → `stg_topics` + 13,000+ post assignments → `stg_post_topics`

**Tính năng đặc biệt:**
- **k-sweep evaluation:** tự động thử k=10..25, tính coherence C_V và U_Mass (via gensim) để chọn k tối ưu
- **Local/Cluster mode:** tự động detect môi trường, đọc parquet directory (Hive-partitioned `source=xxx/`) hoặc CSV
- **Schema mapping:** tự động map `segmented_text` / `clean_text` / `body` tùy cột có sẵn

**Tham khảo:**
> Blei, D.M., Ng, A.Y., Jordan, M.I. (2003). *Latent Dirichlet Allocation*. JMLR 3, 993-1022.

---

### 2.2 BERTopic Inference (`spark_jobs/bertopic_inference_job.py`)

**Kiến trúc:**
- Chạy **weekly** (Chủ nhật 3h sáng) — lấy 7 ngày data tích lũy
- Pipeline: PhoBERT embeddings → UMAP dimensionality reduction → HDBSCAN clustering → c-TF-IDF topic representation
- **top_n_topics=50:** chỉ giữ 50 topic có nhiều bài nhất, loại bỏ noise topic (-1)
- Output: 50 BERTopic topics → cùng bảng `stg_topics` + `stg_post_topics` với `model_type='bertopic'`

**Thiết kế:**
- Toàn bộ inference chạy trong 1 `PythonOperator` duy nhất để tránh serialize model qua XCom
- Không dùng SparkSubmitOperator vì BERTopic là PyTorch + HDBSCAN, không chạy được trên Spark executor
- Cùng nguồn data với LDA: `stg_posts_core` parquet

---

### 2.3 Count-Min Sketch Keyword Tracking (`algorithms/count_min_sketch.py`)

**Thuật toán (Cormode & Muthukrishnan, 2005 — CS246 Stanford):**
- Cấu trúc: ma trận `depth=5 × width=4096`, MurmurHash3
- Bộ nhớ cố định: **~80KB** bất kể số lượng keyword → phù hợp streaming 24/7
- Đảm bảo: với xác suất ≥ 96.9%, ước lượng count không vượt quá `N/width` so với giá trị thật

**Tính năng đặc biệt — Per-source tracking:**
- Mỗi nguồn (voz, vnexpress, vatvo) có **CMS riêng biệt** → `top_k` trả về count chính xác theo từng source
- Trước khi fix: dùng chung 1 CMS → count của từ "máy" giống hệt nhau (10,615) ở mọi nguồn → **bug đã fix**
- Output: Top-K keywords mỗi ngày theo từng source → `stg_keyword_freq` (600 rows/run)

**Tham khảo:**
> Cormode, G., Muthukrishnan, S. (2005). *An Improved Data Stream Summary: The Count-Min Sketch and its Applications*. Journal of Algorithms 55(1), 58-75.

---

### 2.4 Airflow DAG Orchestration (`dags/processing_dag.py`)

**2 DAG được triển khai:**

| DAG | Schedule | Tasks |
|-----|----------|-------|
| `daily_processing_pipeline` | Hàng ngày 2h sáng | crawl → spark_cleaning → LDA + CMS + sentiment (song song) → save to CH → dbt |
| `bertopic_weekly_inference` | Chủ nhật 3h sáng | check_model → run_inference → save to CH |

**Thiết kế quan trọng:**
- LDA, CMS, sentiment chạy **song song** (fan-out sau spark_cleaning) → giảm latency
- `trigger_rule="all_done"` cho dbt_transform → pipeline không bị block nếu sentiment (Member 4) fail
- `USE_LOCAL=True`: tự động switch giữa local filesystem (Docker dev) và HDFS (production cluster)

---

## 3. CÁC LỖI ĐÃ PHÁT HIỆN VÀ SỬA

| # | Lỗi | Root Cause | Fix |
|---|-----|-----------|-----|
| 1 | `Permission denied` khi Spark ghi parquet | uid 50000 tạo dir với 755, Spark worker uid 1001 không write được | Thêm `spark.hadoop.fs.permissions.umask-mode=000` vào Spark conf |
| 2 | LDA `No CSV files found` | Cleaning_job output là Hive-partitioned parquet dir (`source=xxx/`), LDA chỉ tìm CSV | Viết `_has_parquet()` detect Hive dir, đọc bằng `pd.read_parquet()` |
| 3 | CMS `TypeError: Cannot setitem on a Categorical` | Spark partition column `source` đọc lên là Pandas `Categorical`, `fillna("unknown")` fail | `.astype(str)` trước `.fillna("unknown")` |
| 4 | `dbt_transform` stuck ở `upstream_failed` | `sentiment_analysis` fail → default trigger_rule block dbt | `trigger_rule="all_done"` |
| 5 | dbt compile error `arguments` keyword | dbt 1.7+ bỏ wrapper `arguments:` trong `accepted_values` | Remove `arguments:` nesting trong schema.yml |
| 6 | dbt `Connection refused localhost:8123` | `profiles.yml` dùng `host: localhost` nhưng ClickHouse ở container `clickhouse` | Đổi `host: clickhouse` |
| 7 | CMS trả về count giống nhau mọi source | Dùng 1 CMS chung cho tất cả source, `top_k` trả về global count | Per-source CMS dict: `source_cms[src].add(token)` |
| 8 | Chrome crash trong Docker (`SessionNotCreatedException`) | Thiếu `--headless`, `--no-sandbox`, `--disable-dev-shm-usage` | Thêm 3 flags vào cả 3 crawler (vnexpress, voz, vatvo) |

---

## 4. KẾT QUẢ TRONG CLICKHOUSE

| Bảng | Rows | Nội dung |
|------|------|---------|
| `stg_topics` | 70 | 20 LDA topics (`lda_k20`) + 50 BERTopic topics (`bertopic_v1`) |
| `stg_post_topics` | 13,348 | Post → topic assignments của cả 2 model |
| `stg_keyword_freq` | 600 | Top keyword theo từng source (voz, vnexpress, vatvo) |

**Ví dụ topics LDA phát hiện được:**
- Topic 10: `pro | iphone | và` → ["pro","iphone","thế","với","có","là","mới","hơn","max","màu","apple","bán"]
- Topic 5: `sạc | và | thiết` → ["sạc","và","thiết","âm","với","phẩm","sản","bị","lượng","pin","huawei"]

**Ví dụ topics BERTopic phát hiện được:**
- `hệ_trung_quốc_huawei` → ["hệ","trung_quốc","huawei","wick","mỹ","apple","ôi","chộ","smartphone","ai"]
- `đánh_đánh_giá_giá` → ["đánh","đánh_giá","giá","nghiệm","review","nghiệp","trải","sử","dụng","nhanh"]

---

## 5. KIẾN TRÚC DỮ LIỆU

```
[Crawlers] ──raw CSV──► [Cleaning Job/Spark] ──parquet──► stg_posts_core
                                                                │
                    ┌───────────────────────────────────────────┤
                    ▼                   ▼                       ▼
              [LDA/Spark]         [BERTopic]              [Count-Min Sketch]
                    │                   │                       │
                    ▼                   ▼                       ▼
             stg_post_topics      stg_topics             stg_keyword_freq
                    │                   │
                    └────────┬──────────┘
                             ▼
                     [dbt transforms]
                             │
                    fct_topic_activity (gold)
                    dim_topics (gold)
```

---

## 6. CÔNG NGHỆ SỬ DỤNG

| Công nghệ | Phiên bản | Mục đích |
|-----------|-----------|---------|
| Apache Spark MLlib | 3.5+ | LDA training phân tán |
| BERTopic | latest | Neural topic modeling |
| PhoBERT | vinai/phobert-base | Vietnamese embeddings |
| underthesea | latest | Vietnamese word segmentation |
| Count-Min Sketch | custom (CS246) | Streaming keyword counting |
| Apache Airflow | 2.x | Pipeline orchestration |
| ClickHouse | 24.3 | Columnar analytics storage |
| dbt-core | 1.7+ | ELT transformations |
| Selenium + Chrome | headless | Web crawling |

---

## 7. SCHEMA TUÂN THỦ THIẾT KẾ

Tất cả bảng do Member 3 load đều **khớp 100%** với `docs/data_flow_schema_evolution.md`:

- `stg_post_topics`: `ReplacingMergeTree(loaded_at)` — reruns tự động overwrite kết quả cũ
- `stg_topics`: `ReplacingMergeTree(created_at)` ORDER BY `(topic_id, model_version)` — hỗ trợ đồng thời LDA + BERTopic
- `stg_keyword_freq`: `MergeTree()` partitioned by month — hỗ trợ query time-range hiệu quả

```sql
-- stg_post_topics
CREATE TABLE tech_radar.stg_post_topics (
    post_id           String,
    topic_id          Int32,
    topic_probability Float32,
    model_type        LowCardinality(String),
    predicted_at      DateTime,
    loaded_at         DateTime DEFAULT now()
) ENGINE = ReplacingMergeTree(loaded_at)
ORDER BY (post_id);

-- stg_topics
CREATE TABLE tech_radar.stg_topics (
    topic_id        Int32,
    label           String,
    top_keywords    Array(String),
    coherence_score Nullable(Float32),
    model_version   String,
    created_at      DateTime DEFAULT now()
) ENGINE = ReplacingMergeTree(created_at)
ORDER BY (topic_id, model_version);

-- stg_keyword_freq
CREATE TABLE tech_radar.stg_keyword_freq (
    keyword         String,
    window_start    DateTime,
    window_end      DateTime,
    estimated_count Int64,
    source          LowCardinality(String)
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(window_start)
ORDER BY (keyword, window_start);
```
