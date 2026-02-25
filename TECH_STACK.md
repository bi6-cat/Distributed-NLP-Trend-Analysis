# Vietnamese Tech Trend & Controversy Radar
## Tech Stack & System Architecture — Final Reference Document

> **Phiên bản:** v2.0 — Cập nhật: Tháng 2/2026  
> **Trạng thái:** Phase 1 — Thiết kế & Thu thập dữ liệu  

---

## Mục lục

1. [Kiến trúc tổng quan](#1-kiến-trúc-tổng-quan)
2. [Data Ingestion Layer](#2-data-ingestion-layer)
3. [Storage Layer](#3-storage-layer)
4. [Big Data Processing Layer](#4-big-data-processing-layer)
5. [Các thuật toán CS246 cốt lõi](#5-các-thuật-toán-cs246-cốt-lõi)
6. [NLP & Machine Learning Layer](#6-nlp--machine-learning-layer)
7. [Trend & Crisis Scoring Logic](#7-trend--crisis-scoring-logic)
8. [Visualization Layer](#8-visualization-layer)
9. [Orchestration & DevOps Layer](#9-orchestration--devops-layer)
10. [Infrastructure & HPC Cluster](#10-infrastructure--hpc-cluster)
11. [Yêu cầu phần cứng & phụ thuộc hệ thống](#11-yêu-cầu-phần-cứng--phụ-thuộc-hệ-thống)
12. [Performance Benchmarking Plan](#12-performance-benchmarking-plan)
13. [Phân công nhóm](#13-phân-công-nhóm)
14. [Dependency Summary](#14-dependency-summary)

---

## 1. Kiến trúc tổng quan

Hệ thống theo kiến trúc **7 lớp phân tách rõ ràng**, vận hành trên cụm **HPC Semi-Lab** và quản lý tự động bằng **Ansible**.

```
┌─────────────────────────────────────────────────────────────────┐
│                  ORCHESTRATION & DEVOPS LAYER                   │
│             Ansible · Airflow · Git · Docker · Conda            │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                    DATA INGESTION LAYER                         │
│         Requests/BS4 · Selenium · YouTube Data API v3           │
│              Nguồn: VOZ · VnExpress Tech · YouTube              │
└────────────────────────────┬────────────────────────────────────┘
                             │ Raw JSON
┌────────────────────────────▼────────────────────────────────────┐
│                       STORAGE LAYER                             │
│    MongoDB (Data Lake)  ──────────  HDFS (Raw/Intermediate)     │
│         PostgreSQL (Data Warehouse / Dashboard queries)         │
└──────────────┬──────────────────────────────┬───────────────────┘
               │ Spark + Mongo Connector      │ Parquet / ORC
┌──────────────▼──────────────────────────────▼───────────────────┐
│               BIG DATA PROCESSING LAYER                         │
│                   Apache Spark 3.5+ (PySpark)                   │
│    Dedup (LSH/MinHash) · PageRank/HITS · Count-Min Sketch       │
└────────────────────────────┬────────────────────────────────────┘
                             │ Clean DataFrame
┌────────────────────────────▼────────────────────────────────────┐
│              NLP & MACHINE LEARNING LAYER                       │
│  VnCoreNLP · LDA · BERTopic · PhoBERT · Isolation Forest        │
└────────────────────────────┬────────────────────────────────────┘
                             │ Scores & Labels
┌────────────────────────────▼────────────────────────────────────┐
│            TREND & CRISIS DETECTION LAYER                       │
│      Trend Score Engine · Rolling Mean · Anomaly Alerts         │
└────────────────────────────┬────────────────────────────────────┘
                             │ Aggregated Results → PostgreSQL
┌────────────────────────────▼────────────────────────────────────┐
│                   VISUALIZATION LAYER                           │
│                 Streamlit · Plotly · Matplotlib                 │
└─────────────────────────────────────────────────────────────────┘
```

**Luồng dữ liệu chính:**
```
Crawlers → MongoDB (raw) → HDFS (staged) → Spark (clean + compute)
         → PhoBERT/BERTopic (NLP) → Trend/Crisis Score → PostgreSQL → Streamlit
```

---

## 2. Data Ingestion Layer

| Thành phần | Công cụ | Mục đích |
|---|---|---|
| Static crawling | `requests` + `BeautifulSoup4` | VOZ forums, VnExpress articles |
| Dynamic crawling | `Selenium` + `undetected-chromedriver` | Trang render bằng JS, anti-bot |
| API crawling | `YouTube Data API v3` | Comments & metadata video công nghệ |
| Rate limiting | `time.sleep` + `rotating proxies` | Tránh bị block IP |
| Scheduler | Apache Airflow DAG | Tự động hóa lịch crawl định kỳ |

**Nguồn dữ liệu:**

| Nguồn | Loại | Ước lượng volume |
|---|---|---|
| VOZ (voz.vn) | Forum posts, comments | ~50K–200K posts/tháng |
| VnExpress Technology | News articles, comments | ~5K–20K bài/tháng |
| YouTube (tech channels) | Video comments | ~100K–500K comments/tháng |

**Lưu ý kỹ thuật:**
- Selenium cần `chromedriver` tương thích với phiên bản Chrome trên cluster.
- VOZ có Cloudflare protection — cần `undetected-chromedriver` hoặc `cloudscraper`.
- YouTube API có quota giới hạn **10,000 units/ngày** — thiết kế request hợp lý.

---

## 3. Storage Layer

### 3.1 MongoDB — Data Lake

```yaml
Vai trò: Lưu trữ dữ liệu thô dạng JSON, không cần schema cố định
Version: MongoDB 7.x
Collections:
  - raw_voz_posts      # Bài đăng & comment VOZ
  - raw_vnexpress      # Bài báo VnExpress
  - raw_youtube        # Comment YouTube
  - crawl_logs         # Log trạng thái crawl
Index: created_at (TTL), source, topic_keyword
```

### 3.2 HDFS — Raw & Intermediate Storage

```yaml
Vai trò: Lưu trữ trung gian cho Spark, tối ưu cho đọc phân tán
Format: JSON (ingress) → Parquet (sau cleaning)
Thư mục:
  /data/raw/           # Dữ liệu export từ MongoDB
  /data/staged/        # Sau khi clean bằng Spark
  /data/features/      # Feature vectors, embeddings
  /data/results/       # Kết quả scoring
```

> **Tại sao cần HDFS?** Spark đọc dữ liệu từ HDFS nhanh hơn MongoDB nhiều lần (tránh overhead network + BSON serialization). Dữ liệu được export từ MongoDB sang HDFS dạng Parquet trước khi Spark xử lý.

### 3.3 PostgreSQL — Data Warehouse

```yaml
Vai trò: Lưu kết quả đã xử lý, phục vụ dashboard query
Version: PostgreSQL 16.x
Tables:
  - processed_posts     # Bài đăng đã clean + label sentiment
  - trend_scores        # Điểm trend theo ngày/tuần
  - crisis_alerts       # Cảnh báo bất thường
  - topic_clusters      # Kết quả BERTopic/LDA
  - influencer_scores   # PageRank/HITS scores
Index: (source, date), (topic_id), (trend_score DESC)
```

### 3.4 Kết nối Spark ↔ MongoDB

```python
# Yêu cầu: mongo-spark-connector (phiên bản khớp Spark 3.5)
# spark-submit --packages org.mongodb.spark:mongo-spark-connector_2.12:10.3.0

df = spark.read.format("mongodb") \
    .option("uri", "mongodb://localhost:27017/nlp_db.raw_voz_posts") \
    .load()
```

---

## 4. Big Data Processing Layer

### 4.1 Apache Spark 3.5+ (PySpark)

| Task | Spark Module | Mô tả |
|---|---|---|
| Data cleaning | `Spark SQL / DataFrame API` | Loại bỏ null, normalize text, dedup |
| Feature computation | `MLlib` | TF-IDF, feature vectors |
| Distributed NLP | `mapPartitions` | Chạy PhoBERT song song trên từng partition |
| Graph computation | `GraphX` (via Python wrapper) | PageRank / HITS trên mạng thảo luận |
| Batch processing | `Spark Structured Batch` | Xử lý theo ngày/tuần |

### 4.2 Cấu hình Spark trên HPC Cluster

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("VietnameseTrendAnalysis") \
    .master("spark://master-node:7077") \
    .config("spark.executor.memory", "8g") \
    .config("spark.executor.cores", "4") \
    .config("spark.num.executors", "4") \
    .config("spark.sql.shuffle.partitions", "200") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()
```

### 4.3 Thư viện hỗ trợ

| Thư viện | Phiên bản | Vai trò |
|---|---|---|
| `pandas` | 2.x | Local processing, debug |
| `numpy` | 1.26+ | Tính toán số học |
| `pyarrow` | 14.x | Đọc/ghi Parquet |
| `mongo-spark-connector` | 10.3.x | Kết nối MongoDB ↔ Spark |

---

## 5. Các thuật toán CS246 cốt lõi


### 5.1 Shingling + MinHashing + LSH (Near-Duplicate Detection)

**Mục tiêu:** Loại bỏ bài đăng trùng lặp/gần trùng trong hàng triệu posts.

```
Bước 1: k-Shingling
  Input: Chuỗi văn bản đã tokenize
  k = 5 (word-level shingles)
  Output: Tập hợp các shingle (dạng set)

Bước 2: MinHashing
  Số hash functions: 100–200
  Output: Signature matrix (100-200 chiều)
  Triển khai: pyspark.ml.feature.MinHashLSH

Bước 3: LSH (Locality Sensitive Hashing)
  Số bands: 20, rows per band: 5 → threshold Jaccard ~0.5
  Output: Danh sách cặp (pair) nghi ngờ trùng lặp

Bước 4: Xác minh Jaccard similarity
  Lọc các cặp có Jaccard ≥ 0.8 → đánh dấu duplicate
```

```python
from pyspark.ml.feature import MinHashLSH, HashingTF, Tokenizer

tokenizer = Tokenizer(inputCol="clean_text", outputCol="words")
hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=1000)
mh = MinHashLSH(inputCol="rawFeatures", outputCol="hashes", numHashTables=5)

# Join để tìm near-duplicates
similar_pairs = model.approxSimilarityJoin(
    featurized, featurized, 0.8, distCol="jaccardDist"
)
```

### 5.2 PageRank / HITS (Influence Scoring)

**Mục tiêu:** Xác định người dùng/nguồn có authority trong mạng thảo luận.

```
Xây dựng đồ thị:
  Node: User ID hoặc Source URL
  Edge: User A reply/quote User B → A → B (với weight = số lần)

PageRank:
  Damping factor: d = 0.85
  Convergence threshold: 1e-6
  Triển khai: graphframes.GraphFrame.pageRank()

HITS (Hub & Authority):
  Authority(v) = Σ Hub(u) cho mọi u → v
  Hub(u) = Σ Authority(v) cho mọi v mà u → v
  Iterations: 30
```

```python
from graphframes import GraphFrame

g = GraphFrame(vertices_df, edges_df)
results = g.pageRank(resetProbability=0.15, tol=1e-6)
influencers = results.vertices.orderBy("pagerank", ascending=False)
```

> **Yêu cầu:** `graphframes` JAR phải được thêm vào `spark-submit --packages`.

### 5.3 Count-Min Sketch (Streaming Keyword Frequency)

**Mục tiêu:** Đếm tần suất từ khóa theo cửa sổ thời gian mà không lưu toàn bộ dữ liệu vào RAM.

```
Cấu trúc: Ma trận d × w (depth × width)
  d = 5 (số hash function), w = 2048 (độ rộng)
  Sai số ε = e/w, xác suất δ = (1/2)^d

Tri n khai như Spark Structured Streaming:
  Input: Kafka topic (hoặc file stream từ crawler output)
  Window: 1 giờ sliding, 15 phút slide interval
  Output: Top-K keywords mỗi window
```

```python
# Triển khai Count-Min Sketch thuần Python tích hợp với PySpark UDF
import mmh3  # MurmurHash3

class CountMinSketch:
    def __init__(self, d=5, w=2048):
        self.d = d
        self.w = w
        self.table = [[0] * w for _ in range(d)]

    def add(self, item):
        for i in range(self.d):
            col = mmh3.hash(item, seed=i) % self.w
            self.table[i][col] += 1

    def query(self, item):
        return min(
            self.table[i][mmh3.hash(item, seed=i) % self.w]
            for i in range(self.d)
        )
```

---

## 6. NLP & Machine Learning Layer

### 6.1 Text Preprocessing Pipeline

```
Raw Text
  ↓ [1] Lowercase + remove HTML tags, URLs, emojis
  ↓ [2] VnCoreNLP word segmentation (ví dụ: "học sinh" → "học_sinh")
  ↓ [3] Regex: chuẩn hóa số, ký tự đặc biệt
  ↓ [4] Custom slang dictionary (teencode → chuẩn)
  ↓ [5] Stopword removal (Vietnamese stopwords list)
  ↓ Clean Text
```

| Công cụ | Vai trò | Ghi chú |
|---|---|---|
| `VnCoreNLP` | Word segmentation, POS tagging | **Yêu cầu Java 8+** trên mọi worker node |
| `underthesea` | Thay thế nhẹ hơn VnCoreNLP | Thuần Python, dễ cài hơn |
| `regex` | Chuẩn hóa văn bản | Built-in |
| Custom dict | ~5,000 từ teen/slang | File JSON tự xây dựng |

### 6.2 Topic Modeling

#### LDA (Baseline)

```python
from pyspark.ml.clustering import LDA

lda = LDA(k=20, maxIter=50, featuresCol="tfidf_features",
          optimizer="em", subsamplingRate=0.05)
model = lda.fit(tfidf_df)
topics = model.describeTopics(maxTermsPerTopic=10)
```

| Tham số | Giá trị | Lý do |
|---|---|---|
| `k` (số topic) | 15–25 | Thử nghiệm với coherence score |
| `maxIter` | 50 | Cân bằng tốc độ / chất lượng |
| `optimizer` | `em` | Tốt hơn `online` cho batch |

#### BERTopic (Transformer-based)

```python
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

# Dùng PhoBERT làm embedding backbone
embedding_model = SentenceTransformer("vinai/phobert-base")
topic_model = BERTopic(
    embedding_model=embedding_model,
    language="multilingual",
    calculate_probabilities=True,
    nr_topics="auto",
    min_topic_size=30
)
topics, probs = topic_model.fit_transform(docs)
```

> **Lưu ý GPU:** BERTopic + PhoBERT cần ít nhất **8GB VRAM**. Chạy trên GPU node của HPC hoặc dùng `batch_size=16` để giảm tải.

### 6.3 Sentiment Analysis

**Model:** `vinai/phobert-base` fine-tuned với bộ dữ liệu tiếng Việt (UIT-VSFC hoặc tự gán nhãn).

| Class | Label | Ý nghĩa |
|---|---|---|
| 0 | Negative | Tiêu cực / phản đối |
| 1 | Neutral | Trung lập |
| 2 | Positive | Tích cực / ủng hộ |

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
model = AutoModelForSequenceClassification.from_pretrained(
    "./phobert-sentiment-finetuned", num_labels=3
)

# Chạy song song trên Spark worker thông qua mapPartitions
def predict_partition(iterator):
    model.eval()
    for batch in batched(iterator, 32):
        inputs = tokenizer(batch, return_tensors="pt",
                           truncation=True, padding=True, max_length=256)
        with torch.no_grad():
            logits = model(**inputs).logits
        preds = torch.argmax(logits, dim=-1).tolist()
        yield from preds
```

**Metrics đánh giá:**

| Metric | Mục tiêu |
|---|---|
| Accuracy | ≥ 80% |
| Macro F1-score | ≥ 0.75 |
| Confusion Matrix | Phân tích nhầm lẫn Negative ↔ Neutral |

### 6.4 Crisis Detection (Anomaly Detection)

```python
from sklearn.ensemble import IsolationForest
import numpy as np

# Features: [mention_count, neg_sentiment_ratio, velocity, engagement]
X = feature_df[["mention_count", "neg_ratio", "velocity", "engagement"]].values

clf = IsolationForest(
    n_estimators=100,
    contamination=0.05,   # Giả định 5% là bất thường
    random_state=42
)
clf.fit(X)
anomaly_labels = clf.predict(X)  # -1: bất thường, 1: bình thường

# Rolling Mean Threshold (bổ sung)
rolling_mean = pd.Series(neg_counts).rolling(window=24).mean()
spike_threshold = rolling_mean + 2 * rolling_std
```

---

## 7. Trend & Crisis Scoring Logic

### 7.1 Trend Score Formula

$$\text{TrendScore}(t) = \alpha \cdot V(t) + \beta \cdot A(t) + \gamma \cdot E(t) + \delta \cdot I(t)$$

| Ký hiệu | Ý nghĩa | Trọng số |
|---|---|---|
| $V(t)$ | Mention Velocity (số đề cập / giờ) | $\alpha = 0.35$ |
| $A(t)$ | Acceleration (tốc độ tăng của V) | $\beta = 0.25$ |
| $E(t)$ | Engagement Weight (like + share + comment) | $\gamma = 0.25$ |
| $I(t)$ | Influencer Boost (PageRank score của tác giả) | $\delta = 0.15$ |

> Trọng số trên là giá trị khởi đầu — điều chỉnh qua thực nghiệm.

### 7.2 Crisis Alert Conditions

Cảnh báo được kích hoạt khi **ít nhất 2 trong 3 điều kiện** sau xảy ra đồng thời:

1. Tỷ lệ sentiment tiêu cực tăng đột biến ≥ **30% so với baseline** trong vòng 2 giờ.
2. Volume đề cập vượt ngưỡng **2σ** so với rolling mean 7 ngày.
3. Isolation Forest gán nhãn **anomaly** cho điểm thời gian đó.

---

## 8. Visualization Layer

### 8.1 Streamlit Dashboard

```
Trang chủ (Overview):
  - Top 10 trending topics (bar chart, cập nhật theo ngày)
  - Sentiment distribution pie chart toàn hệ thống
  - Crisis alert banner (nếu có)

Trang Trend Detail:
  - Time-series chart trend score theo topic
  - Word cloud từ khóa nổi bật
  - Danh sách bài đăng tiêu biểu

Trang Influencer:
  - Bảng xếp hạng PageRank / Authority score
  - Network graph mạng thảo luận (subset)

Trang Crisis Monitor:
  - Timeline các sự kiện bất thường
  - Anomaly score chart (Isolation Forest output)
```

### 8.2 Thư viện Visualization

| Thư viện | Vai trò |
|---|---|
| `streamlit` 1.32+ | Web app framework |
| `plotly` | Interactive charts (time-series, bar, pie) |
| `matplotlib` | Static charts cho báo cáo |
| `wordcloud` | Word cloud từ khóa |
| `networkx` | Vẽ network graph (influencer network) |

---

## 9. Orchestration & DevOps Layer

### 9.1 Apache Airflow (Pipeline Scheduling)

```
Airflow DAG: daily_pipeline
  Schedule: 0 2 * * *  (chạy lúc 2:00 AM mỗi ngày)

  Task 1: crawl_sources          (crawl VOZ, VnExpress, YouTube)
  Task 2: export_mongo_to_hdfs   (export raw JSON → HDFS)
  Task 3: spark_cleaning         (Spark job: clean + dedup LSH)
  Task 4: spark_nlp              (Spark job: PhoBERT sentiment)
  Task 5: topic_modeling         (BERTopic/LDA weekly)
  Task 6: trend_scoring          (Tính TrendScore + PageRank)
  Task 7: crisis_detection       (Isolation Forest check)
  Task 8: load_to_postgres       (Write kết quả → PostgreSQL)
  Task 9: refresh_dashboard      (Notify Streamlit cache clear)
```

> **Lý do chọn Airflow thay vì Cron:** Airflow cung cấp UI giám sát, retry tự động, dependency giữa tasks, và logging tập trung — cần thiết với pipeline nhiều bước.

### 9.2 Docker

```yaml
# docker-compose.yml (development environment)
services:
  mongodb:
    image: mongo:7
    ports: ["27017:27017"]
  postgres:
    image: postgres:16
    environment:
      POSTGRES_DB: nlp_db
  airflow:
    image: apache/airflow:2.9.0
  streamlit:
    build: ./dashboard
    ports: ["8501:8501"]
```

### 9.3 Git Workflow

```
main          ← Production-ready code
develop       ← Integration branch
feature/*     ← Mỗi thành viên làm việc trên branch riêng
hotfix/*      ← Sửa lỗi khẩn cấp
```

### 9.4 Dependency Management

```bash
# Môi trường Python trên HPC
conda create -n nlp-trend python=3.10
conda activate nlp-trend
pip install -r requirements.txt
```

---

## 10. Infrastructure & HPC Cluster

### 10.1 Ansible — Tự động hóa cấu hình

```
Ansible Playbooks:
  setup_java.yml        → Cài Java 8+ (cần cho VnCoreNLP)
  setup_spark.yml       → Cài Spark 3.5 trên tất cả nodes
  setup_hdfs.yml        → Cấu hình HDFS NameNode + DataNodes
  setup_mongodb.yml     → Cài MongoDB trên storage node
  setup_postgres.yml    → Cài PostgreSQL trên storage node
  setup_airflow.yml     → Cài Airflow trên master node
  deploy_app.yml        → Deploy Streamlit dashboard
```

### 10.2 Cấu hình HPC Cluster

```
Cluster: HPC Semi-Lab
  ├── Master Node (1 node)
  │     └── HDFS NameNode, Spark Master, Airflow, Streamlit
  ├── Worker Nodes (3–4 nodes)
  │     └── HDFS DataNode, Spark Worker
  │     └── Chạy PhoBERT inference (nếu có GPU)
  └── Storage Node (1 node)
        └── MongoDB, PostgreSQL
```

| Node | RAM | CPU | Vai trò |
|---|---|---|---|
| Master | 16 GB | 8 cores | Spark Master, HDFS NameNode, Airflow |
| Worker x3 | 8 GB | 4 cores | Spark Executor, HDFS DataNode |
| Storage | 8 GB | 4 cores | MongoDB, PostgreSQL |

---

## 11. Yêu cầu phần cứng & phụ thuộc hệ thống

### 11.1 Phụ thuộc hệ thống (cần cài trên toàn cluster)

| Phụ thuộc | Phiên bản | Lý do |
|---|---|---|
| **Java JDK** | 8 hoặc 11 | VnCoreNLP, Spark, HDFS đều yêu cầu |
| **Python** | 3.10.x | Tương thích với tất cả thư viện |
| **CUDA** | 11.8+ | Nếu chạy PhoBERT/BERTopic trên GPU |
| **Hadoop** | 3.3.x | HDFS |
| **Spark** | 3.5.x | Core processing |

### 11.2 Thư viện Python chính (requirements.txt)

```txt
# Crawling
requests==2.31.0
beautifulsoup4==4.12.3
selenium==4.18.0
undetected-chromedriver==3.5.5
google-api-python-client==2.120.0

# Big Data
pyspark==3.5.1
pyarrow==14.0.2
pymongo==4.6.2

# NLP
transformers==4.39.0
torch==2.2.0
vncorenlp==1.0.3           # Yêu cầu Java
underthesea==6.8.0
sentence-transformers==2.6.1
bertopic==0.16.0

# ML
scikit-learn==1.4.0
numpy==1.26.4
pandas==2.2.1
mmh3==4.1.0                # Count-Min Sketch hashing

# Database
psycopg2-binary==2.9.9

# Visualization
streamlit==1.32.0
plotly==5.20.0
matplotlib==3.8.0
wordcloud==1.9.3
networkx==3.2.1

# Orchestration
apache-airflow==2.9.0
```

---

## 12. Performance Benchmarking Plan

> Yêu cầu Phase 4: Đánh giá hiệu năng của hệ thống phân tán.

### 12.1 Chỉ số cần đo

| Chỉ số | Công cụ đo | Mục tiêu |
|---|---|---|
| Spark job execution time | Spark UI / `time` | So sánh 1 node vs 3 nodes |
| LSH deduplication accuracy | Ground truth dataset | Precision ≥ 0.90 |
| PhoBERT throughput | records/giây | ≥ 500 records/s trên 3 workers |
| Sentiment F1-score | scikit-learn `classification_report` | Macro F1 ≥ 0.75 |
| Speedup ratio | $S = T_1 / T_n$ | Linear speedup target |
| Pipeline end-to-end latency | Airflow task duration log | < 30 phút cho 100K records |

### 12.2 Thực nghiệm Scalability

```
Thực nghiệm 1: Strong Scaling
  Dataset cố định: 500K records
  Số workers: 1 → 2 → 3 → 4
  Đo: Execution time, Speedup ratio

Thực nghiệm 2: Weak Scaling
  Tăng dataset tỷ lệ với số workers
  1 worker: 100K records
  2 workers: 200K records
  3 workers: 300K records
  Đo: Execution time (lý tưởng là hằng số)
```

---

## 13. Phân công nhóm

| Thành viên | Trách nhiệm chính | Deliverable |
|---|---|---|
| **Member 1** | Data Crawling (VOZ, VnExpress, YouTube) + MongoDB setup | Scripts crawler, raw dataset |
| **Member 2** | Ansible + Spark environment + HDFS pipeline + LSH/MinHash | Playbooks, Spark cleaning job |
| **Member 3** | Topic Modeling (LDA + BERTopic) + Count-Min Sketch | Topic model, keyword frequency |
| **Member 4** | PhoBERT fine-tuning + Sentiment pipeline + Isolation Forest | Model checkpoint, evaluation report |
| **Member 5** | Trend Scoring + PageRank/HITS + Streamlit Dashboard | Score engine, dashboard app |

---

## 14. Dependency Summary

```
┌──────────────────────────────────────────────────────────┐
│ Layer              │ Tool / Library          │ Version   │
├──────────────────────────────────────────────────────────┤
│ Crawling           │ Requests, BS4, Selenium │ Latest    │
│                    │ YouTube Data API v3     │ v3        │
├──────────────────────────────────────────────────────────┤
│ Storage            │ MongoDB                 │ 7.x       │
│                    │ HDFS (Hadoop)           │ 3.3.x     │
│                    │ PostgreSQL              │ 16.x      │
├──────────────────────────────────────────────────────────┤
│ Big Data           │ Apache Spark (PySpark)  │ 3.5.x     │
│                    │ GraphFrames             │ 0.12.0    │
│                    │ mongo-spark-connector   │ 10.3.x    │
├──────────────────────────────────────────────────────────┤
│ NLP                │ VnCoreNLP / underthesea │ Latest    │
│                    │ PhoBERT (HuggingFace)   │ base      │
│                    │ BERTopic                │ 0.16.x    │
│                    │ PyTorch                 │ 2.2.0     │
├──────────────────────────────────────────────────────────┤
│ ML                 │ scikit-learn            │ 1.4.x     │
│                    │ Isolation Forest        │ (sklearn) │
├──────────────────────────────────────────────────────────┤
│ Visualization      │ Streamlit               │ 1.32.x    │
│                    │ Plotly                  │ 5.20.x    │
├──────────────────────────────────────────────────────────┤
│ Orchestration      │ Apache Airflow          │ 2.9.0     │
│                    │ Docker                  │ 24.x      │
│                    │ Ansible                 │ 2.16.x    │
├──────────────────────────────────────────────────────────┤
│ System             │ Java JDK                │ 11        │
│                    │ Python                  │ 3.10.x    │
│                    │ CUDA (optional GPU)     │ 11.8+     │
└──────────────────────────────────────────────────────────┘
```

---

## Changelog

| Phiên bản | Ngày | Thay đổi |
|---|---|---|
| v1.0 | 2026-02-01 | Bản khởi thảo ban đầu (README) |
| v2.0 | 2026-02-25 | Bổ sung HDFS, CS246 algorithms, Ansible, Docker, Benchmarking plan, Java dependency, full requirements.txt |
