# Chương 4. Thiết kế kiến trúc hệ thống

## 4.1. Tổng quan kiến trúc

Hệ thống **Vietnamese Tech Trend & Controversy Radar** được thiết kế theo kiến trúc nhiều tầng, kết hợp giữa Big Data, NLP, Data Warehouse và Dashboard. Mục tiêu của kiến trúc là xây dựng một pipeline có khả năng xử lý dữ liệu từ nhiều nguồn, chuẩn hóa dữ liệu, phân tích chủ đề/cảm xúc, phát hiện bất thường và hiển thị kết quả cho người dùng cuối.

Luồng tổng thể của hệ thống:

```text
Nguồn dữ liệu web
-> Crawler
-> HDFS Raw Zone
-> Spark Processing
-> HDFS Staged Parquet
-> ClickHouse Staging
-> dbt Intermediate/Marts
-> Next.js Dashboard
```

Kiến trúc có thể chia thành các layer chính:

| Layer | Thành phần | Vai trò |
|---|---|---|
| Ingestion Layer | Python crawlers | Thu thập dữ liệu từ web |
| Storage Layer | HDFS | Lưu raw data và staged data |
| Processing Layer | Apache Spark | Làm sạch, chuẩn hóa, NLP, ML jobs |
| Orchestration Layer | Apache Airflow | Điều phối pipeline end-to-end |
| Warehouse Layer | ClickHouse | Lưu dữ liệu staging và dữ liệu phân tích |
| Transformation Layer | dbt | Join, aggregate, tính trend score, tạo mart |
| Serving Layer | Next.js Dashboard | Hiển thị kết quả phân tích |
| Deployment Layer | Docker, Ansible | Đóng gói và tự động hóa triển khai |

Sơ đồ logic:

```text
                +----------------------+
                |   Web Data Sources   |
                | VOZ / VnExpress /... |
                +----------+-----------+
                           |
                           v
                +----------------------+
                |       Crawlers       |
                +----------+-----------+
                           |
                           v
                +----------------------+
                |       HDFS Raw       |
                +----------+-----------+
                           |
                           v
                +----------------------+
                |   Spark Processing   |
                | Clean / Dedup / NLP  |
                +----------+-----------+
                           |
                           v
                +----------------------+
                | HDFS Staged Parquet  |
                +----------+-----------+
                           |
                           v
                +----------------------+
                | ClickHouse Staging   |
                +----------+-----------+
                           |
                           v
                +----------------------+
                |      dbt Marts       |
                +----------+-----------+
                           |
                           v
                +----------------------+
                |   Next.js Dashboard  |
                +----------------------+
```

## 4.2. Nguyên tắc thiết kế

Khi thiết kế hệ thống, nhóm áp dụng một số nguyên tắc chính:

### 4.2.1. Tách biệt các tầng xử lý

Dữ liệu được chia thành nhiều tầng:

- Raw data: dữ liệu gốc từ crawler.
- Staged Parquet: dữ liệu đã được xử lý bởi Spark.
- ClickHouse staging: dữ liệu được load vào warehouse.
- dbt marts: dữ liệu đã được join, aggregate và tính toán chỉ số.
- Dashboard: tầng hiển thị.

Việc tách tầng giúp hệ thống dễ debug. Nếu kết quả dashboard sai, nhóm có thể kiểm tra lần lượt từ mart, staging, staged Parquet cho đến raw data.

### 4.2.2. Không ghi đè kết quả NLP vào bảng lõi

Một quyết định quan trọng là không ghi trực tiếp sentiment và topic vào `stg_posts_core`. Thay vào đó, mỗi module tạo output riêng:

```text
stg_posts_core      <- dữ liệu lõi từ M2
stg_posts_nlp       <- sentiment từ M4
stg_post_topics     <- topic assignment từ M3
stg_topics          <- topic metadata từ M3
stg_keyword_freq    <- keyword frequency từ M3
stg_crisis_events   <- crisis detection từ M4
```

Các bảng này được join ở tầng dbt. Cách thiết kế này có các lợi ích:

- Các job M2, M3, M4 có thể chạy độc lập.
- Nếu sentiment job lỗi, không cần chạy lại cleaning job.
- Nếu topic model được cập nhật, có thể ghi lại `stg_post_topics` mà không ảnh hưởng `stg_posts_core`.
- Tránh Spark shuffle lớn khi join nhiều output NLP trong Spark.
- Warehouse/dbt là nơi hợp nhất dữ liệu, phù hợp mô hình ELT.

### 4.2.3. Giữ các metric atomic

Trong tầng Spark, hệ thống giữ các chỉ số như `reaction_count`, `comment_count`, `view_count` ở dạng atomic. Spark không tính sẵn `engagement`.

Lý do:

- Công thức engagement có thể thay đổi.
- Nếu tính engagement trong Spark, mỗi lần đổi công thức phải chạy lại Spark job.
- Khi giữ metric atomic, dbt có thể tính lại engagement bằng SQL.

Trong `int_posts_enriched.sql`, engagement được tính từ các biến dbt:

```text
engagement = reaction_count * weight_reaction
           + comment_count * weight_comment
           + view_count * weight_view
```

### 4.2.4. Dashboard chỉ đọc dữ liệu mart

Dashboard không đọc trực tiếp raw data, HDFS hoặc staging tables. Dashboard query các bảng mart đã được dbt chuẩn bị sẵn:

```text
dbt_fct_topic_activity
dbt_dim_topics
dbt_fct_crisis_events
dbt_int_posts_enriched
```

Cách này giúp:

- Dashboard query nhanh hơn.
- Logic phân tích tập trung ở dbt.
- Giao diện không phải xử lý join phức tạp.
- Tránh làm nặng database bằng các truy vấn raw lớn.

## 4.3. Kiến trúc hạ tầng

Hệ thống được triển khai bằng Docker Compose trong môi trường local/development. Ngoài ra, repo cũng có Ansible để mô tả việc tự động hóa cài đặt cluster.

### 4.3.1. Các thành phần hạ tầng chính

| Thành phần | Vai trò |
|---|---|
| HDFS NameNode | Quản lý metadata của file system |
| HDFS DataNode | Lưu dữ liệu vật lý |
| Spark Master | Điều phối Spark workers |
| Spark Worker | Thực thi Spark tasks |
| Airflow Webserver | Giao diện quản lý DAG |
| Airflow Scheduler | Lên lịch và trigger task |
| ClickHouse | Kho dữ liệu OLAP |
| Dashboard Next.js | Giao diện hiển thị kết quả |

Các file cấu hình chính:

```text
docker-compose.yml
Dockerfile.spark
Dockerfile.airflow
env.example
LOCAL_GUIDE.md
```

### 4.3.2. Docker Compose

Docker Compose giúp khởi động các service của hệ thống trong cùng một network. Các container có thể giao tiếp với nhau bằng hostname nội bộ, ví dụ:

```text
namenode:9000
spark-master:7077
clickhouse:8123
airflow-webserver:8080
```

Trong môi trường local, các cổng thường dùng:

| Service | Cổng |
|---|---|
| Airflow UI | `8081` |
| Spark UI | `8080` |
| HDFS Web UI | `9870` |
| ClickHouse HTTP | `8123` |
| Dashboard | `3000` |

Docker Compose giúp nhóm có thể demo pipeline mà không cần cài thủ công toàn bộ stack trên máy.

### 4.3.3. Ansible

Ansible được dùng để tự động hóa việc cài đặt và cấu hình khi triển khai theo mô hình cluster. Các playbook chính:

```text
ansible/playbooks/01_java.yml
ansible/playbooks/02_conda.yml
ansible/playbooks/03_hdfs.yml
ansible/playbooks/04_spark.yml
ansible/playbooks/05_clickhouse.yml
ansible/playbooks/06_dbt.yml
ansible/playbooks/07_airflow.yml
ansible/playbooks/install_nlp_deps.yml
ansible/playbooks/start_services.yml
```

Vai trò của Ansible:

- Cài Java cho Hadoop/Spark.
- Cài Conda và môi trường Python.
- Cấu hình HDFS.
- Cấu hình Spark.
- Cài ClickHouse.
- Cài dbt.
- Cài Airflow.
- Cài các dependency NLP.

Nhờ Ansible, quá trình triển khai có thể tái lập, giảm phụ thuộc vào thao tác thủ công.

## 4.4. Kiến trúc luồng dữ liệu

Luồng dữ liệu chính gồm các bước:

```text
Raw crawler files
-> upload_to_hdfs
-> spark_cleaning
-> stg_posts_core Parquet
-> ingest_stg_posts_core
-> ClickHouse stg_posts_core
-> topic/sentiment/CMS/crisis jobs
-> ClickHouse staging tables
-> dbt_transform
-> dashboard marts
-> dashboard
```

### 4.4.1. Tầng raw data

Crawler thu thập dữ liệu từ các nguồn như VOZ, VnExpress và VatVo. Dữ liệu raw có thể được lưu dưới dạng CSV/local files, sau đó được upload lên HDFS bằng:

```text
crawlers/upload_to_hdfs.py
```

Raw data là đầu vào cho Spark cleaning job.

### 4.4.2. Tầng Spark cleaning

Spark cleaning job nằm trong:

```text
spark_jobs/cleaning_job.py
```

Job này thực hiện:

- Đọc raw data.
- Chuẩn hóa dữ liệu từ nhiều nguồn.
- Làm sạch text.
- Tách từ tiếng Việt.
- Tạo các cột như `body`, `clean_text`, `segmented_text`, `topic_text`.
- Loại bỏ record rỗng/lỗi.
- Loại bỏ exact duplicate.
- Loại bỏ near-duplicate bằng MinHash/LSH.
- Ghi output Parquet ra `stg_posts_core`.

`stg_posts_core` là dataset lõi của toàn hệ thống.

### 4.4.3. Tầng NLP và ML outputs

Sau khi có `stg_posts_core`, các job khác chạy để tạo output riêng:

| Job | Input | Output |
|---|---|---|
| LDA topic modeling | `stg_posts_core` | `stg_post_topics`, `stg_topics` |
| BERTopic inference | `stg_posts_core` | `stg_post_topics`, `stg_topics` |
| Sentiment analysis | `stg_posts_core` | `stg_posts_nlp` |
| Count-Min Sketch | `stg_posts_core` | `stg_keyword_freq` |
| Crisis detection | `stg_posts_core`, `stg_posts_nlp` | `stg_crisis_events` |

Thiết kế này giúp mỗi module phụ trách một output riêng, giảm phụ thuộc chặt giữa các nhóm.

### 4.4.4. Tầng ClickHouse staging

Các staged Parquet được load vào ClickHouse bằng:

```text
scripts/hdfs_to_clickhouse.py
```

Các bảng staging:

```text
tech_radar.stg_posts_core
tech_radar.stg_posts_nlp
tech_radar.stg_post_topics
tech_radar.stg_topics
tech_radar.stg_keyword_freq
tech_radar.stg_crisis_events
```

Mỗi dataset có ingest mode riêng:

| Dataset | Mode |
|---|---|
| `stg_posts_core` | full_refresh |
| `stg_posts_nlp` | replace_latest |
| `stg_post_topics` | replace_latest |
| `stg_topics` | replace_latest |
| `stg_keyword_freq` | append |
| `stg_crisis_events` | append, allow_empty |

Việc định nghĩa ingest contract rõ ràng giúp tránh lỗi lệch schema và tránh insert nhầm cột.

### 4.4.5. Tầng dbt marts

dbt đọc dữ liệu từ ClickHouse staging và tạo các bảng phục vụ dashboard.

Các model chính:

| Model | Vai trò |
|---|---|
| `stg_posts_core` | Chuẩn hóa bảng core từ ClickHouse source |
| `stg_posts_nlp` | Chuẩn hóa sentiment table |
| `stg_post_topics` | Chuẩn hóa topic assignment |
| `stg_topics` | Chuẩn hóa topic lookup |
| `int_posts_enriched` | Join post core với sentiment và topic |
| `int_topic_sentiment_hourly` | Tổng hợp topic/sentiment theo giờ |
| `fct_topic_activity` | Fact table chính cho trend |
| `dim_topics` | Dimension topic |
| `fct_crisis_events` | Mart cho crisis monitor |

dbt cũng tính các chỉ số như:

- `engagement`
- `velocity`
- `acceleration`
- `engagement_normalized`
- `trend_score`
- `neg_ratio`
- `z_score_neg_ratio`
- `volume_zscore`

## 4.5. Kiến trúc Airflow DAG

DAG chính của hệ thống là:

```text
full_processing_pipeline
```

File:

```text
dags/processing_dag.py
```

Luồng task:

```text
pipeline_start
-> validate_runtime_mounts
-> crawl_sources
-> upload_reference_files_to_hdfs
-> spark_cleaning
-> ingest_stg_posts_core
-> lda_topic_modeling
-> sentiment_analysis
-> ingest_stg_posts_nlp
-> ingest_stg_post_topics
-> ingest_stg_topics
-> cms_keyword_counting
-> ingest_stg_keyword_freq
-> crisis_to_hdfs_stg_crisis_events
-> ingest_stg_crisis_events
-> dbt_transform
-> pipeline_end
```

### 4.5.1. Nhóm task ingestion

| Task | Vai trò |
|---|---|
| `validate_runtime_mounts` | Kiểm tra raw data, reference files và model files |
| `crawl_sources` | Upload raw crawler data lên HDFS |
| `upload_reference_files_to_hdfs` | Upload stopwords/slang dictionary lên HDFS |

### 4.5.2. Nhóm task core processing

| Task | Vai trò |
|---|---|
| `spark_cleaning` | Chạy Spark job tạo `stg_posts_core` |
| `ingest_stg_posts_core` | Load `stg_posts_core` vào ClickHouse |

### 4.5.3. Nhóm task NLP/ML

| Task | Vai trò |
|---|---|
| `lda_topic_modeling` | Chạy LDA topic modeling |
| `sentiment_analysis` | Chạy PhoBERT sentiment inference |
| `cms_keyword_counting` | Chạy Count-Min Sketch keyword counting |
| `crisis_to_hdfs_stg_crisis_events` | Chạy crisis detection |

### 4.5.4. Nhóm task ingest và transformation

| Task | Vai trò |
|---|---|
| `ingest_stg_posts_nlp` | Load sentiment vào ClickHouse |
| `ingest_stg_post_topics` | Load topic assignments |
| `ingest_stg_topics` | Load topic metadata |
| `ingest_stg_keyword_freq` | Load keyword frequency |
| `ingest_stg_crisis_events` | Load crisis events |
| `dbt_transform` | Chạy dbt để build marts |

### 4.5.5. Lợi ích của thiết kế DAG

Thiết kế DAG giúp:

- Pipeline chạy theo đúng thứ tự phụ thuộc.
- Dễ quan sát task nào lỗi.
- Dễ trigger demo.
- Các bước ingest tách riêng với Spark job.
- Các output staged có thể kiểm tra trước khi load vào ClickHouse.
- Dễ retry từng task mà không cần chạy lại toàn bộ pipeline.

## 4.6. Thiết kế schema và bảng dữ liệu

### 4.6.1. Bảng `stg_posts_core`

`stg_posts_core` là bảng lõi sau khi dữ liệu đã được clean và dedup.

Các nhóm cột chính:

| Nhóm | Cột |
|---|---|
| Định danh | `post_id`, `parent_id` |
| Nguồn | `source` |
| Tác giả | `author_id`, `author_name` |
| Nội dung | `title`, `body`, `segmented_text` |
| Tương tác | `reaction_count`, `comment_count`, `view_count` |
| Thời gian | `created_at`, `crawled_at`, `loaded_at` |

Bảng này dùng engine `MergeTree`, partition theo tháng của `created_at`, order theo `(source, created_at, post_id)`.

### 4.6.2. Bảng `stg_posts_nlp`

Lưu kết quả sentiment:

```text
post_id
sentiment_label
sentiment_score
model_version
predicted_at
loaded_at
```

Bảng dùng `ReplacingMergeTree(loaded_at)` để khi chạy lại inference, bản mới có thể thay thế bản cũ theo `post_id`.

### 4.6.3. Bảng `stg_post_topics` và `stg_topics`

`stg_post_topics` lưu topic assignment:

```text
post_id
topic_id
topic_probability
model_type
predicted_at
loaded_at
```

`stg_topics` lưu metadata của topic:

```text
topic_id
label
top_keywords
coherence_score
model_version
created_at
```

### 4.6.4. Bảng `stg_keyword_freq`

Lưu keyword frequency từ Count-Min Sketch:

```text
keyword
window_start
window_end
estimated_count
source
```

Bảng này phục vụ việc theo dõi keyword nổi bật theo thời gian và nguồn dữ liệu.

### 4.6.5. Bảng `stg_crisis_events`

Lưu các sự kiện bất thường:

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
```

Bảng này có thể rỗng nếu trong ngày không có crisis event. Đây là trạng thái hợp lệ, không phải lỗi pipeline.

## 4.7. Thiết kế trend score và marts

Trend score được tính ở tầng dbt, không tính trong Spark. Điều này giúp thay đổi công thức dễ hơn.

Trong `fct_topic_activity`, các thành phần chính gồm:

- `mention_count`: số lượng bài/comment theo topic và giờ.
- `velocity`: tốc độ mention hiện tại.
- `acceleration`: độ thay đổi của velocity so với bucket trước.
- `engagement_sum`: tổng engagement.
- `engagement_normalized`: engagement đã chuẩn hóa.
- `neg_ratio`: tỷ lệ sentiment tiêu cực.
- `trend_score`: điểm xu hướng tổng hợp.

Công thức tổng quát:

```text
TrendScore = α * Velocity + β * Acceleration + γ * EngagementNormalized
```

Trong dự án, công thức được đặt trong macro:

```text
warehouse/dbt_project/macros/trend_score_calc.sql
```

Việc đặt công thức trong dbt macro giúp tái sử dụng và dễ điều chỉnh trọng số.

## 4.8. Kiến trúc dashboard

Dashboard được xây dựng bằng Next.js. Dashboard không xử lý dữ liệu thô mà chỉ query dữ liệu đã được chuẩn bị trong ClickHouse marts.

### 4.8.1. Các trang chính

| Trang | Chức năng |
|---|---|
| Overview | Hiển thị KPI, top trending topics, sentiment tổng quan |
| Trends Explorer | Phân tích chi tiết topic, trend score, sentiment timeline, keywords, evidence posts |
| Crisis Monitor | Hiển thị crisis events, severity, trigger conditions và evidence |

### 4.8.2. Data access layer

Dashboard kết nối ClickHouse thông qua:

```text
dashboard/src/lib/clickhouse.ts
dashboard/src/lib/dal/radar.ts
```

Các hàm query chính:

- `getOverviewKPIsFromCH`
- `getTrendingTopicsFromCH`
- `getOverallSentimentFromCH`
- `getActiveTopicsFromCH`
- `getTopicTrendScoreFromCH`
- `getTopicSentimentFromCH`
- `getTopicKeywordsFromCH`
- `getTopicEvidencePostsFromCH`
- `getRecentCrisesFromCH`
- `getCrisisStatsFromCH`

### 4.8.3. Health check

Dashboard có endpoint kiểm tra kết nối ClickHouse:

```text
/api/health/clickhouse
```

Endpoint này giúp xác định dashboard có kết nối được database và thấy các bảng cần thiết hay không.

## 4.9. Thiết kế khả năng mở rộng

Hệ thống có thể mở rộng theo một số hướng:

### 4.9.1. Mở rộng dữ liệu

Khi thêm nguồn mới, chỉ cần:

1. Viết crawler mới.
2. Viết adapter mới để map dữ liệu về schema chung.
3. Thêm source vào pipeline.

Nếu source mới vẫn map được về `stg_posts_core`, các module M3/M4/M5 có thể dùng tiếp mà không cần sửa lớn.

### 4.9.2. Mở rộng xử lý

Spark cho phép tăng số worker/executor để xử lý nhiều dữ liệu hơn. Các job như cleaning, LDA và sentiment inference có thể tận dụng partition để chạy song song.

### 4.9.3. Mở rộng warehouse

ClickHouse phù hợp với truy vấn OLAP và dữ liệu dạng cột. Khi dữ liệu tăng, có thể tối ưu bằng:

- Partition theo thời gian.
- Order key hợp lý.
- Materialized view hoặc pre-aggregated tables.
- TTL để giới hạn thời gian lưu.

### 4.9.4. Mở rộng dashboard

Dashboard có thể bổ sung thêm:

- Bộ lọc theo nguồn.
- Bộ lọc theo topic.
- Alert realtime.
- Xuất báo cáo CSV/PDF.
- Trang phân tích chi tiết theo thương hiệu/sản phẩm.

## 4.10. Thiết kế khả năng chịu lỗi và debug

Hệ thống có nhiều điểm hỗ trợ debug:

- Airflow hiển thị trạng thái từng task.
- `validate_runtime_mounts` giúp fail sớm nếu thiếu input.
- HDFS staged Parquet cho phép kiểm tra output từng job.
- ClickHouse staging tables cho phép kiểm tra số dòng từng dataset.
- dbt models tách staging/intermediate/mart rõ ràng.
- Dashboard health endpoint giúp kiểm tra kết nối ClickHouse.

Một số trường hợp đặc biệt được xử lý:

- Nếu không có crisis event trong ngày, `stg_crisis_events` có thể rỗng và ingest vẫn có thể pass.
- Nếu sentiment hoặc topic chưa có, dbt dùng LEFT JOIN để dữ liệu core vẫn tồn tại.
- Nếu model artifact crisis chưa sẵn sàng, crisis detection có thể fallback rule-based.

## 4.11. Đánh giá thiết kế kiến trúc

### 4.11.1. Ưu điểm

- Kiến trúc rõ ràng, tách biệt crawler, processing, warehouse và dashboard.
- Spark xử lý dữ liệu lớn tốt hơn so với xử lý local.
- HDFS lưu được raw và staged data, thuận tiện kiểm tra lineage.
- ClickHouse phù hợp dashboard analytics.
- dbt giúp quản lý SQL transformation rõ ràng.
- Airflow giúp tự động hóa pipeline end-to-end.
- Các output NLP tách riêng, dễ retry và bảo trì.

### 4.11.2. Hạn chế

- Pipeline hiện tại chủ yếu là batch, chưa phải real-time streaming.
- Một số job ML/NLP cần nhiều tài nguyên, đặc biệt PhoBERT và BERTopic.
- MinHash/LSH hiện tại có bước gom dữ liệu về driver, cần tối ưu nếu dữ liệu rất lớn.
- Chất lượng hệ thống phụ thuộc vào chất lượng crawler và dữ liệu raw.
- Crisis detection cần thêm dữ liệu lịch sử và dữ liệu gán nhãn để đánh giá tốt hơn.

## 4.12. Tổng kết chương

Chương này đã trình bày thiết kế kiến trúc tổng thể của hệ thống, bao gồm kiến trúc hạ tầng, kiến trúc luồng dữ liệu, Airflow DAG, schema ClickHouse, dbt marts và dashboard. Hệ thống được thiết kế theo hướng nhiều tầng, tách biệt rõ raw data, staged data, warehouse và serving layer. Cách thiết kế này giúp pipeline dễ mở rộng, dễ debug và phù hợp với bài toán phân tích dữ liệu mạng xã hội tiếng Việt ở quy mô lớn.

