# Khung báo cáo đồ án - Vietnamese Tech Trend & Controversy Radar

## Thông tin cần bổ sung trước khi nộp

Các thông tin dưới đây cần nhóm điền theo yêu cầu thực tế của giảng viên/khoa:

- Tên trường, khoa, bộ môn.
- Tên học phần.
- Tên giảng viên hướng dẫn.
- Danh sách thành viên, mã sinh viên, lớp.
- Vai trò cụ thể của từng thành viên.
- Thời gian thực hiện đồ án.
- Số liệu thực nghiệm cuối cùng: số record, thời gian chạy pipeline, kết quả benchmark, kết quả model.
- Ảnh chụp demo dashboard, Airflow, Spark UI, ClickHouse/dbt nếu giảng viên yêu cầu minh chứng.

---

# Trang bìa

**Tên đề tài:** Xây dựng hệ thống phân tích xu hướng và dư luận mạng xã hội tiếng Việt trên nền tảng Big Data và NLP  
**Tên tiếng Anh đề xuất:** Vietnamese Tech Trend & Controversy Radar  

**Môn học:** [Điền tên môn học]  
**Giảng viên hướng dẫn:** [Điền tên giảng viên]  
**Nhóm thực hiện:** [Tên nhóm]  
**Thành viên:**  

| STT | Họ tên | Mã sinh viên | Vai trò |
|---|---|---|---|
| 1 | [Member 1] | [Mã SV] | Data Ingestion / Crawling |
| 2 | [Member 2] | [Mã SV] | DevOps / Data Infrastructure & Spark Processing |
| 3 | [Member 3] | [Mã SV] | Topic Modeling / Count-Min Sketch |
| 4 | [Member 4] | [Mã SV] | Sentiment Analysis / Crisis Detection |
| 5 | [Member 5] | [Mã SV] | Data Warehouse / Dashboard |

---

# Lời cảm ơn

Viết ngắn gọn 1/2 trang:

- Cảm ơn giảng viên hướng dẫn.
- Cảm ơn nhà trường/khoa đã tạo điều kiện.
- Cảm ơn các nguồn tài liệu, thư viện mã nguồn mở được sử dụng trong đồ án.

---

# Tóm tắt đồ án

Phần này nên dài khoảng 1/2 đến 1 trang.

Nội dung cần trình bày:

- Bối cảnh: lượng dữ liệu mạng xã hội tiếng Việt ngày càng lớn, cần hệ thống tự động thu thập và phân tích.
- Mục tiêu: xây dựng hệ thống end-to-end để thu thập dữ liệu, xử lý phân tán, phân tích chủ đề, phân tích cảm xúc, phát hiện khủng hoảng và hiển thị dashboard.
- Công nghệ chính: Python, Spark, HDFS, Airflow, ClickHouse, dbt, Next.js, PhoBERT, LDA, BERTopic, MinHash/LSH, Count-Min Sketch, Isolation Forest.
- Kết quả: hệ thống chạy được pipeline từ crawler đến dashboard, tạo các bảng phân tích xu hướng, cảm xúc và cảnh báo khủng hoảng.

Đoạn mẫu:

> Đồ án xây dựng một hệ thống phân tích xu hướng và dư luận mạng xã hội tiếng Việt trong lĩnh vực công nghệ. Hệ thống thu thập dữ liệu từ các nguồn như VOZ, VnExpress và VatVo, sau đó xử lý dữ liệu bằng Apache Spark, lưu trữ trên HDFS và ClickHouse. Các mô hình NLP được sử dụng để phân tích chủ đề, cảm xúc và phát hiện dấu hiệu khủng hoảng dư luận. Kết quả cuối cùng được trực quan hóa trên dashboard, giúp người dùng theo dõi các chủ đề đang nổi bật, diễn biến cảm xúc và các sự kiện bất thường theo thời gian.

---

# Mục lục đề xuất

1. Giới thiệu đề tài  
2. Cơ sở lý thuyết và công nghệ sử dụng  
3. Phân tích yêu cầu hệ thống  
4. Thiết kế kiến trúc hệ thống  
5. Thiết kế dữ liệu và luồng xử lý  
6. Xây dựng các module chính  
7. Điều phối pipeline và triển khai hệ thống  
8. Kết quả thực nghiệm và đánh giá  
9. Phân công công việc trong nhóm  
10. Kết luận và hướng phát triển  
11. Tài liệu tham khảo  
12. Phụ lục  

---

# Chương 1. Giới thiệu đề tài

## 1.1. Bối cảnh

Trình bày vấn đề thực tế:

- Mạng xã hội, diễn đàn và báo điện tử tạo ra lượng lớn nội dung tiếng Việt mỗi ngày.
- Các chủ đề công nghệ có thể thay đổi nhanh, ví dụ sản phẩm mới, lỗi thiết bị, tranh cãi về thương hiệu.
- Việc theo dõi thủ công mất thời gian và khó phát hiện sớm xu hướng hoặc khủng hoảng.
- Dữ liệu tiếng Việt có nhiều khó khăn: dấu câu không chuẩn, teencode, slang, emoji, comment ngắn, nhiều nguồn dữ liệu khác nhau.

## 1.2. Lý do chọn đề tài

Nêu các lý do:

- Phù hợp với bài toán Big Data vì dữ liệu lớn, nhiều nguồn, cần xử lý phân tán.
- Phù hợp với NLP tiếng Việt vì cần tiền xử lý, tách từ, phân tích cảm xúc và phân tích chủ đề.
- Có tính ứng dụng thực tế trong social listening, marketing, quản trị truyền thông, theo dõi sản phẩm công nghệ.
- Có thể kết hợp nhiều kỹ thuật trong môn học: crawling, distributed processing, data warehouse, machine learning, dashboard.

## 1.3. Mục tiêu đồ án

Các mục tiêu chính:

- Xây dựng pipeline thu thập dữ liệu từ các nguồn tiếng Việt.
- Lưu trữ dữ liệu thô và dữ liệu đã xử lý trên HDFS.
- Xử lý dữ liệu phân tán bằng Spark.
- Chuẩn hóa dữ liệu về schema thống nhất.
- Loại bỏ dữ liệu trùng/gần trùng bằng MinHash/LSH.
- Phân tích chủ đề bằng LDA/BERTopic.
- Đếm tần suất keyword bằng Count-Min Sketch.
- Phân tích cảm xúc bằng PhoBERT.
- Phát hiện dấu hiệu khủng hoảng bằng các đặc trưng sentiment/volume/anomaly.
- Xây dựng warehouse bằng ClickHouse và dbt.
- Hiển thị kết quả trên dashboard.

## 1.4. Phạm vi đề tài

Nêu rõ phạm vi:

- Nguồn dữ liệu: VOZ, VnExpress, VatVo; nếu có YouTube/Tinhte thì ghi theo tình trạng thực tế.
- Dữ liệu tập trung vào lĩnh vực công nghệ.
- Hệ thống xử lý theo batch/daily pipeline, không phải streaming real-time tuyệt đối.
- Dashboard hiển thị overview, trends explorer và crisis monitor.
- Các mô hình ML/NLP tập trung vào thực nghiệm trong phạm vi đồ án, chưa phải sản phẩm production quy mô lớn.

## 1.5. Kết quả đạt được

Liệt kê ngắn gọn:

- Crawler và raw data.
- Hạ tầng Docker/Ansible.
- Spark cleaning pipeline.
- HDFS staged datasets.
- ClickHouse staging tables.
- dbt intermediate/mart tables.
- LDA/BERTopic topic assignments.
- PhoBERT sentiment labels.
- Count-Min Sketch keyword frequency.
- Crisis events.
- Dashboard Next.js.

---

# Chương 2. Cơ sở lý thuyết và công nghệ sử dụng

## 2.1. Big Data và xử lý phân tán

Nội dung cần trình bày:

- Khái niệm dữ liệu lớn: volume, velocity, variety.
- Vì sao bài toán cần xử lý phân tán.
- Vai trò của Spark trong xử lý batch.
- Vai trò của HDFS trong lưu trữ phân tán.

## 2.2. Apache Spark

Trình bày:

- Spark là framework xử lý dữ liệu phân tán.
- Spark DataFrame, RDD, partition, executor, driver.
- Vì sao dùng Spark cho cleaning, LDA, sentiment inference.
- Các kỹ thuật trong repo: `mapPartitions`, `repartition`, `persist`, `checkpoint`, ghi Parquet.

File minh chứng:

- `spark_jobs/cleaning_job.py`
- `spark_jobs/lda_job.py`
- `spark_jobs/sentiment_job.py`
- `spark_jobs/crisis_detection.py`

## 2.3. HDFS và Data Lake

Trình bày:

- HDFS dùng để lưu raw data và staged Parquet.
- Phân tầng dữ liệu: raw/bronze, staged/silver, warehouse/gold.
- Ưu điểm của Parquet: columnar, nén tốt, phù hợp phân tích.

Các path chính:

```text
/user/root/raw_data
/user/root/staged/stg_posts_core
/user/root/staged/stg_posts_nlp
/user/root/staged/stg_post_topics
/user/root/staged/stg_topics
/user/root/staged/stg_keyword_freq
/user/root/staged/stg_crisis_events
```

## 2.4. Apache Airflow

Trình bày:

- Airflow dùng để định nghĩa DAG và điều phối pipeline.
- Các task chính trong `full_processing_pipeline`.
- Lợi ích: retry, scheduling, dependency management, theo dõi trạng thái task.

File minh chứng:

- `dags/processing_dag.py`
- `dags/weekly_retrain_dag.py`
- `dags/daily_processing_dag.py`

## 2.5. ClickHouse và dbt

Trình bày:

- ClickHouse là OLAP database, phù hợp truy vấn phân tích nhanh.
- dbt dùng để biến đổi dữ liệu từ staging sang intermediate/mart.
- Mô hình ELT: Spark ghi staging, dbt join/aggregate trong warehouse.

File minh chứng:

- `warehouse/clickhouse/init_schema.sql`
- `warehouse/dbt_project/models/staging/`
- `warehouse/dbt_project/models/intermediate/`
- `warehouse/dbt_project/models/marts/`

## 2.6. MinHash và Locality-Sensitive Hashing

Nội dung cần có:

- Bài toán near-duplicate detection.
- Jaccard similarity.
- Shingling văn bản.
- MinHash signature.
- LSH để tìm nhanh các cặp gần giống.
- Ứng dụng trong dự án: loại bài viết/comment trùng hoặc gần trùng.

File minh chứng:

- `algorithms/minhash_dedup.py`

## 2.7. Count-Min Sketch

Nội dung cần có:

- Bài toán đếm tần suất keyword với bộ nhớ hạn chế.
- Ý tưởng hash nhiều hàng/cột.
- Ước lượng tần suất và sai số.
- Ứng dụng trong dự án: keyword frequency theo window.

File minh chứng:

- `algorithms/count_min_sketch.py`

## 2.8. Topic Modeling: LDA và BERTopic

Nội dung cần có:

- LDA: mô hình chủ đề xác suất, document-topic distribution, topic-word distribution.
- BERTopic: embedding, UMAP, HDBSCAN, c-TF-IDF.
- Vì sao dùng cả LDA và BERTopic: LDA chạy phân tán bằng Spark, BERTopic cho chất lượng semantic tốt hơn.

File minh chứng:

- `spark_jobs/lda_job.py`
- `spark_jobs/bertopic_inference_job.py`
- `models/bertopic_model.py`

## 2.9. Sentiment Analysis với PhoBERT

Nội dung cần có:

- PhoBERT là mô hình ngôn ngữ tiếng Việt dựa trên RoBERTa.
- Bài toán phân loại cảm xúc 3 lớp: positive, neutral, negative.
- Cách tích hợp inference trong Spark bằng `mapPartitions`.

File minh chứng:

- `models/train_phobert.py`
- `models/sentiment_predictor.py`
- `spark_jobs/sentiment_job.py`

## 2.10. Phát hiện bất thường và khủng hoảng dư luận

Nội dung cần có:

- Các tín hiệu: lượng nhắc đến, tỷ lệ tiêu cực, velocity, acceleration, z-score.
- Isolation Forest và rule-based fallback.
- Cách tạo crisis event.

File minh chứng:

- `models/isolation_forest.py`
- `models/rolling_threshold.py`
- `spark_jobs/crisis_detection.py`
- `spark_jobs/weekly_retrain.py`

## 2.11. Dashboard Next.js

Nội dung cần có:

- Next.js dùng để xây dựng dashboard.
- Dashboard query ClickHouse marts, không đọc trực tiếp HDFS.
- Các trang chính: Overview, Trends Explorer, Crisis Monitor.

File minh chứng:

- `dashboard/src/app/page.tsx`
- `dashboard/src/app/trends/page.tsx`
- `dashboard/src/app/crises/page.tsx`
- `dashboard/src/lib/dal/radar.ts`

---

# Chương 3. Phân tích yêu cầu hệ thống

## 3.1. Yêu cầu chức năng

| Mã | Yêu cầu | Mô tả |
|---|---|---|
| F1 | Thu thập dữ liệu | Crawl dữ liệu từ các nguồn tiếng Việt như VOZ, VnExpress, VatVo |
| F2 | Upload raw data | Đưa dữ liệu crawler vào vùng raw trên HDFS |
| F3 | Làm sạch dữ liệu | Chuẩn hóa schema, làm sạch text, tạo `stg_posts_core` |
| F4 | Loại trùng dữ liệu | Loại bỏ bản ghi trùng/gần trùng bằng MinHash/LSH |
| F5 | Phân tích chủ đề | Gán topic cho bài viết bằng LDA/BERTopic |
| F6 | Đếm keyword | Ước lượng tần suất keyword bằng Count-Min Sketch |
| F7 | Phân tích cảm xúc | Gán nhãn positive/neutral/negative bằng PhoBERT |
| F8 | Phát hiện khủng hoảng | Tạo cảnh báo dựa trên sentiment và volume bất thường |
| F9 | Lưu warehouse | Load staging data vào ClickHouse và build marts bằng dbt |
| F10 | Dashboard | Hiển thị trend, sentiment, crisis và evidence posts |

## 3.2. Yêu cầu phi chức năng

| Nhóm yêu cầu | Mô tả |
|---|---|
| Khả năng mở rộng | Có thể xử lý dữ liệu lớn bằng Spark/HDFS |
| Tính tự động | Pipeline chạy qua Airflow DAG |
| Tính mô-đun | Các output NLP tách riêng theo bảng/dataset |
| Hiệu năng truy vấn | Dashboard đọc từ ClickHouse marts |
| Khả năng tái lập | Hạ tầng cấu hình bằng Docker/Ansible |
| Khả năng bảo trì | Tách module crawler, Spark jobs, dbt, dashboard |

## 3.3. Đối tượng sử dụng

- Người theo dõi xu hướng công nghệ.
- Nhóm marketing/social listening.
- Nhóm quản trị truyền thông.
- Người phân tích dữ liệu/NLP muốn khai thác dữ liệu mạng xã hội tiếng Việt.

## 3.4. Use case chính

| Use case | Actor | Mô tả |
|---|---|---|
| UC1 | Hệ thống | Tự động crawl và upload dữ liệu thô |
| UC2 | Hệ thống | Chạy Spark cleaning và dedup |
| UC3 | Hệ thống | Chạy topic modeling và sentiment analysis |
| UC4 | Hệ thống | Build warehouse marts |
| UC5 | Người dùng | Xem top trending topics |
| UC6 | Người dùng | Xem diễn biến sentiment theo topic |
| UC7 | Người dùng | Xem cảnh báo crisis và evidence posts |

---

# Chương 4. Thiết kế kiến trúc hệ thống

## 4.1. Kiến trúc tổng thể

Nên chèn sơ đồ tổng thể:

```text
Crawlers
  -> HDFS Raw
  -> Spark Processing
  -> HDFS Staged Parquet
  -> ClickHouse Staging
  -> dbt Intermediate/Marts
  -> Next.js Dashboard
```

Mô tả các layer:

- **Ingestion layer:** crawler thu thập dữ liệu.
- **Storage layer:** HDFS lưu raw/staged data.
- **Processing layer:** Spark xử lý dữ liệu và chạy NLP jobs.
- **Warehouse layer:** ClickHouse + dbt lưu và biến đổi dữ liệu phân tích.
- **Serving layer:** Dashboard query mart tables.
- **Orchestration layer:** Airflow quản lý toàn bộ DAG.

## 4.2. Kiến trúc hạ tầng

Mô tả:

- Docker Compose dùng cho môi trường local/development.
- Ansible dùng cho cài đặt/cấu hình cluster.
- Các service chính: namenode, datanode, spark-master, spark-worker, airflow-webserver, airflow-scheduler, clickhouse, dashboard.

File minh chứng:

- `docker-compose.yml`
- `Dockerfile.spark`
- `Dockerfile.airflow`
- `ansible/`

## 4.3. Kiến trúc pipeline dữ liệu

Trình bày luồng Airflow chính:

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

Nêu rõ:

- `stg_posts_core` là dataset trung tâm.
- Các kết quả NLP/topic/sentiment không ghi ngược vào `posts_core`.
- Việc join được thực hiện ở dbt/ClickHouse, giúp giảm Spark shuffle và cho phép retry độc lập.

## 4.4. Kiến trúc dữ liệu theo tầng

| Tầng | Nơi lưu | Dữ liệu |
|---|---|---|
| Raw/Bronze | HDFS raw | Dữ liệu crawler gốc |
| Silver | HDFS staged Parquet | Dữ liệu đã clean, topic, sentiment, keyword, crisis |
| Staging | ClickHouse staging | Các bảng `stg_*` |
| Gold | dbt marts | Bảng `dbt_fct_topic_activity`, `dbt_dim_topics`, `dbt_fct_crisis_events` |
| Serving | Dashboard | Query result phục vụ trực quan hóa |

---

# Chương 5. Thiết kế dữ liệu và luồng xử lý

## 5.1. Dữ liệu đầu vào

Mô tả các nguồn:

- VOZ: bài viết và bình luận diễn đàn.
- VnExpress: bài viết và bình luận tin tức.
- VatVo: bài viết/chủ đề công nghệ.

File minh chứng:

- `crawlers/voz.py`
- `crawlers/vnexpress.py`
- `crawlers/vatvo.py`
- `crawlers/upload_to_hdfs.py`

## 5.2. Schema chuẩn hóa `stg_posts_core`

Trình bày bảng:

| Cột | Ý nghĩa |
|---|---|
| `post_id` | Định danh bài viết/comment |
| `source` | Nguồn dữ liệu |
| `author_id` / `author_name` | Thông tin tác giả |
| `title` | Tiêu đề |
| `body` | Nội dung đã bỏ HTML cơ bản |
| `segmented_text` | Text đã tách từ |
| `parent_id` | Bài viết cha nếu là comment |
| `reaction_count` | Số reaction |
| `comment_count` | Số comment |
| `view_count` | Số lượt xem |
| `created_at` | Thời gian tạo |
| `crawled_at` | Thời gian crawl |

Ghi chú: trong `spark_jobs/cleaning_job.py` có thêm các cột trung gian như `clean_text`, `topic_text` để phục vụ xử lý/NLP.

## 5.3. Các bảng ClickHouse staging

Theo `warehouse/clickhouse/init_schema.sql`:

| Bảng | Nguồn sinh dữ liệu | Vai trò |
|---|---|---|
| `stg_posts_core` | M2 Spark cleaning | Dữ liệu lõi đã clean/dedup |
| `stg_posts_nlp` | M4 Sentiment job | Nhãn cảm xúc |
| `stg_post_topics` | M3 LDA/BERTopic | Gán topic cho post |
| `stg_topics` | M3 topic model | Từ khóa và nhãn topic |
| `stg_keyword_freq` | M3 Count-Min Sketch | Tần suất keyword theo window |
| `stg_crisis_events` | M4 crisis detection | Sự kiện khủng hoảng |

## 5.4. dbt intermediate và mart

Các model chính:

- `int_posts_enriched`: join `stg_posts_core`, `stg_posts_nlp`, `stg_post_topics`.
- `int_topic_sentiment_hourly`: tổng hợp sentiment/topic theo giờ.
- `fct_topic_activity`: fact table chính cho trend.
- `dim_topics`: dimension topic.
- `fct_crisis_events`: mart cho crisis monitor.

File minh chứng:

- `warehouse/dbt_project/models/intermediate/int_posts_enriched.sql`
- `warehouse/dbt_project/models/intermediate/int_topic_sentiment_hourly.sql`
- `warehouse/dbt_project/models/marts/fct_topic_activity.sql`
- `warehouse/dbt_project/models/marts/dim_topics.sql`
- `warehouse/dbt_project/models/marts/fct_crisis_events.sql`

---

# Chương 6. Xây dựng các module chính

## 6.1. Module crawling và upload HDFS

Nội dung:

- Crawler thu thập dữ liệu từ web.
- Lưu dữ liệu thành CSV/local data.
- Upload dữ liệu lên HDFS raw zone.
- Có checkpoint/progress để tránh crawl trùng.

File minh chứng:

- `crawlers/voz.py`
- `crawlers/vnexpress.py`
- `crawlers/vatvo.py`
- `crawlers/upload_to_hdfs.py`

## 6.2. Module chuẩn hóa schema

Nội dung:

- Dùng adapters để chuyển dữ liệu từng nguồn về DataFrame chuẩn.
- `UniversalSocialPost` đóng vai trò schema chung ở tầng logic.

File minh chứng:

- `schemas/models.py`
- `schemas/voz_adapter.py`
- `schemas/vnexpress_adapter.py`
- `schemas/vatvo_adapter.py`

## 6.3. Module Spark cleaning và preprocessing

Nội dung:

- Đọc raw data từ HDFS/local.
- Chuẩn hóa field từ nhiều nguồn.
- Làm sạch text.
- Tách từ tiếng Việt.
- Tạo `stg_posts_core`.
- Ghi Parquet partition theo `source`.

File minh chứng:

- `spark_jobs/cleaning_job.py`
- `preprocessing/text_cleaner.py`
- `preprocessing/slang_normalizer.py`
- `preprocessing/vncorenlp_tokenizer.py`

## 6.4. Module MinHash/LSH dedup

Nội dung:

- Shingling text.
- Tạo MinHash signature.
- Dùng LSH để tìm near-duplicate.
- Gom nhóm duplicate bằng union-find.
- Giữ một record đại diện.

File minh chứng:

- `algorithms/minhash_dedup.py`

## 6.5. Module topic modeling

Nội dung:

- LDA chạy trên Spark MLlib.
- BERTopic dùng embedding/PhoBERT và clustering.
- Output gồm `stg_post_topics` và `stg_topics`.

File minh chứng:

- `spark_jobs/lda_job.py`
- `spark_jobs/bertopic_inference_job.py`
- `models/bertopic_model.py`

## 6.6. Module Count-Min Sketch

Nội dung:

- Dùng Count-Min Sketch để đếm keyword frequency với bộ nhớ hạn chế.
- Tích hợp vào Airflow task `cms_keyword_counting`.
- Output là `stg_keyword_freq`.

File minh chứng:

- `algorithms/count_min_sketch.py`
- `dags/processing_dag.py`

## 6.7. Module sentiment analysis

Nội dung:

- Fine-tune hoặc sử dụng PhoBERT cho phân loại cảm xúc.
- Chạy inference theo partition trong Spark.
- Output là `stg_posts_nlp`.

File minh chứng:

- `models/train_phobert.py`
- `models/sentiment_predictor.py`
- `spark_jobs/sentiment_job.py`

## 6.8. Module crisis detection

Nội dung:

- Tạo feature theo thời gian: comment count, unique users, negative ratio, velocity, z-score.
- Dùng Isolation Forest hoặc fallback rule-based.
- Tạo event nếu có dấu hiệu bất thường.

File minh chứng:

- `spark_jobs/crisis_detection.py`
- `models/isolation_forest.py`
- `models/rolling_threshold.py`
- `spark_jobs/weekly_retrain.py`

## 6.9. Module warehouse và transformation

Nội dung:

- Ingest staged Parquet vào ClickHouse.
- dbt build intermediate/mart.
- Tính engagement, velocity, acceleration, trend score.

File minh chứng:

- `scripts/hdfs_to_clickhouse.py`
- `warehouse/dbt_project/macros/trend_score_calc.sql`
- `warehouse/dbt_project/models/`

## 6.10. Module dashboard

Nội dung:

- Dashboard Next.js query ClickHouse.
- Trang Overview: KPI, top trending topics, sentiment.
- Trang Trends Explorer: trend score, sentiment timeline, word cloud, evidence posts.
- Trang Crisis Monitor: crisis timeline, incident detail, severity.

File minh chứng:

- `dashboard/src/app/page.tsx`
- `dashboard/src/app/trends/page.tsx`
- `dashboard/src/app/crises/page.tsx`
- `dashboard/src/lib/dal/radar.ts`

---

# Chương 7. Điều phối pipeline và triển khai hệ thống

## 7.1. Airflow DAG chính

Trình bày DAG `full_processing_pipeline` trong `dags/processing_dag.py`.

Nội dung cần viết:

- Schedule daily.
- Kiểm tra mount/input.
- Upload raw/reference files.
- Chạy Spark cleaning.
- Ingest vào ClickHouse.
- Chạy topic/sentiment/CMS/crisis.
- Chạy dbt transform.
- Kết thúc pipeline.

## 7.2. Ingest contract HDFS sang ClickHouse

Trình bày:

- File `scripts/hdfs_to_clickhouse.py` định nghĩa `TABLE_SPECS`.
- Mỗi dataset có table, mode, path, column list và select/cast expression riêng.
- Không dùng `SELECT *`, tránh lỗi lệch schema.

Các ingest mode:

| Dataset | Mode |
|---|---|
| `stg_posts_core` | full_refresh |
| `stg_posts_nlp` | replace_latest |
| `stg_post_topics` | replace_latest |
| `stg_topics` | replace_latest |
| `stg_keyword_freq` | append |
| `stg_crisis_events` | append, allow_empty |

## 7.3. Triển khai bằng Docker Compose

Nội dung:

- Cách khởi động hệ thống local bằng `docker-compose up -d`.
- Các service và port quan trọng:

| Service | Port |
|---|---|
| Airflow UI | `8081` |
| Spark UI | `8080` |
| ClickHouse HTTP | `8123` |
| Dashboard Next.js | `3000` |
| HDFS Web UI | `9870` |

## 7.4. Triển khai bằng Ansible

Nội dung:

- Inventory phân vai node.
- Playbook cài Java, Conda, Hadoop, Spark, ClickHouse, dbt, Airflow.
- Lợi ích: tự động hóa, giảm sai khác môi trường, dễ tái lập.

## 7.5. Quy trình chạy demo

Đề xuất trình tự:

1. Khởi động Docker Compose.
2. Mở Airflow UI.
3. Trigger DAG `full_processing_pipeline`.
4. Kiểm tra staged Parquet trên HDFS.
5. Kiểm tra ClickHouse staging tables.
6. Chạy/kiểm tra dbt marts.
7. Mở dashboard.

---

# Chương 8. Kết quả thực nghiệm và đánh giá

## 8.1. Môi trường thực nghiệm

Điền thông tin:

| Thành phần | Cấu hình |
|---|---|
| CPU | [Điền] |
| RAM | [Điền] |
| OS | [Điền] |
| Docker | [Điền version] |
| Spark | [Điền version] |
| ClickHouse | [Điền version] |
| Node.js | [Điền version] |

## 8.2. Thống kê dữ liệu

Điền số liệu:

| Nguồn | Số post | Số comment | Tổng record |
|---|---:|---:|---:|
| VOZ | [ ] | [ ] | [ ] |
| VnExpress | [ ] | [ ] | [ ] |
| VatVo | [ ] | [ ] | [ ] |
| Tổng | [ ] | [ ] | [ ] |

## 8.3. Kết quả Spark cleaning

Các số liệu nên có:

- Số record raw đầu vào.
- Số record sau filter text.
- Số record sau exact dedup.
- Số record sau MinHash/LSH dedup.
- Thời gian chạy job.

| Metric | Giá trị |
|---|---:|
| Raw records | [ ] |
| Sau filter | [ ] |
| Sau exact dedup | [ ] |
| Sau MinHash/LSH | [ ] |
| Tỷ lệ loại bỏ | [ ] |
| Thời gian chạy | [ ] |

## 8.4. Kết quả topic modeling

Điền:

- Số topic.
- Top keywords một số topic.
- Coherence score nếu có.
- So sánh LDA và BERTopic nếu có.

| Model | Số topic | Coherence | Ghi chú |
|---|---:|---:|---|
| LDA | [ ] | [ ] | [ ] |
| BERTopic | [ ] | [ ] | [ ] |

## 8.5. Kết quả Count-Min Sketch

Điền:

- Top keywords.
- Window thời gian.
- So sánh với exact count nếu có.
- Sai số ước lượng nếu có.

| Keyword | Estimated count | Exact count | Error |
|---|---:|---:|---:|
| [ ] | [ ] | [ ] | [ ] |

## 8.6. Kết quả sentiment analysis

Điền:

- Accuracy.
- Macro F1.
- Confusion matrix.
- Phân bố positive/neutral/negative.

| Metric | Giá trị |
|---|---:|
| Accuracy | [ ] |
| Macro F1 | [ ] |
| Precision | [ ] |
| Recall | [ ] |

## 8.7. Kết quả crisis detection

Điền:

- Số event phát hiện.
- Severity distribution.
- Ví dụ một crisis event.
- Các trigger conditions.

| Event ID | Severity | Neg ratio | Mention velocity | Trigger |
|---|---|---:|---:|---|
| [ ] | [ ] | [ ] | [ ] | [ ] |

## 8.8. Kết quả warehouse và dashboard

Điền:

- Số dòng trong các bảng staging/mart.
- Ảnh dashboard.
- Thời gian query trung bình nếu có.

| Bảng | Số dòng |
|---|---:|
| `stg_posts_core` | [ ] |
| `stg_posts_nlp` | [ ] |
| `stg_post_topics` | [ ] |
| `dbt_fct_topic_activity` | [ ] |
| `dbt_dim_topics` | [ ] |
| `dbt_fct_crisis_events` | [ ] |

## 8.9. Benchmark hiệu năng

Các benchmark nên có:

- Thời gian chạy Spark cleaning với các số lượng record khác nhau.
- So sánh local/single node và distributed nếu có.
- Thời gian ingest ClickHouse.
- Thời gian dbt transform.
- Thời gian query dashboard.

| Thử nghiệm | Dữ liệu | Cấu hình | Thời gian |
|---|---:|---|---:|
| Spark cleaning | [ ] | [ ] | [ ] |
| LDA | [ ] | [ ] | [ ] |
| Sentiment inference | [ ] | [ ] | [ ] |
| dbt transform | [ ] | [ ] | [ ] |

## 8.10. Đánh giá chung

Nêu:

- Hệ thống đã chạy được end-to-end.
- Dữ liệu được chuẩn hóa thành các bảng rõ ràng.
- Dashboard đọc từ mart nên truy vấn nhanh.
- Mô hình và thuật toán đã tích hợp được vào pipeline.
- Một số hạn chế còn tồn tại.

---

# Chương 9. Phân công công việc trong nhóm

## 9.1. Tổng quan phân công

| Thành viên | Vai trò | Phần việc chính |
|---|---|---|
| M1 | Data Engineer | Crawlers, raw data, upload HDFS |
| M2 | DevOps / Data Infrastructure & Spark Processing | Docker, Ansible, Spark cleaning, MinHash/LSH, Airflow integration |
| M3 | ML Engineer | LDA, BERTopic, Count-Min Sketch |
| M4 | NLP Engineer | PhoBERT sentiment, crisis detection |
| M5 | Full-stack / Analytics | ClickHouse/dbt marts, dashboard |

## 9.2. Chi tiết phần việc M1

Nội dung:

- Viết crawler VOZ/VnExpress/VatVo.
- Lưu dữ liệu raw.
- Upload dữ liệu lên HDFS.
- Hỗ trợ chuẩn hóa schema đầu vào.

## 9.3. Chi tiết phần việc M2

Nội dung:

- Dựng infrastructure bằng Docker/Ansible.
- Cấu hình Spark, HDFS, Airflow, ClickHouse.
- Xây dựng `spark_jobs/cleaning_job.py`.
- Tích hợp MinHash/LSH trong `algorithms/minhash_dedup.py`.
- Tích hợp task Airflow: `validate_runtime_mounts`, `upload_reference_files_to_hdfs`, `spark_cleaning`, `ingest_stg_posts_core`.

## 9.4. Chi tiết phần việc M3

Nội dung:

- Xây dựng LDA pipeline bằng Spark MLlib.
- Xây dựng BERTopic model/inference.
- Triển khai Count-Min Sketch.
- Sinh `stg_post_topics`, `stg_topics`, `stg_keyword_freq`.

## 9.5. Chi tiết phần việc M4

Nội dung:

- Xây dựng preprocessing/NLP utilities.
- Fine-tune hoặc tích hợp PhoBERT sentiment.
- Xây dựng Spark sentiment job.
- Xây dựng crisis detection và weekly retrain.

## 9.6. Chi tiết phần việc M5

Nội dung:

- Thiết kế ClickHouse/dbt marts.
- Tính trend score.
- Xây dựng dashboard Next.js.
- Kết nối dashboard với ClickHouse.

---

# Chương 10. Kết luận và hướng phát triển

## 10.1. Kết luận

Nội dung:

- Đồ án đã xây dựng được một pipeline phân tích xu hướng tiếng Việt end-to-end.
- Hệ thống kết hợp Big Data, NLP, data warehouse và dashboard.
- Các thuật toán như MinHash/LSH, Count-Min Sketch, LDA/BERTopic, PhoBERT và Isolation Forest đã được tích hợp vào pipeline.
- Dashboard giúp người dùng theo dõi chủ đề nổi bật, cảm xúc và khủng hoảng.

## 10.2. Hạn chế

Nêu trung thực:

- Dữ liệu còn phụ thuộc vào nguồn crawler và chất lượng HTML từng trang.
- Một số job vẫn chạy batch, chưa phải real-time streaming.
- MinHash/LSH hiện tại có phần xử lý gom về driver, cần tối ưu nếu dữ liệu rất lớn.
- BERTopic có thể cần GPU và tài nguyên lớn.
- Crisis detection cần thêm dữ liệu gán nhãn để đánh giá chính xác hơn.
- Dashboard phụ thuộc vào chất lượng mart tables và freshness của pipeline.

## 10.3. Hướng phát triển

Gợi ý:

- Bổ sung thêm nguồn dữ liệu như YouTube, Facebook public pages, Tinhte.
- Chuyển một số pipeline sang streaming bằng Kafka/Spark Structured Streaming.
- Tối ưu MinHash/LSH theo hướng phân tán hoàn toàn.
- Tăng chất lượng sentiment model bằng dữ liệu gán nhãn nhiều hơn.
- Tích hợp alert realtime qua Telegram/Email/Slack.
- Thêm phân tích entity/brand/product.
- Triển khai production trên Kubernetes hoặc cloud.

---

# Tài liệu tham khảo

Gợi ý danh sách:

1. Apache Spark Documentation.
2. Apache Airflow Documentation.
3. ClickHouse Documentation.
4. dbt Documentation.
5. PhoBERT paper / VinAI PhoBERT repository.
6. BERTopic Documentation.
7. scikit-learn Documentation.
8. Mining of Massive Datasets - Jure Leskovec, Anand Rajaraman, Jeff Ullman.
9. Tài liệu về MinHash, Locality-Sensitive Hashing.
10. Tài liệu về Count-Min Sketch.

---

# Phụ lục

## Phụ lục A. Cấu trúc thư mục chính

```text
Distributed-NLP-Trend-Analysis/
├── ansible/
├── crawlers/
├── dags/
├── dashboard/
├── data/
├── docs/
├── models/
├── preprocessing/
├── schemas/
├── scripts/
├── spark_jobs/
├── algorithms/
├── warehouse/
├── docker-compose.yml
├── Dockerfile.airflow
├── Dockerfile.spark
└── requirements.txt
```

## Phụ lục B. Các file code quan trọng

| Nhóm | File |
|---|---|
| Crawler | `crawlers/voz.py`, `crawlers/vnexpress.py`, `crawlers/vatvo.py` |
| Schema | `schemas/models.py`, `schemas/*_adapter.py` |
| Spark cleaning | `spark_jobs/cleaning_job.py` |
| Dedup | `algorithms/minhash_dedup.py` |
| LDA | `spark_jobs/lda_job.py` |
| BERTopic | `models/bertopic_model.py`, `spark_jobs/bertopic_inference_job.py` |
| Count-Min Sketch | `algorithms/count_min_sketch.py` |
| Sentiment | `models/sentiment_predictor.py`, `spark_jobs/sentiment_job.py` |
| Crisis | `spark_jobs/crisis_detection.py`, `models/isolation_forest.py` |
| Airflow | `dags/processing_dag.py` |
| ClickHouse ingest | `scripts/hdfs_to_clickhouse.py` |
| dbt | `warehouse/dbt_project/models/` |
| Dashboard | `dashboard/src/` |

## Phụ lục C. Các câu lệnh kiểm tra demo

Khởi động hệ thống:

```bash
docker-compose up -d
```

Mở Airflow:

```text
http://localhost:8081
```

Mở dashboard:

```text
http://localhost:3000
```

Kiểm tra ClickHouse:

```sql
SHOW TABLES FROM tech_radar;
SELECT count() FROM tech_radar.stg_posts_core;
SELECT count() FROM tech_radar.dbt_fct_topic_activity;
```

Kiểm tra dashboard health:

```text
http://localhost:3000/api/health/clickhouse
```

## Phụ lục D. Danh sách hình/bảng nên đưa vào báo cáo

Hình nên có:

- Sơ đồ kiến trúc tổng thể.
- Sơ đồ Airflow DAG.
- Sơ đồ data flow từ raw đến dashboard.
- Ảnh Spark UI khi chạy job.
- Ảnh ClickHouse tables.
- Ảnh dbt mart tables.
- Ảnh dashboard Overview.
- Ảnh dashboard Trends Explorer.
- Ảnh dashboard Crisis Monitor.

Bảng nên có:

- Bảng công nghệ sử dụng.
- Bảng yêu cầu chức năng/phi chức năng.
- Bảng schema `stg_posts_core`.
- Bảng ClickHouse staging tables.
- Bảng dbt marts.
- Bảng kết quả thực nghiệm.
- Bảng benchmark hiệu năng.
- Bảng phân công công việc.

