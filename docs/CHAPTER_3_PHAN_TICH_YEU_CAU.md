# Chương 3. Phân tích yêu cầu hệ thống

## 3.1. Tổng quan bài toán

Đồ án xây dựng hệ thống phân tích xu hướng và dư luận mạng xã hội tiếng Việt trong lĩnh vực công nghệ. Hệ thống có nhiệm vụ thu thập dữ liệu từ nhiều nguồn, xử lý dữ liệu ở quy mô lớn, phân tích chủ đề và cảm xúc, sau đó trực quan hóa kết quả trên dashboard.

Bài toán có thể được mô tả ngắn gọn như sau:

> Từ dữ liệu bài viết và bình luận tiếng Việt trên các nền tảng công nghệ, hệ thống cần tự động phát hiện người dùng đang bàn luận về chủ đề gì, cảm xúc của họ ra sao, chủ đề nào đang tăng mạnh và có xuất hiện dấu hiệu khủng hoảng dư luận hay không.

Để giải quyết bài toán này, hệ thống cần thực hiện một pipeline end-to-end:

```text
Thu thập dữ liệu
-> Lưu trữ dữ liệu thô
-> Làm sạch và chuẩn hóa dữ liệu
-> Loại bỏ dữ liệu trùng/gần trùng
-> Phân tích chủ đề
-> Phân tích cảm xúc
-> Đếm keyword nổi bật
-> Phát hiện bất thường/khủng hoảng
-> Lưu vào data warehouse
-> Hiển thị dashboard
```

Đặc điểm quan trọng của bài toán là dữ liệu đầu vào không đồng nhất. Mỗi nguồn dữ liệu có cấu trúc riêng, ví dụ bài viết từ VOZ khác với bài viết VnExpress, comment cũng khác với article. Ngoài ra, văn bản tiếng Việt trên mạng xã hội thường chứa nhiều nhiễu như teencode, emoji, HTML, URL, viết tắt, sai chính tả hoặc nội dung quá ngắn. Vì vậy, hệ thống phải có bước chuẩn hóa và làm sạch dữ liệu trước khi chạy các mô hình phân tích.

## 3.2. Mục tiêu hệ thống

Hệ thống được thiết kế với các mục tiêu chính sau:

1. **Tự động thu thập dữ liệu**

   Hệ thống cần có khả năng thu thập dữ liệu từ các nguồn tiếng Việt như diễn đàn, báo điện tử hoặc các trang công nghệ. Dữ liệu thu thập gồm bài viết, bình luận, tiêu đề, tác giả, thời gian đăng và các chỉ số tương tác nếu có.

2. **Xử lý dữ liệu phân tán**

   Dữ liệu sau khi thu thập cần được xử lý bằng Spark để có thể mở rộng khi số lượng bản ghi tăng. Các bước xử lý gồm đọc dữ liệu thô, chuẩn hóa schema, làm sạch văn bản, tách từ, lọc dữ liệu lỗi và ghi dữ liệu dạng Parquet.

3. **Loại bỏ dữ liệu trùng lặp**

   Hệ thống cần loại bỏ các bản ghi trùng hoặc gần trùng để tránh làm sai lệch kết quả phân tích. Bên cạnh `dropDuplicates()` cho các dòng giống hệt nhau, hệ thống sử dụng MinHash/LSH để phát hiện near-duplicate.

4. **Phân tích chủ đề**

   Hệ thống cần gán topic cho bài viết/bình luận để xác định người dùng đang nói về những chủ đề nào. Các phương pháp topic modeling được sử dụng gồm LDA và BERTopic.

5. **Phân tích cảm xúc**

   Hệ thống cần xác định cảm xúc của nội dung văn bản theo ba lớp: tích cực, trung lập và tiêu cực. Kết quả sentiment được dùng để phân tích dư luận và hỗ trợ phát hiện khủng hoảng.

6. **Theo dõi keyword nổi bật**

   Hệ thống cần đếm tần suất keyword theo từng nguồn và khung thời gian. Count-Min Sketch được sử dụng để ước lượng tần suất keyword với bộ nhớ giới hạn.

7. **Phát hiện khủng hoảng dư luận**

   Hệ thống cần phát hiện các dấu hiệu bất thường như lượng thảo luận tăng mạnh, tỷ lệ tiêu cực cao hoặc các topic có biến động bất thường. Kết quả được lưu thành crisis events.

8. **Xây dựng kho dữ liệu phân tích**

   Dữ liệu sau xử lý cần được lưu vào ClickHouse, sau đó được dbt biến đổi thành các bảng mart phục vụ dashboard.

9. **Trực quan hóa kết quả**

   Dashboard cần hiển thị các thông tin chính như tổng số mentions, top trending topics, sentiment breakdown, trend score timeline, keywords, evidence posts và crisis events.

## 3.3. Đối tượng sử dụng hệ thống

Hệ thống hướng tới các nhóm người dùng sau:

| Đối tượng | Nhu cầu |
|---|---|
| Người phân tích dữ liệu | Theo dõi xu hướng, chủ đề nổi bật và biến động dữ liệu |
| Nhóm marketing/social listening | Theo dõi dư luận về sản phẩm, thương hiệu hoặc sự kiện công nghệ |
| Nhóm quản trị truyền thông | Phát hiện sớm khủng hoảng hoặc làn sóng phản hồi tiêu cực |
| Người quan tâm công nghệ | Xem chủ đề công nghệ nào đang được bàn luận nhiều |
| Nhóm phát triển hệ thống | Theo dõi pipeline, dữ liệu và trạng thái xử lý |

Trong phạm vi đồ án, người dùng chính của dashboard là người cần quan sát kết quả phân tích, không phải người trực tiếp vận hành pipeline. Việc vận hành pipeline chủ yếu do nhóm phát triển thực hiện thông qua Airflow, Docker và các công cụ kiểm tra dữ liệu.

## 3.4. Yêu cầu chức năng

Yêu cầu chức năng mô tả các chức năng hệ thống cần thực hiện.

| Mã yêu cầu | Chức năng | Mô tả |
|---|---|---|
| F1 | Thu thập dữ liệu | Hệ thống thu thập bài viết/bình luận từ các nguồn như VOZ, VnExpress, VatVo |
| F2 | Lưu dữ liệu thô | Dữ liệu sau crawl được lưu vào local/HDFS raw zone để làm input cho Spark |
| F3 | Kiểm tra dữ liệu đầu vào | Pipeline kiểm tra dữ liệu raw, reference files và model path trước khi chạy |
| F4 | Upload reference files | Stopwords, slang dictionary và các file cần thiết được upload lên HDFS để Spark executor sử dụng |
| F5 | Chuẩn hóa schema | Dữ liệu từ nhiều nguồn được map về schema chung `stg_posts_core` |
| F6 | Làm sạch văn bản | Hệ thống loại HTML, URL, emoji, ký tự nhiễu, chuẩn hóa Unicode, slang và khoảng trắng |
| F7 | Tách từ tiếng Việt | Văn bản được tách từ bằng tokenizer như underthesea hoặc VnCoreNLP |
| F8 | Tạo các cột text | Hệ thống tạo `body`, `clean_text`, `segmented_text`, `topic_text` phục vụ downstream tasks |
| F9 | Loại bỏ dữ liệu trùng | Hệ thống loại dòng trùng chính xác và gần trùng bằng MinHash/LSH |
| F10 | Ghi dữ liệu staged | Spark ghi output ra Parquet, ví dụ `stg_posts_core` |
| F11 | Ingest vào ClickHouse | Các staged datasets được load vào ClickHouse staging tables |
| F12 | Phân tích chủ đề | Hệ thống chạy LDA/BERTopic để gán topic cho post |
| F13 | Lưu topic assignment | Kết quả topic được lưu vào `stg_post_topics` và `stg_topics` |
| F14 | Phân tích cảm xúc | Hệ thống chạy PhoBERT để gán nhãn positive/neutral/negative |
| F15 | Lưu sentiment | Kết quả sentiment được lưu vào `stg_posts_nlp` |
| F16 | Đếm keyword | Hệ thống sử dụng Count-Min Sketch để ước lượng keyword frequency |
| F17 | Lưu keyword frequency | Kết quả keyword được lưu vào `stg_keyword_freq` |
| F18 | Phát hiện crisis | Hệ thống phát hiện các event bất thường dựa trên sentiment, velocity và volume |
| F19 | Lưu crisis events | Crisis events được lưu vào `stg_crisis_events` |
| F20 | Build mart tables | dbt tạo các bảng intermediate/mart từ ClickHouse staging |
| F21 | Tính trend score | Hệ thống tính trend score dựa trên velocity, acceleration và engagement |
| F22 | Hiển thị dashboard Overview | Dashboard hiển thị KPI, top trending topics và sentiment tổng quan |
| F23 | Hiển thị Trends Explorer | Dashboard hiển thị trend score, sentiment timeline, keywords và evidence posts theo topic |
| F24 | Hiển thị Crisis Monitor | Dashboard hiển thị danh sách crisis events, severity và trigger conditions |
| F25 | Health check dashboard | Hệ thống có endpoint kiểm tra kết nối ClickHouse |

## 3.5. Yêu cầu phi chức năng

Yêu cầu phi chức năng mô tả các tiêu chí về hiệu năng, khả năng mở rộng, bảo trì và vận hành.

| Nhóm yêu cầu | Mô tả |
|---|---|
| Khả năng mở rộng | Hệ thống cần xử lý được dữ liệu lớn bằng Spark và HDFS, có thể tăng worker khi cần |
| Tính tự động | Pipeline cần được điều phối bằng Airflow, hạn chế chạy thủ công từng bước |
| Tính mô-đun | Các module crawler, cleaning, topic, sentiment, crisis, dbt và dashboard được tách riêng |
| Tính tái lập | Môi trường có thể dựng lại bằng Docker Compose hoặc Ansible |
| Tính ổn định | Pipeline có bước preflight để fail sớm khi thiếu input hoặc mount sai |
| Tính nhất quán dữ liệu | Các bảng staging có schema rõ ràng, ingest không dùng `SELECT *` để tránh lệch schema |
| Hiệu năng truy vấn | Dashboard đọc từ ClickHouse mart tables thay vì raw data |
| Dễ debug | Dữ liệu được chia thành raw, staged, staging và mart để dễ kiểm tra từng tầng |
| Khả năng mở rộng nguồn | Có thể bổ sung crawler mới nếu map được về schema chung |
| Khả năng bảo trì | Logic nghiệp vụ như trend score được đặt ở dbt để thay đổi mà không cần rerun Spark |

## 3.6. Yêu cầu dữ liệu đầu vào

Dữ liệu đầu vào là bài viết hoặc bình luận tiếng Việt từ các nguồn công nghệ. Trong repo hiện tại, các nguồn chính gồm:

- VOZ.
- VnExpress.
- VatVo.

Dữ liệu đầu vào có thể bao gồm các trường:

| Nhóm thông tin | Ví dụ trường |
|---|---|
| Định danh | `post_id`, `comment_id`, `id_post` |
| Nội dung | `title`, `body`, `content`, `comment` |
| Tác giả | `author`, `author_name`, `user` |
| Thời gian | `created_at`, `comment_time`, `published_at` |
| Quan hệ cha-con | `parent_id`, `id_post` |
| Tương tác | `reaction_count`, `comment_count`, `view_count` |
| Nguồn | `source` |

Do mỗi crawler sinh dữ liệu có format khác nhau, hệ thống cần adapter để chuyển dữ liệu về format chuẩn. Các adapter hiện có:

- `schemas/voz_adapter.py`
- `schemas/vnexpress_adapter.py`
- `schemas/vatvo_adapter.py`

## 3.7. Yêu cầu dữ liệu đầu ra

Hệ thống tạo ra nhiều lớp dữ liệu đầu ra.

### 3.7.1. Output của Spark cleaning

Output quan trọng nhất của tầng cleaning là `stg_posts_core`.

Các cột chính:

```text
post_id
source
author
title
body
clean_text
segmented_text
topic_text
parent_id
reaction_count
view_count
comment_count
created_at
crawled_at
```

Dataset này là dữ liệu lõi cho các module phía sau.

### 3.7.2. Output của sentiment analysis

Dataset:

```text
stg_posts_nlp
```

Các cột chính:

```text
post_id
sentiment_label
sentiment_score
model_version
predicted_at
```

### 3.7.3. Output của topic modeling

Datasets:

```text
stg_post_topics
stg_topics
```

`stg_post_topics` lưu topic assignment cho từng post:

```text
post_id
topic_id
topic_probability
model_type
predicted_at
```

`stg_topics` lưu thông tin topic:

```text
topic_id
label
top_keywords
coherence_score
model_version
created_at
```

### 3.7.4. Output của Count-Min Sketch

Dataset:

```text
stg_keyword_freq
```

Các cột chính:

```text
keyword
window_start
window_end
estimated_count
source
```

### 3.7.5. Output của crisis detection

Dataset:

```text
stg_crisis_events
```

Các cột chính:

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

### 3.7.6. Output phục vụ dashboard

Các bảng mart chính:

```text
dbt_fct_topic_activity
dbt_dim_topics
dbt_fct_crisis_events
dbt_int_posts_enriched
```

Dashboard đọc các bảng này để hiển thị:

- KPI tổng quan.
- Top trending topics.
- Sentiment breakdown.
- Topic trend score timeline.
- Topic sentiment timeline.
- Keywords.
- Evidence posts.
- Crisis events.

## 3.8. Tác nhân và use case hệ thống

### 3.8.1. Tác nhân

| Tác nhân | Vai trò |
|---|---|
| Người dùng dashboard | Xem kết quả phân tích xu hướng, sentiment và crisis |
| Người vận hành pipeline | Trigger DAG, kiểm tra trạng thái task, debug lỗi |
| Crawler | Thu thập dữ liệu từ web |
| Airflow | Điều phối các bước pipeline |
| Spark | Xử lý dữ liệu phân tán |
| ClickHouse/dbt | Lưu trữ và biến đổi dữ liệu phân tích |

### 3.8.2. Use case tổng quan

| Mã use case | Tên use case | Tác nhân chính | Mô tả |
|---|---|---|---|
| UC1 | Thu thập dữ liệu | Crawler | Crawler lấy dữ liệu từ các nguồn web |
| UC2 | Upload raw data | Airflow/Crawler | Dữ liệu raw được upload vào HDFS |
| UC3 | Làm sạch dữ liệu | Spark | Spark chuẩn hóa dữ liệu và tạo `stg_posts_core` |
| UC4 | Loại trùng dữ liệu | Spark | MinHash/LSH loại bỏ near-duplicate |
| UC5 | Phân tích chủ đề | Spark/ML module | LDA/BERTopic gán topic cho post |
| UC6 | Phân tích cảm xúc | Spark/NLP module | PhoBERT gán sentiment label |
| UC7 | Đếm keyword | CMS module | Count-Min Sketch ước lượng keyword frequency |
| UC8 | Phát hiện crisis | Crisis module | Phát hiện các event bất thường |
| UC9 | Build warehouse | dbt | Tạo intermediate/mart tables |
| UC10 | Xem dashboard | Người dùng | Người dùng xem trend, sentiment và crisis |
| UC11 | Kiểm tra hệ thống | Người vận hành | Kiểm tra Airflow, HDFS, ClickHouse và dashboard health |

## 3.9. Mô tả use case chính

### 3.9.1. UC3 - Làm sạch dữ liệu

| Thành phần | Mô tả |
|---|---|
| Tác nhân | Airflow, Spark |
| Đầu vào | Raw data từ HDFS/local |
| Xử lý | Đọc dữ liệu, chuẩn hóa schema, làm sạch text, tách từ, lọc record lỗi |
| Đầu ra | Parquet dataset `stg_posts_core` |
| File liên quan | `spark_jobs/cleaning_job.py` |

Luồng chính:

1. Airflow gọi task `spark_cleaning`.
2. Task chạy `spark-submit` file `cleaning_job.py`.
3. Spark đọc raw data từ các nguồn.
4. Adapter chuẩn hóa dữ liệu về schema chung.
5. Text được làm sạch và tách từ.
6. Dữ liệu lỗi/rỗng bị loại bỏ.
7. Output được ghi ra Parquet.

### 3.9.2. UC4 - Loại trùng dữ liệu

| Thành phần | Mô tả |
|---|---|
| Tác nhân | Spark |
| Đầu vào | DataFrame sau cleaning |
| Xử lý | Exact dedup và near-duplicate dedup bằng MinHash/LSH |
| Đầu ra | DataFrame đã loại trùng |
| File liên quan | `algorithms/minhash_dedup.py` |

Luồng chính:

1. Dữ liệu được loại trùng chính xác bằng `dropDuplicates()`.
2. Với near-duplicate, hệ thống tạo shingles từ `segmented_text`.
3. MinHash signature được tạo cho từng văn bản.
4. LSH tìm các văn bản gần giống.
5. Các bản ghi duplicate được gom nhóm.
6. Mỗi nhóm chỉ giữ một bản ghi đại diện.

### 3.9.3. UC5 - Phân tích chủ đề

| Thành phần | Mô tả |
|---|---|
| Tác nhân | Airflow, Spark, ML module |
| Đầu vào | `stg_posts_core` |
| Xử lý | LDA/BERTopic gán topic |
| Đầu ra | `stg_post_topics`, `stg_topics` |
| File liên quan | `spark_jobs/lda_job.py`, `spark_jobs/bertopic_inference_job.py` |

Luồng chính:

1. Airflow chạy task `lda_topic_modeling`.
2. Spark đọc `stg_posts_core`.
3. Text được vector hóa.
4. Mô hình LDA gán topic cho post.
5. Topic assignment và topic lookup được ghi ra staged dataset.

### 3.9.4. UC6 - Phân tích cảm xúc

| Thành phần | Mô tả |
|---|---|
| Tác nhân | Airflow, Spark, NLP module |
| Đầu vào | `stg_posts_core` |
| Xử lý | PhoBERT inference |
| Đầu ra | `stg_posts_nlp` |
| File liên quan | `spark_jobs/sentiment_job.py`, `models/sentiment_predictor.py` |

Luồng chính:

1. Airflow chạy task `sentiment_analysis`.
2. Spark đọc `stg_posts_core`.
3. Model PhoBERT được load theo partition.
4. Văn bản được phân loại thành positive/neutral/negative.
5. Kết quả được ghi ra `stg_posts_nlp`.

### 3.9.5. UC10 - Xem dashboard

| Thành phần | Mô tả |
|---|---|
| Tác nhân | Người dùng dashboard |
| Đầu vào | ClickHouse mart tables |
| Xử lý | Dashboard query dữ liệu theo time range/topic |
| Đầu ra | Biểu đồ, bảng, KPI, crisis cards |
| File liên quan | `dashboard/src/lib/dal/radar.ts` |

Luồng chính:

1. Người dùng mở dashboard.
2. Dashboard kết nối ClickHouse.
3. Dashboard query các bảng `dbt_fct_topic_activity`, `dbt_dim_topics`, `dbt_fct_crisis_events`.
4. Kết quả được hiển thị dưới dạng KPI, bảng topic, chart sentiment/trend và danh sách crisis.

## 3.10. Ràng buộc và giả định

### 3.10.1. Ràng buộc

- Dữ liệu phụ thuộc vào cấu trúc HTML/API của nguồn crawl.
- Một số nguồn có thể thay đổi layout, làm crawler cần cập nhật.
- PhoBERT và BERTopic có yêu cầu tài nguyên tính toán tương đối lớn.
- Pipeline hiện tại chủ yếu là batch processing, chưa phải real-time streaming hoàn toàn.
- Chất lượng topic/sentiment phụ thuộc vào chất lượng text preprocessing.
- Crisis detection cần dữ liệu lịch sử đủ lớn để baseline đáng tin cậy hơn.

### 3.10.2. Giả định

- Dữ liệu raw đã được crawler lưu đúng vị trí trước khi chạy Spark cleaning.
- Các file reference như stopwords và slang dictionary tồn tại.
- ClickHouse, HDFS, Spark và Airflow đã được khởi động đúng.
- Các bảng ClickHouse staging đã được tạo trước khi ingest.
- Dashboard chỉ đọc từ các bảng mart đã được dbt build.

## 3.11. Tiêu chí hoàn thành

Hệ thống được xem là hoàn thành ở mức đồ án khi đáp ứng các tiêu chí:

| Tiêu chí | Mô tả |
|---|---|
| Pipeline end-to-end | Chạy được từ raw data đến dashboard |
| Dữ liệu lõi | Tạo được `stg_posts_core` sau cleaning/dedup |
| Topic modeling | Sinh được topic assignment và topic lookup |
| Sentiment analysis | Sinh được sentiment label cho bài viết/comment |
| Keyword frequency | Sinh được `stg_keyword_freq` bằng Count-Min Sketch |
| Crisis detection | Sinh được `stg_crisis_events` hoặc xử lý được trường hợp không có event |
| Warehouse | ClickHouse có staging tables và dbt mart tables |
| Dashboard | Hiển thị được overview, trends và crisis |
| Tự động hóa | Airflow DAG điều phối được các bước chính |
| Tài liệu | Có tài liệu báo cáo, hướng dẫn chạy và mô tả kiến trúc |

## 3.12. Tổng kết chương

Chương này đã phân tích các yêu cầu chính của hệ thống, bao gồm yêu cầu chức năng, yêu cầu phi chức năng, dữ liệu đầu vào, dữ liệu đầu ra, các use case chính và tiêu chí hoàn thành. Từ các yêu cầu này, hệ thống được thiết kế theo hướng pipeline nhiều tầng, trong đó dữ liệu đi từ crawler đến HDFS, Spark, ClickHouse, dbt và cuối cùng là dashboard. Các yêu cầu đã đặt nền tảng cho phần thiết kế kiến trúc và triển khai hệ thống ở các chương tiếp theo.

