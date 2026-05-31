# Chương 2. Cơ sở lý thuyết và công nghệ sử dụng

## 2.1. Big Data và xử lý dữ liệu phân tán

Trong các hệ thống phân tích mạng xã hội, dữ liệu thường có khối lượng lớn, tốc độ phát sinh nhanh và định dạng không đồng nhất. Dữ liệu có thể đến từ nhiều nguồn khác nhau như diễn đàn, báo điện tử, nền tảng video hoặc mạng xã hội. Mỗi nguồn lại có cách biểu diễn dữ liệu riêng, ví dụ bài viết, bình luận, phản hồi, lượt thích, lượt xem, thời gian đăng và thông tin người dùng.

Bài toán trong đồ án không chỉ dừng lại ở việc thu thập dữ liệu, mà còn cần xử lý, làm sạch, phân tích và trực quan hóa dữ liệu theo thời gian. Nếu xử lý toàn bộ dữ liệu trên một máy đơn, hệ thống dễ gặp các vấn đề như:

- Không đủ bộ nhớ khi dữ liệu tăng lên.
- Thời gian xử lý lâu khi phải chạy nhiều bước như cleaning, deduplication, topic modeling và sentiment analysis.
- Khó mở rộng khi số lượng nguồn dữ liệu hoặc số lượng bản ghi tăng.
- Khó tự động hóa pipeline nhiều bước.

Vì vậy, đồ án sử dụng kiến trúc Big Data với các thành phần như HDFS, Apache Spark, Airflow và ClickHouse. Trong đó, HDFS đảm nhiệm lưu trữ dữ liệu, Spark xử lý dữ liệu phân tán, Airflow điều phối pipeline, ClickHouse phục vụ truy vấn phân tích và dashboard.

Một hệ thống Big Data thường được đặc trưng bởi các yếu tố:

- **Volume**: khối lượng dữ liệu lớn, có thể lên đến hàng trăm nghìn hoặc hàng triệu bản ghi.
- **Velocity**: dữ liệu được cập nhật theo ngày hoặc theo từng chu kỳ crawl.
- **Variety**: dữ liệu có nhiều dạng khác nhau, như HTML, CSV, text, JSON-like records, bài viết và bình luận.
- **Veracity**: dữ liệu có nhiễu, thiếu trường, trùng lặp, sai định dạng hoặc chứa nhiều nội dung không chuẩn.
- **Value**: giá trị thu được sau khi xử lý là xu hướng, cảm xúc, chủ đề nổi bật và cảnh báo bất thường.

Trong dự án này, pipeline được thiết kế theo hướng xử lý batch hằng ngày. Dữ liệu được crawler thu thập, đưa vào vùng raw, sau đó được Spark xử lý thành dữ liệu staged, tiếp tục được load vào ClickHouse và biến đổi bằng dbt để phục vụ dashboard.

## 2.2. Apache Spark

Apache Spark là framework xử lý dữ liệu phân tán, cho phép xử lý dữ liệu lớn trên nhiều worker node. Spark hỗ trợ nhiều API như RDD, DataFrame, SQL, MLlib và Structured Streaming. Trong đồ án, Spark được sử dụng chủ yếu cho các job xử lý batch như cleaning, deduplication, topic modeling, sentiment inference và crisis detection.

### 2.2.1. Vai trò của Spark trong đồ án

Spark đóng vai trò là tầng xử lý chính của hệ thống. Các job Spark đọc dữ liệu từ HDFS hoặc local mounted path, xử lý dữ liệu theo partition và ghi kết quả ra dạng Parquet.

Các job Spark chính trong dự án gồm:

- `spark_jobs/cleaning_job.py`: chuẩn hóa raw data, làm sạch text, tạo `stg_posts_core`.
- `spark_jobs/lda_job.py`: chạy LDA bằng Spark MLlib.
- `spark_jobs/sentiment_job.py`: chạy PhoBERT inference phân tán.
- `spark_jobs/crisis_detection.py`: tổng hợp feature và phát hiện crisis events.
- `spark_jobs/weekly_retrain.py`: retrain baseline/model cho crisis detection.

### 2.2.2. Spark Driver, Executor và Partition

Trong Spark, **driver** là tiến trình điều phối job, tạo SparkSession, xây dựng execution plan và gửi task xuống các executor. **Executor** là tiến trình chạy trên worker node, chịu trách nhiệm thực thi task và lưu trữ dữ liệu trung gian. Dữ liệu trong Spark được chia thành nhiều **partition**, mỗi partition có thể được xử lý song song.

Trong đồ án, việc chia dữ liệu thành partition giúp:

- Tăng tốc độ xử lý khi dữ liệu lớn.
- Tận dụng nhiều core CPU trên Spark worker.
- Cho phép các bước cleaning, tokenization và inference chạy song song.

Ví dụ, trong `cleaning_job.py`, dữ liệu raw được đọc từ nhiều file CSV, sau đó được repartition để tăng mức độ song song:

```text
read_partitions = max(parallelism * 2, 8)
```

Điều này giúp số partition tối thiểu đủ lớn để tận dụng tài nguyên Spark.

### 2.2.3. DataFrame và RDD

Spark DataFrame cung cấp API dạng bảng, gần với SQL, phù hợp với các thao tác filter, select, join, groupBy và write Parquet. RDD là abstraction thấp hơn, cho phép xử lý linh hoạt hơn ở mức record/partition.

Trong dự án, Spark DataFrame được dùng cho các thao tác dữ liệu có cấu trúc, còn RDD/mapPartitions được dùng khi cần xử lý logic tùy chỉnh, ví dụ:

- Chuẩn hóa dữ liệu từng nguồn crawler.
- Khởi tạo text preprocessor một lần trên mỗi partition.
- Chạy model inference theo batch trong từng partition.

### 2.2.4. `mapPartitions`

Một điểm quan trọng trong các Spark job của dự án là sử dụng `mapPartitions`. Thay vì khởi tạo tài nguyên cho từng dòng dữ liệu, `mapPartitions` cho phép mỗi partition khởi tạo tài nguyên một lần rồi xử lý nhiều dòng.

Trong `spark_jobs/cleaning_job.py`, `mapPartitions` được dùng để:

- Load adapter tương ứng với từng nguồn dữ liệu.
- Khởi tạo `TextPreprocessor`.
- Load stopwords, slang dictionary và tokenizer.
- Xử lý nhiều bản ghi trong cùng một partition.

Cách làm này hiệu quả hơn so với việc khởi tạo preprocessor cho từng record, đặc biệt khi tokenizer hoặc model NLP có chi phí khởi tạo lớn.

### 2.2.5. Persist và Checkpoint

Spark có cơ chế lazy evaluation, tức là các transformation chỉ được thực thi khi có action như `count()`, `write()` hoặc `collect()`. Khi một DataFrame được dùng lại nhiều lần, có thể dùng `persist()` để lưu tạm dữ liệu, giảm việc tính toán lại.

Trong `cleaning_job.py`, dữ liệu sau khi chuẩn hóa `post_id` được persist bằng `MEMORY_AND_DISK`. Điều này phù hợp khi dữ liệu có thể lớn hơn RAM, Spark sẽ lưu phần dư xuống disk.

Checkpoint được dùng trước bước MinHash/LSH dedup để cắt lineage Spark. Khi pipeline có nhiều transformation, lineage dài có thể làm job khó phục hồi hoặc tốn chi phí tính toán lại. Checkpoint giúp lưu trạng thái trung gian ra HDFS và làm execution plan gọn hơn.

## 2.3. HDFS và mô hình Data Lake

HDFS là hệ thống file phân tán trong hệ sinh thái Hadoop. HDFS được thiết kế để lưu trữ file lớn trên nhiều node, hỗ trợ cơ chế replication để tăng độ tin cậy. Trong đồ án, HDFS đóng vai trò là data lake, lưu cả dữ liệu thô và dữ liệu đã xử lý.

### 2.3.1. Vai trò của HDFS trong hệ thống

HDFS được dùng để lưu:

- Dữ liệu raw từ crawler.
- Reference files như stopwords và slang dictionary.
- Output Parquet của các Spark job.
- Staged datasets trước khi ingest vào ClickHouse.

Các path chính trong pipeline:

```text
/user/root/raw_data
/user/root/ref/stopwords_vi.txt
/user/root/ref/slang_dict.json
/user/root/staged/stg_posts_core
/user/root/staged/stg_posts_nlp
/user/root/staged/stg_post_topics
/user/root/staged/stg_topics
/user/root/staged/stg_keyword_freq
/user/root/staged/stg_crisis_events
```

### 2.3.2. Mô hình phân tầng dữ liệu

Pipeline dữ liệu của dự án có thể được hiểu theo mô hình nhiều tầng:

- **Raw/Bronze layer**: lưu dữ liệu gốc từ crawler, chưa xử lý nhiều.
- **Silver layer**: dữ liệu đã được Spark làm sạch, chuẩn hóa, gán sentiment/topic hoặc tính keyword frequency.
- **Staging layer**: dữ liệu được load vào ClickHouse dưới dạng các bảng `stg_*`.
- **Gold layer**: dữ liệu đã được dbt tổng hợp thành mart tables phục vụ dashboard.

Cách chia tầng này giúp hệ thống dễ kiểm soát lineage dữ liệu. Nếu một bước xử lý bị lỗi, nhóm có thể kiểm tra từng tầng để xác định lỗi đến từ raw data, Spark job, ingest ClickHouse hay dbt transformation.

### 2.3.3. Định dạng Parquet

Các Spark job ghi output ra định dạng Parquet. Parquet là định dạng lưu trữ dạng cột, phù hợp với workload phân tích vì:

- Chỉ đọc các cột cần thiết thay vì đọc toàn bộ record.
- Nén tốt hơn so với CSV hoặc JSON.
- Giữ được schema.
- Tương thích tốt với Spark và ClickHouse.

Trong dự án, output của `spark_cleaning` được ghi:

```text
stg_posts_core/
  source=voz/
  source=vatvo/
  source=vnexpress/
```

Việc partition theo `source` giúp các job downstream có thể lọc dữ liệu theo nguồn hiệu quả hơn.

## 2.4. Apache Airflow

Apache Airflow là công cụ điều phối workflow. Airflow cho phép định nghĩa pipeline dưới dạng DAG, trong đó mỗi node là một task và các cạnh thể hiện quan hệ phụ thuộc giữa các task.

### 2.4.1. Vai trò của Airflow

Trong đồ án, Airflow được dùng để tự động hóa toàn bộ pipeline:

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

DAG chính nằm trong file:

```text
dags/processing_dag.py
```

### 2.4.2. Các loại task trong Airflow

Pipeline sử dụng một số loại operator:

- `DummyOperator`: đánh dấu điểm bắt đầu/kết thúc.
- `PythonOperator`: chạy logic Python, ví dụ validate mount hoặc ingest dataset.
- `BashOperator`: chạy lệnh shell, ví dụ `spark-submit` hoặc `dbt run`.

Các task như `spark_cleaning`, `lda_topic_modeling`, `sentiment_analysis` được chạy thông qua `spark-submit`. Cách này giúp Airflow chỉ đóng vai trò điều phối, còn xử lý dữ liệu lớn do Spark đảm nhiệm.

### 2.4.3. Lợi ích của Airflow trong dự án

Airflow giúp:

- Quản lý thứ tự chạy các bước.
- Theo dõi trạng thái từng task.
- Tự động retry khi task lỗi.
- Tách biệt logic từng module.
- Dễ trigger demo pipeline.
- Giúp pipeline end-to-end có tính tự động thay vì chạy thủ công từng file.

## 2.5. ClickHouse và dbt

Sau khi Spark xử lý dữ liệu và ghi ra HDFS staged Parquet, dữ liệu được ingest vào ClickHouse. ClickHouse đóng vai trò là kho dữ liệu phân tích, còn dbt được dùng để biến đổi dữ liệu từ staging sang các bảng mart phục vụ dashboard.

### 2.5.1. ClickHouse

ClickHouse là hệ quản trị cơ sở dữ liệu dạng cột, tối ưu cho truy vấn phân tích OLAP. Với dữ liệu phân tích xu hướng, dashboard thường cần các truy vấn như:

- Tổng số mentions theo giờ.
- Top topic theo trend score.
- Phân bố sentiment theo topic.
- Tỷ lệ negative sentiment.
- Danh sách crisis events gần đây.

Các truy vấn này thường đọc nhiều dòng nhưng chỉ cần một số cột nhất định. Vì vậy, lưu trữ dạng cột của ClickHouse phù hợp hơn so với cơ sở dữ liệu row-based truyền thống.

Các bảng staging chính trong ClickHouse:

```text
tech_radar.stg_posts_core
tech_radar.stg_posts_nlp
tech_radar.stg_post_topics
tech_radar.stg_topics
tech_radar.stg_keyword_freq
tech_radar.stg_crisis_events
```

DDL tạo bảng nằm trong:

```text
warehouse/clickhouse/init_schema.sql
```

### 2.5.2. dbt

dbt là công cụ transformation trong mô hình ELT. Thay vì biến đổi toàn bộ dữ liệu trong Spark, hệ thống đưa dữ liệu đã staged vào ClickHouse, sau đó dùng dbt để viết các transformation bằng SQL.

Trong dự án, dbt đảm nhiệm:

- Join dữ liệu core, sentiment và topic.
- Tính engagement từ các chỉ số atomic.
- Tổng hợp dữ liệu theo topic, source và hour bucket.
- Tính velocity, acceleration, trend score.
- Tạo mart phục vụ dashboard.

Các model chính:

- `int_posts_enriched`: join `stg_posts_core`, `stg_posts_nlp`, `stg_post_topics`.
- `int_topic_sentiment_hourly`: tổng hợp sentiment/topic theo giờ.
- `fct_topic_activity`: fact table chính cho dashboard.
- `dim_topics`: dimension topic.
- `fct_crisis_events`: mart cho crisis monitor.

### 2.5.3. Lý do tách Spark và dbt

Spark được dùng cho xử lý dữ liệu lớn và các bước nặng như cleaning, dedup, ML inference. dbt được dùng cho logic phân tích ở warehouse như join, aggregate, tính điểm và chuẩn hóa bảng phục vụ dashboard.

Cách tách này có các lợi ích:

- Không cần rerun Spark nếu chỉ thay đổi công thức trend score.
- Giảm cross-job shuffle trong Spark.
- Các module M2, M3, M4 có thể ghi output độc lập.
- dbt là nơi hợp nhất dữ liệu và phục vụ dashboard.

## 2.6. MinHash và Locality-Sensitive Hashing

Trong dữ liệu crawler, bài viết và bình luận có thể bị trùng lặp hoặc gần trùng lặp. Ví dụ, cùng một nội dung có thể được crawl nhiều lần, được repost, được quote lại hoặc chỉ khác một vài ký tự. Nếu không xử lý, dữ liệu trùng có thể làm sai lệch topic modeling, sentiment analysis và trend score.

### 2.6.1. Jaccard Similarity

Độ tương đồng Jaccard giữa hai tập hợp A và B được tính bằng:

```text
J(A, B) = |A ∩ B| / |A ∪ B|
```

Trong bài toán văn bản, mỗi văn bản có thể được chuyển thành tập các shingles. Hai văn bản càng giống nhau thì tập shingles của chúng càng có nhiều phần tử chung.

### 2.6.2. Shingling

Shingling là kỹ thuật cắt văn bản thành các đoạn nhỏ liên tiếp. Trong dự án, class `MinHashDeduplicator` dùng tham số:

```text
k = 5
```

Tức là văn bản được cắt thành các shingles có độ dài 5 ký tự. Tập shingles này là biểu diễn để tính độ giống giữa hai văn bản.

### 2.6.3. MinHash

Việc tính Jaccard trực tiếp giữa mọi cặp văn bản rất tốn kém, vì nếu có n văn bản thì số cặp cần so sánh là O(n²). MinHash giải quyết vấn đề này bằng cách tạo signature ngắn đại diện cho mỗi tập shingles.

Ý tưởng chính: nếu hai tập có Jaccard similarity cao thì MinHash signature của chúng có xác suất giống nhau cao.

Trong dự án, tham số:

```text
num_perm = 128
```

Số lượng permutation càng lớn thì ước lượng càng ổn định, nhưng chi phí tính toán và bộ nhớ cũng tăng.

### 2.6.4. Locality-Sensitive Hashing

LSH giúp tìm nhanh các văn bản có khả năng giống nhau mà không cần so sánh tất cả các cặp. Các MinHash signature được đưa vào LSH index; khi query một signature, LSH trả về các candidate có độ tương đồng cao.

Trong `spark_jobs/cleaning_job.py`, pipeline gọi:

```text
threshold = 0.85
```

Điều này nghĩa là chỉ những văn bản có độ giống cao mới bị coi là near-duplicate, giúp giảm nguy cơ xóa nhầm các nội dung chỉ hơi giống nhau.

### 2.6.5. Ứng dụng trong dự án

File chính:

```text
algorithms/minhash_dedup.py
```

Quy trình dedup:

1. Tạo `__dedup_row_id` cho mỗi dòng.
2. Lấy `segmented_text` để tạo shingles.
3. Tạo MinHash signature.
4. Query LSH để tìm near-duplicate.
5. Gom nhóm duplicate bằng union-find.
6. Giữ lại một bản ghi đại diện, ưu tiên bản ghi có `created_at` sớm hơn.
7. Join lại với DataFrame gốc để lấy dữ liệu sau dedup.

## 2.7. Count-Min Sketch

Count-Min Sketch là cấu trúc dữ liệu xác suất dùng để ước lượng tần suất xuất hiện của phần tử trong stream dữ liệu. Thay vì lưu exact count cho mọi keyword, Count-Min Sketch dùng một ma trận đếm có kích thước cố định và nhiều hàm băm.

### 2.7.1. Bài toán

Trong phân tích xu hướng, hệ thống cần biết từ khóa nào đang xuất hiện nhiều. Nếu lưu exact count cho mọi keyword, bộ nhớ có thể tăng rất nhanh khi số lượng từ vựng lớn. Count-Min Sketch giúp ước lượng tần suất với bộ nhớ cố định.

### 2.7.2. Cách hoạt động

Count-Min Sketch gồm:

- `depth`: số hàng, tương ứng số hàm băm.
- `width`: số cột trong mỗi hàng.
- Ma trận đếm `depth x width`.

Khi thêm một keyword:

1. Mỗi hàm băm ánh xạ keyword vào một cột.
2. Tăng bộ đếm tại vị trí tương ứng ở từng hàng.
3. Khi query tần suất, lấy giá trị nhỏ nhất trong các hàng.

Việc lấy giá trị nhỏ nhất giúp giảm ảnh hưởng của collision.

### 2.7.3. Ứng dụng trong dự án

File chính:

```text
algorithms/count_min_sketch.py
```

Count-Min Sketch được dùng để tạo dataset:

```text
stg_keyword_freq
```

Schema output:

```text
keyword
window_start
window_end
estimated_count
source
```

Task Airflow liên quan:

```text
cms_keyword_counting
```

Ý nghĩa: hệ thống có thể theo dõi keyword nổi bật theo từng time window mà không cần lưu exact count cho toàn bộ từ vựng.

## 2.8. Topic Modeling: LDA và BERTopic

Topic modeling là bài toán tự động khám phá các chủ đề tiềm ẩn trong tập văn bản. Trong đồ án, topic modeling giúp trả lời câu hỏi: người dùng đang bàn luận về chủ đề gì?

### 2.8.1. LDA

LDA, viết tắt của Latent Dirichlet Allocation, là mô hình xác suất cho topic modeling. LDA giả định rằng:

- Mỗi document là một hỗn hợp của nhiều topic.
- Mỗi topic là một phân phối trên các từ.
- Khi sinh một từ trong document, mô hình chọn topic trước, sau đó chọn từ từ phân phối của topic đó.

Trong dự án, LDA được triển khai bằng Spark MLlib:

```text
spark_jobs/lda_job.py
```

Pipeline LDA gồm các bước:

- Đọc `stg_posts_core`.
- Lấy text đã xử lý.
- Tokenize/normalize.
- Dùng CountVectorizer/TF-IDF.
- Train hoặc infer LDA.
- Gán `topic_id` cho từng post.
- Xuất `stg_post_topics` và `stg_topics`.

Ưu điểm của LDA:

- Có thể chạy phân tán bằng Spark.
- Phù hợp với batch pipeline.
- Kết quả dễ biểu diễn bằng top keywords.

Hạn chế:

- Không hiểu ngữ nghĩa sâu như embedding model.
- Nhạy với chất lượng preprocessing.
- Có thể khó đặt nhãn topic nếu keywords nhiễu.

### 2.8.2. BERTopic

BERTopic là phương pháp topic modeling dựa trên embedding. Thay vì chỉ dựa vào bag-of-words, BERTopic biểu diễn văn bản bằng vector embedding, sau đó clustering các văn bản gần nhau và dùng c-TF-IDF để rút ra từ khóa đại diện cho topic.

Trong dự án, BERTopic liên quan đến:

```text
models/bertopic_model.py
spark_jobs/bertopic_inference_job.py
```

Pipeline tổng quát:

1. Tạo embedding cho văn bản.
2. Giảm chiều embedding bằng UMAP.
3. Clustering bằng HDBSCAN.
4. Tính c-TF-IDF để lấy top keywords.
5. Gán topic cho document.

Ưu điểm:

- Bắt được ngữ nghĩa tốt hơn LDA.
- Phù hợp với văn bản ngắn nếu embedding tốt.
- Topic có thể tự nhiên hơn.

Hạn chế:

- Tốn tài nguyên hơn LDA.
- Có thể cần GPU hoặc máy mạnh.
- Khó phân tán hoàn toàn như Spark MLlib.

### 2.8.3. Kết hợp LDA và BERTopic trong dự án

Dự án dùng cả LDA và BERTopic để tận dụng ưu điểm của từng hướng:

- LDA chạy hằng ngày, phù hợp pipeline phân tán.
- BERTopic có thể chạy theo chu kỳ dài hơn, ví dụ hằng tuần, để cải thiện chất lượng topic.
- Cả hai cùng ghi về các bảng topic như `stg_post_topics` và `stg_topics`, có trường `model_type` để phân biệt.

## 2.9. Sentiment Analysis với PhoBERT

Sentiment analysis là bài toán xác định cảm xúc hoặc thái độ trong văn bản. Trong đồ án, hệ thống phân loại cảm xúc thành ba lớp:

```text
positive
neutral
negative
```

### 2.9.1. PhoBERT

PhoBERT là mô hình ngôn ngữ được huấn luyện cho tiếng Việt, dựa trên kiến trúc RoBERTa. Do tiếng Việt có đặc thù về tách từ và dấu, việc sử dụng mô hình chuyên cho tiếng Việt giúp cải thiện chất lượng phân tích so với mô hình tiếng Anh hoặc multilingual model thông thường.

Trong dự án, các file liên quan:

```text
models/train_phobert.py
models/sentiment_predictor.py
spark_jobs/sentiment_job.py
```

### 2.9.2. Fine-tuning và inference

Quy trình sentiment gồm:

- Chuẩn bị dữ liệu gán nhãn.
- Fine-tune PhoBERT cho bài toán 3-class sentiment.
- Lưu model checkpoint.
- Dùng `SentimentPredictor` để predict batch văn bản.
- Tích hợp predictor vào Spark job.

Trong Spark job, model được load theo partition thay vì load cho từng dòng. Điều này rất quan trọng vì model PhoBERT có kích thước lớn, nếu load lại liên tục sẽ gây chậm và tốn bộ nhớ.

Output của sentiment job là:

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

### 2.9.3. Ý nghĩa trong hệ thống

Sentiment labels được dùng để:

- Tính phân bố positive/neutral/negative theo topic.
- Tính `neg_ratio`.
- Hỗ trợ phát hiện crisis khi tỷ lệ tiêu cực tăng cao.
- Hiển thị sentiment timeline trên dashboard.

## 2.10. Phát hiện bất thường và khủng hoảng dư luận

Khủng hoảng dư luận trong đồ án được hiểu là tình huống một chủ đề hoặc nhóm bài viết có dấu hiệu tăng mạnh về thảo luận, đi kèm tỷ lệ cảm xúc tiêu cực cao hoặc biến động bất thường so với baseline.

### 2.10.1. Các tín hiệu sử dụng

Các feature thường dùng:

- `comment_count`: số lượng comment trong một khoảng thời gian.
- `unique_posts`: số bài viết liên quan.
- `unique_users`: số người tham gia.
- `neg_ratio`: tỷ lệ sentiment tiêu cực.
- `mention_velocity`: tốc độ tăng mention theo giờ.
- `acceleration`: độ tăng của velocity.
- `z_score`: mức độ lệch so với baseline.
- `cross_source_ratio`: mức độ lan rộng qua nhiều nguồn.

### 2.10.2. Isolation Forest

Isolation Forest là thuật toán phát hiện bất thường. Ý tưởng là các điểm bất thường thường dễ bị cô lập hơn các điểm bình thường. Thuật toán xây dựng nhiều cây ngẫu nhiên, sau đó đo độ dài đường đi để cô lập một điểm. Điểm nào có đường đi ngắn hơn thường có khả năng là anomaly.

Trong dự án, Isolation Forest được dùng trong:

```text
models/isolation_forest.py
spark_jobs/crisis_detection.py
```

Nếu model artifact không sẵn sàng, crisis job có thể fallback sang rule-based logic để pipeline không bị crash.

### 2.10.3. Rolling baseline và rule-based detection

Ngoài Isolation Forest, hệ thống còn có thể dùng rolling baseline:

- Tính trung bình trong 24 giờ gần nhất.
- Tính trung bình trong 7 ngày gần nhất.
- Tính z-score của volume hoặc negative ratio.
- Cảnh báo khi nhiều điều kiện cùng xảy ra.

Ví dụ điều kiện:

- Tỷ lệ negative sentiment cao.
- Mention velocity tăng mạnh.
- Volume z-score vượt ngưỡng.

Output của crisis detection là:

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

### 2.10.4. Ý nghĩa trong hệ thống

Crisis detection giúp dashboard không chỉ hiển thị xu hướng phổ biến, mà còn phát hiện các tín hiệu bất thường có thể cần chú ý sớm. Đây là phần quan trọng nếu hệ thống được ứng dụng trong social listening hoặc quản trị truyền thông.

## 2.11. Dashboard Next.js

Dashboard là tầng hiển thị cuối cùng của hệ thống. Trong đồ án, dashboard được xây dựng bằng Next.js và đọc dữ liệu từ ClickHouse marts.

### 2.11.1. Vai trò của dashboard

Dashboard giúp người dùng:

- Xem tổng quan số lượng mentions.
- Xem top trending topics.
- Xem phân bố cảm xúc.
- Xem diễn biến trend score theo thời gian.
- Xem keywords đại diện của topic.
- Xem các bài viết/bình luận làm bằng chứng.
- Theo dõi crisis events.

### 2.11.2. Nguồn dữ liệu dashboard

Dashboard không đọc trực tiếp từ HDFS hoặc raw data. Thay vào đó, dashboard query các bảng đã được dbt build trong ClickHouse:

```text
tech_radar.dbt_fct_topic_activity
tech_radar.dbt_dim_topics
tech_radar.dbt_fct_crisis_events
tech_radar.dbt_int_posts_enriched
```

File query chính:

```text
dashboard/src/lib/dal/radar.ts
```

### 2.11.3. Các trang chính

Dashboard gồm các trang:

- **Overview**: hiển thị KPI tổng quan, top trending topics và sentiment breakdown.
- **Trends Explorer**: xem chi tiết một topic, gồm trend score timeline, sentiment timeline, keywords và evidence posts.
- **Crisis Monitor**: hiển thị crisis events, severity, trigger conditions và evidence posts.

### 2.11.4. Lợi ích của việc query mart tables

Dashboard đọc từ mart tables thay vì staging/raw data vì:

- Truy vấn nhanh hơn.
- Dữ liệu đã được join và aggregate sẵn.
- Logic nghiệp vụ như trend score đã được tính trong dbt.
- Dashboard code đơn giản hơn.
- Giảm tải cho hệ thống xử lý phía sau.

## 2.12. Tổng kết chương

Chương này đã trình bày các cơ sở lý thuyết và công nghệ chính được sử dụng trong đồ án. Hệ thống kết hợp nhiều thành phần: HDFS để lưu trữ dữ liệu phân tán, Spark để xử lý dữ liệu lớn, Airflow để điều phối pipeline, ClickHouse và dbt để xây dựng warehouse phục vụ phân tích, cùng các thuật toán NLP/Data Mining như MinHash/LSH, Count-Min Sketch, LDA, BERTopic, PhoBERT và Isolation Forest.

Việc kết hợp các công nghệ này giúp đồ án xây dựng được một pipeline tương đối đầy đủ từ thu thập dữ liệu, xử lý dữ liệu, phân tích chủ đề/cảm xúc, phát hiện bất thường cho đến trực quan hóa kết quả trên dashboard.

