# Báo cáo phần việc M2 - Data Infrastructure & Spark Processing

## 1. Vai trò và phạm vi phụ trách

Trong dự án **Distributed NLP Trend Analysis**, em phụ trách phần **M2 - Data Infrastructure & Spark Processing**. Đây là phần nền tảng của hệ thống, có nhiệm vụ chuẩn bị môi trường xử lý dữ liệu lớn và xây dựng pipeline xử lý dữ liệu lõi để các module phía sau có thể sử dụng.

Mục tiêu chính của phần M2 là biến dữ liệu thô từ các crawler thành một bảng dữ liệu chuẩn, sạch và có cấu trúc thống nhất. Bảng dữ liệu lõi này là `stg_posts_core`, được dùng làm đầu vào cho các phần như topic modeling, sentiment analysis, crisis detection và dashboard.

Phạm vi công việc của em gồm 3 nhóm chính:

- Xây dựng hạ tầng DevOps / Infrastructure cho hệ thống Big Data.
- Xây dựng Spark cleaning pipeline để chuẩn hóa dữ liệu thô thành `stg_posts_core`.
- Tích hợp thuật toán MinHash/LSH để loại bỏ dữ liệu trùng lặp và gần trùng lặp.

Các phần như crawler, topic modeling, sentiment analysis, crisis detection và dashboard không phải phần em trực tiếp xây dựng chính. Tuy nhiên, phần M2 có vai trò kết nối và cung cấp dữ liệu đầu vào để các phần đó chạy được trong pipeline tổng.

## 2. DevOps / Infrastructure

Phần infrastructure có nhiệm vụ dựng môi trường chạy cho toàn bộ hệ thống xử lý dữ liệu. Vì dự án có nhiều thành phần như HDFS, Spark, Airflow, ClickHouse và dbt, nên nếu cấu hình thủ công sẽ khó kiểm soát và dễ lỗi. Vì vậy, em triển khai phần hạ tầng theo hướng có thể tự động hóa và tái lập được.

Các thành phần chính trong hạ tầng gồm:

- **HDFS/Hadoop**: lưu trữ dữ liệu thô và dữ liệu đã xử lý ở dạng staged data.
- **Apache Spark**: xử lý dữ liệu phân tán, chạy các job cleaning và NLP.
- **Apache Airflow**: điều phối pipeline, quản lý thứ tự chạy các task.
- **ClickHouse**: lưu dữ liệu staging và phục vụ truy vấn phân tích.
- **Docker/Docker Compose**: tạo môi trường chạy local/development cho các service.
- **Ansible**: tự động hóa việc cài đặt và cấu hình cluster.

### 2.1. Cấu hình Docker và môi trường chạy

Em cấu hình Docker/Docker Compose để các service trong hệ thống có thể chạy cùng nhau. Docker giúp đóng gói môi trường chạy, giảm khác biệt giữa các máy và giúp việc khởi động hệ thống đơn giản hơn.

Các file liên quan:

- `docker-compose.yml`
- `Dockerfile.spark`
- `Dockerfile.airflow`
- `env.example`
- `LOCAL_GUIDE.md`

Trong đó:

- `docker-compose.yml` định nghĩa các service chính của hệ thống.
- `Dockerfile.spark` chuẩn bị môi trường Spark để chạy các Spark job.
- `Dockerfile.airflow` chuẩn bị môi trường Airflow để điều phối pipeline.
- `env.example` lưu các biến môi trường mẫu như đường dẫn HDFS, ClickHouse, Spark master.
- `LOCAL_GUIDE.md` hướng dẫn cách chạy hệ thống ở môi trường local.

Kết quả của phần này là nhóm có thể chạy pipeline trong môi trường local hoặc containerized environment mà không cần cài đặt thủ công toàn bộ stack trên máy.

### 2.2. Tự động hóa bằng Ansible

Ngoài Docker, em cũng triển khai phần Ansible để tự động hóa cài đặt và cấu hình cluster. Ansible giúp mô tả hạ tầng bằng code, từ đó việc cài đặt có thể lặp lại trên nhiều node mà không cần thao tác thủ công từng máy.

Các thành phần được Ansible hỗ trợ cài đặt/cấu hình gồm:

- Java, là dependency cần thiết cho Hadoop và Spark.
- Conda và môi trường Python.
- Hadoop/HDFS.
- Apache Spark.
- ClickHouse.
- dbt.
- Airflow.
- Các dependency cần thiết cho NLP pipeline.

Các file/folder liên quan:

- `ansible/`
- `ansible/inventory/hosts.ini`
- `ansible/playbooks/01_java.yml`
- `ansible/playbooks/02_conda.yml`
- `ansible/playbooks/03_hdfs.yml`
- `ansible/playbooks/04_spark.yml`
- `ansible/playbooks/05_clickhouse.yml`
- `ansible/playbooks/06_dbt.yml`
- `ansible/playbooks/07_airflow.yml`
- `ansible/playbooks/install_nlp_deps.yml`
- `ansible/playbooks/start_services.yml`
- `ansible/roles/`

Ý nghĩa của từng nhóm playbook:

- `01_java.yml`: chuẩn bị Java cho các node.
- `02_conda.yml`: chuẩn bị môi trường Python.
- `03_hdfs.yml`: cấu hình Hadoop/HDFS.
- `04_spark.yml`: cấu hình Spark cluster.
- `05_clickhouse.yml`: cài đặt/cấu hình ClickHouse.
- `06_dbt.yml`: chuẩn bị dbt để chạy transformation layer.
- `07_airflow.yml`: cài đặt/cấu hình Airflow.
- `install_nlp_deps.yml`: cài các thư viện phục vụ xử lý NLP.
- `start_services.yml`: hỗ trợ khởi động các service cần thiết.

Nhờ phần này, hệ thống có nền tảng để chạy pipeline end-to-end: dữ liệu được đưa vào HDFS, Spark đọc và xử lý, Airflow điều phối các bước, ClickHouse lưu kết quả để các phần phân tích và dashboard sử dụng.

## 3. Spark Cleaning Pipeline cho dữ liệu lõi

File chính của phần này:

- `spark_jobs/cleaning_job.py`

Đây là phần quan trọng nhất của M2. Spark cleaning pipeline có nhiệm vụ đọc dữ liệu thô từ nhiều crawler khác nhau, chuẩn hóa về cùng một schema, làm sạch text và ghi ra dataset lõi `stg_posts_core`.

### 3.1. Mục tiêu của cleaning pipeline

Dữ liệu crawler ban đầu thường không đồng nhất. Mỗi nguồn có cấu trúc khác nhau, tên cột khác nhau và chất lượng text khác nhau. Ví dụ, dữ liệu từ VOZ có post và comment, VnExpress có post và comment, còn VatVo có article. Nếu các module phía sau đọc trực tiếp từng nguồn này thì sẽ rất khó xử lý.

Vì vậy, Spark cleaning pipeline được xây dựng để:

- Đọc dữ liệu thô từ HDFS hoặc local path.
- Chuẩn hóa dữ liệu từ nhiều nguồn về một schema chung.
- Làm sạch nội dung văn bản.
- Tạo các cột text phục vụ cho NLP và topic modeling.
- Loại bỏ các bản ghi lỗi hoặc thiếu nội dung.
- Chuẩn hóa các trường thời gian, số lượng reaction/comment/view.
- Ghi output ra Parquet để các job sau sử dụng.

### 3.2. Đầu vào của pipeline

Trong `cleaning_job.py`, các nguồn dữ liệu được khai báo trong `RAW_INPUTS`:

```python
RAW_INPUTS = (
    ("voz_comments", "voz/comments.csv"),
    ("voz_posts", "voz/posts.csv"),
    ("vatvo_articles", "vatvo/articles.csv"),
    ("vnexpress_posts", "vnexpress/post_vnexpress.csv"),
    ("vnexpress_comments", "vnexpress/comment_vnexpress.csv"),
)
```

Như vậy pipeline xử lý 5 loại input:

- Comment từ VOZ.
- Post từ VOZ.
- Article từ VatVo.
- Post từ VnExpress.
- Comment từ VnExpress.

Các input này có thể được đọc từ HDFS hoặc local tùy theo cấu hình biến môi trường. Điều này giúp pipeline chạy được cả trong môi trường local và môi trường cluster.

### 3.3. Cấu hình đường dẫn và biến môi trường

Trong file `cleaning_job.py`, nhiều cấu hình được lấy từ biến môi trường:

- `HDFS_BASE`: địa chỉ HDFS base.
- `HDFS_INPUT`: đường dẫn dữ liệu raw.
- `HDFS_OUTPUT`: đường dẫn output của `stg_posts_core`.
- `NLP_SLANG_DICT`: đường dẫn slang dictionary.
- `NLP_STOPWORDS`: đường dẫn stopwords.
- `NLP_TOKENIZER`: lựa chọn tokenizer, ví dụ `underthesea` hoặc `vncorenlp`.
- `NLP_VNCORENLP_JAR`: đường dẫn JAR của VnCoreNLP nếu dùng tokenizer này.

Việc cấu hình bằng biến môi trường giúp Airflow có thể inject giá trị khi chạy job. Nhờ đó cùng một code có thể chạy được ở nhiều môi trường khác nhau mà không cần sửa trực tiếp trong source code.

### 3.4. Chuẩn hóa dữ liệu từ nhiều nguồn

Mỗi nguồn crawler có format khác nhau, vì vậy pipeline có các hàm build riêng:

- `_build_voz_comment`
- `_build_voz_post`
- `_build_vatvo_article`
- `_build_vnexpress_post`
- `_build_vnexpress_comment`

Các hàm này có nhiệm vụ map dữ liệu thô về schema chung. Ví dụ:

- `comment`, `body`, `content` được đưa về nội dung chung.
- `user`, `author`, `author_name` được đưa về `author`.
- `id_post`, `post_id`, `comment_id` được đưa về `post_id`.
- `reaction_count`, `view_count`, `comment_count` được ép về dạng số.
- `created_at` được chuyển về dạng datetime.

Nhờ bước này, các module sau không cần quan tâm dữ liệu đến từ nguồn nào, mà chỉ cần đọc một schema thống nhất là `stg_posts_core`.

### 3.5. Schema output `stg_posts_core`

Output của pipeline được định nghĩa trong `OUTPUT_SCHEMA`, gồm các cột:

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

Ý nghĩa các cột chính:

- `post_id`: khóa định danh của bài viết/comment. Nếu thiếu thì pipeline sinh UUID.
- `source`: nguồn dữ liệu, ví dụ `voz`, `vatvo`, `vnexpress`.
- `author`: tác giả/người đăng.
- `title`: tiêu đề nếu có.
- `body`: nội dung sau khi bỏ HTML/quote cơ bản, giữ gần với bản gốc.
- `clean_text`: nội dung đã được làm sạch.
- `segmented_text`: nội dung đã clean, tách từ tiếng Việt và loại stopwords.
- `topic_text`: nội dung tối ưu hơn cho topic modeling, loại nhiễu mạnh hơn.
- `parent_id`: dùng cho comment để biết comment thuộc bài viết nào.
- `reaction_count`: số lượng reaction/like.
- `view_count`: số lượt xem nếu nguồn có dữ liệu này.
- `comment_count`: số lượng comment.
- `created_at`: thời gian bài viết/comment được tạo.
- `crawled_at`: thời gian dữ liệu được crawler thu thập.

Đây là schema lõi vì nó chứa cả thông tin nội dung, metadata và các chỉ số tương tác cần thiết cho phân tích xu hướng.

### 3.6. Làm sạch và xử lý text

Pipeline sử dụng `TextPreprocessor` để tạo ra nhiều phiên bản text khác nhau. Trong `cleaning_job.py`, hàm `_text_outputs` trả về:

```python
body, clean_text, segmented_text, topic_text
```

Mỗi loại text có mục đích riêng:

#### `body`

`body` là nội dung đã được làm sạch nhẹ, chủ yếu bỏ HTML tag và quote thừa. Cột này vẫn giữ tương đối gần với nội dung gốc để phục vụ việc kiểm tra hoặc hiển thị lại khi cần.

#### `clean_text`

`clean_text` là text đã được làm sạch kỹ hơn. Các bước xử lý gồm:

- Chuyển về chữ thường.
- Xóa HTML.
- Xóa URL.
- Xóa email.
- Xóa mention.
- Xóa emoji.
- Chuẩn hóa Unicode.
- Chuẩn hóa slang bằng slang dictionary.
- Xóa các từ nhiễu đặc thù diễn đàn/mạng xã hội.
- Xóa ký tự đặc biệt.
- Chuẩn hóa khoảng trắng.

#### `segmented_text`

`segmented_text` là text đã qua cleaning, sau đó được tách từ tiếng Việt và loại bỏ stopwords. Đây là cột quan trọng cho các job NLP phía sau vì tiếng Việt cần word segmentation để mô hình hiểu đúng các cụm từ.

Ví dụ ý tưởng:

```text
học máy rất tốt -> học_máy tốt
```

#### `topic_text`

`topic_text` là biến thể dành riêng cho topic modeling. Cột này loại bỏ stopwords mạnh hơn `segmented_text`, giúp giảm nhiễu và làm nổi bật các từ khóa chủ đề.

Việc tách riêng `segmented_text` và `topic_text` là cần thiết vì sentiment analysis và topic modeling có nhu cầu khác nhau. Sentiment analysis có thể cần giữ nhiều tín hiệu ngữ nghĩa hơn, còn topic modeling cần text gọn và ít nhiễu hơn.

### 3.7. Xử lý phân tán bằng Spark

Pipeline dùng Spark để xử lý dữ liệu phân tán. Một điểm quan trọng là sử dụng `mapPartitions` thay vì xử lý từng dòng một cách độc lập.

Trong hàm `process_partition`, mỗi partition sẽ:

1. Chuyển các Spark Row thành dictionary.
2. Khởi tạo adapter tương ứng với nguồn dữ liệu.
3. Khởi tạo `TextPreprocessor`.
4. Chuẩn hóa dữ liệu bằng adapter.
5. Tạo output row theo schema chuẩn.

Lý do dùng `mapPartitions`:

- Giảm chi phí khởi tạo preprocessor.
- Không phải load slang dictionary, stopwords hoặc tokenizer cho từng record.
- Phù hợp hơn với xử lý phân tán trên Spark.
- Tận dụng được executor/partition của Spark khi dữ liệu lớn.

Cách này giúp pipeline hiệu quả hơn so với việc tạo công cụ xử lý text ở từng dòng dữ liệu.

### 3.8. Lọc dữ liệu và chuẩn hóa sau xử lý

Sau khi tạo DataFrame theo schema chuẩn, pipeline tiếp tục lọc bỏ các record không hợp lệ. Các điều kiện chính:

- `body` không được null hoặc rỗng.
- `segmented_text` không được null hoặc rỗng.
- `topic_text` không được null hoặc rỗng.

Sau đó pipeline thực hiện:

- `dropDuplicates()` để loại bỏ các dòng trùng hoàn toàn.
- Chuẩn hóa `post_id`; nếu null, rỗng hoặc `"None"` thì sinh UUID.
- Persist DataFrame bằng `MEMORY_AND_DISK` để tối ưu cho các bước sau.
- Checkpoint trước khi chạy MinHash/LSH dedup để giảm lineage Spark.

### 3.9. Ghi output

Cuối cùng, pipeline ghi dữ liệu ra Parquet:

```python
result_df.write.mode("overwrite").partitionBy("source").parquet(HDFS_OUT)
```

Output được partition theo `source`, ví dụ:

```text
stg_posts_core/
  source=voz/
  source=vatvo/
  source=vnexpress/
```

Partition theo nguồn giúp dễ kiểm tra dữ liệu, dễ filter theo nguồn và có lợi cho các job downstream khi chỉ cần đọc một số nguồn nhất định.

## 4. Dedup dữ liệu bằng MinHash/LSH

File chính của phần này:

- `algorithms/minhash_dedup.py`

Sau bước cleaning, pipeline cần loại bỏ dữ liệu trùng lặp. Dữ liệu từ crawler có thể bị trùng vì nhiều lý do: cùng một bài được crawl nhiều lần, nội dung được repost, bài viết bị quote lại, hoặc hai nội dung chỉ khác nhau một vài ký tự.

Nếu chỉ dùng `dropDuplicates()` thì pipeline chỉ loại bỏ được các dòng giống hệt nhau. Vì vậy, em tích hợp thêm MinHash/LSH để phát hiện các bản ghi gần trùng lặp.

### 4.1. Vì sao cần MinHash/LSH?

Trong dữ liệu văn bản, hai bài có thể gần giống nhau nhưng không giống hoàn toàn. Ví dụ:

```text
Bài A: iPhone mới ra mắt có pin tốt và camera đẹp
Bài B: iPhone mới ra mắt, pin tốt, camera đẹp
```

Hai câu này không giống 100% theo chuỗi ký tự, nên `dropDuplicates()` không loại được. Tuy nhiên, về mặt nội dung chúng gần như trùng nhau. MinHash/LSH giúp phát hiện những trường hợp như vậy dựa trên độ tương đồng của tập shingles.

### 4.2. Ý tưởng xử lý

Quy trình MinHash/LSH trong pipeline:

1. Lấy text đầu vào, chủ yếu là `segmented_text`.
2. Cắt text thành các shingles có độ dài `k`.
3. Tạo MinHash signature cho mỗi văn bản.
4. Dùng LSH index để tìm nhanh các văn bản có khả năng giống nhau.
5. Gom các record gần trùng thành nhóm.
6. Mỗi nhóm chỉ giữ lại một record đại diện.
7. Join lại với Spark DataFrame gốc để lấy dữ liệu đã dedup.

### 4.3. Shingling

Trong `MinHashDeduplicator`, hàm `_shingling` cắt text thành các đoạn nhỏ có độ dài `k`.

Ví dụ với `k=5`, một chuỗi sẽ được cắt thành nhiều đoạn con liên tiếp. Sau đó các đoạn này tạo thành một tập shingles. Hai văn bản càng giống nhau thì tập shingles của chúng càng có nhiều phần tử chung.

Tham số:

- `k=5`: độ dài mỗi shingle.

Nếu text quá ngắn hoặc rỗng thì không tạo được shingle và record đó được bỏ qua ở bước tạo signature.

### 4.4. MinHash signature

Sau khi có tập shingles, pipeline dùng MinHash để tạo chữ ký đại diện cho văn bản.

Tham số:

- `num_perm=128`: số lượng permutation hash.

`num_perm` càng lớn thì ước lượng độ tương đồng càng ổn định hơn, nhưng chi phí tính toán và bộ nhớ cũng tăng. Trong pipeline, giá trị 128 là mức cân bằng giữa độ chính xác và hiệu năng.

### 4.5. LSH để tìm near-duplicate

Sau khi tạo MinHash signature, pipeline dùng `MinHashLSH` để tìm các văn bản có khả năng giống nhau cao.

Tham số:

- Class mặc định có `threshold=0.80`.
- Trong `cleaning_job.py`, pipeline gọi với `threshold=0.85`.

Điều này có nghĩa là pipeline chỉ loại những bản ghi có độ tương đồng cao, tránh xóa nhầm các bài chỉ hơi giống nhau nhưng thực chất khác nội dung.

### 4.6. Gom nhóm duplicate bằng union-find

Trong `fit_transform`, khi LSH phát hiện các record gần giống nhau, pipeline dùng union-find để gom chúng vào cùng một nhóm duplicate.

Ý tưởng:

- Mỗi record ban đầu là một nhóm riêng.
- Nếu hai record gần trùng, union hai nhóm lại.
- Sau khi xử lý xong, mỗi nhóm duplicate chỉ giữ một đại diện.

Khi chọn đại diện, pipeline ưu tiên record có `created_at` sớm hơn. Điều này giúp giữ lại bản ghi gốc hoặc bản ghi xuất hiện trước.

### 4.7. Join lại với Spark DataFrame

Sau khi xác định danh sách `row_id` cần giữ, pipeline tạo `keep_df`, broadcast DataFrame này và join lại với `working_df` ban đầu:

```python
deduped_df = working_df.join(broadcast(keep_df), "__dedup_row_id", "inner")
```

Cách này giúp Spark chỉ giữ lại các dòng thuộc danh sách đại diện, sau đó bỏ cột kỹ thuật `__dedup_row_id`.

### 4.8. Ý nghĩa của dedup trong hệ thống

Dedup có vai trò quan trọng vì dữ liệu trùng lặp có thể làm sai lệch phân tích:

- Topic modeling có thể bị lệch nếu một nội dung xuất hiện quá nhiều lần.
- Sentiment analysis có thể bị thiên lệch nếu các bài trùng bị tính nhiều lần.
- Trend score có thể bị tăng ảo do dữ liệu duplicate.
- Dashboard có thể hiển thị sai mức độ phổ biến của chủ đề.

Vì vậy, MinHash/LSH giúp chất lượng dữ liệu đầu vào tốt hơn trước khi chuyển sang các bước phân tích tiếp theo.

### 4.9. Hạn chế hiện tại

Implementation hiện tại có bước `collect()` để đưa một số thông tin cần thiết về driver khi chạy dedup. Cách này đơn giản và dễ tích hợp với thư viện `datasketch`, phù hợp với quy mô dữ liệu vừa phải trong phạm vi dự án.

Tuy nhiên, nếu mở rộng lên dữ liệu rất lớn, phần dedup có thể cần tối ưu theo hướng phân tán hơn, ví dụ dùng Spark MLlib MinHashLSH hoặc thiết kế lại để tránh gom dữ liệu về driver.

## 5. Vai trò của output `stg_posts_core` trong pipeline tổng

Kết quả quan trọng nhất của phần M2 là dataset/bảng `stg_posts_core`.

Dataset này đã qua các bước:

- Đọc raw data từ nhiều crawler.
- Chuẩn hóa format.
- Làm sạch text.
- Tạo text phục vụ NLP và topic modeling.
- Loại bỏ record lỗi.
- Loại bỏ dữ liệu trùng và gần trùng.
- Ghi ra Parquet.
- Ingest vào ClickHouse.

`stg_posts_core` là dữ liệu đầu vào cho các phần sau:

- **M3**: đọc `segmented_text` hoặc `topic_text` để chạy LDA, BERTopic và Count-Min Sketch.
- **M4**: đọc dữ liệu lõi để chạy sentiment analysis và crisis detection.
- **M5**: dùng ClickHouse/dbt để join dữ liệu lõi với kết quả NLP và hiển thị dashboard.

Như vậy, M2 đóng vai trò là tầng chuẩn hóa dữ liệu trung tâm. Nếu không có `stg_posts_core`, các nhóm sau sẽ phải tự xử lý nhiều format raw data khác nhau, gây trùng lặp logic và khó đảm bảo tính nhất quán.

## 6. Các file chính trong phần M2

Các file/folder chính mà em phụ trách hoặc tích hợp trực tiếp:

- `docker-compose.yml`
- `Dockerfile.spark`
- `Dockerfile.airflow`
- `env.example`
- `LOCAL_GUIDE.md`
- `ansible/`
- `spark_jobs/cleaning_job.py`
- `algorithms/minhash_dedup.py`
- `dags/processing_dag.py`
- `scripts/hdfs_to_clickhouse.py`
- `warehouse/clickhouse/init_schema.sql`

Trong đó, hai file quan trọng nhất khi giải thích phần xử lý dữ liệu là:

- `spark_jobs/cleaning_job.py`
- `algorithms/minhash_dedup.py`

## 7. Tóm tắt kết quả đạt được

Phần M2 đã hoàn thành các công việc chính sau:

- Dựng nền tảng hạ tầng cho hệ thống Big Data bằng Docker và Ansible.
- Chuẩn bị môi trường chạy Spark, HDFS, Airflow và ClickHouse.
- Xây dựng Spark cleaning pipeline để đọc dữ liệu thô từ nhiều crawler.
- Chuẩn hóa dữ liệu về schema chung `stg_posts_core`.
- Làm sạch text và tạo các cột `body`, `clean_text`, `segmented_text`, `topic_text`.
- Tích hợp dedup bằng `dropDuplicates()` và MinHash/LSH.
- Ghi output ra Parquet và partition theo `source`.
- Cung cấp dữ liệu lõi cho các module M3, M4 và M5.

## 8. Kết luận

Phần em triển khai tập trung vào nền tảng dữ liệu và xử lý lõi của hệ thống. Em xây dựng hạ tầng để pipeline có thể chạy được, đồng thời phát triển Spark cleaning job để chuẩn hóa dữ liệu thô thành `stg_posts_core`. Bên cạnh đó, em tích hợp MinHash/LSH để nâng cao chất lượng dữ liệu bằng cách loại bỏ các bản ghi trùng hoặc gần trùng.

Output `stg_posts_core` là điểm nối quan trọng giữa dữ liệu crawler và các module phân tích phía sau. Nhờ có phần M2, các nhóm M3, M4 và M5 có thể làm việc trên một nguồn dữ liệu sạch, thống nhất và có cấu trúc rõ ràng.

