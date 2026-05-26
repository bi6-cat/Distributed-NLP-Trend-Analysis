# M3 Output Report - B22DCKH096

File này tổng hợp các output chính của M3 trong pipeline topic modeling, sentiment và keyword frequency. Output của M3 nằm ở 3 nơi:

- HDFS: nơi lưu file parquet/model cho pipeline chạy thật.
- ClickHouse: nơi dashboard/dbt đọc dữ liệu.
- Local `output/`: nơi lưu output dev/test, fallback và handoff.

## 1. Tong Quan Output

| Nhóm output | Nơi lưu | Path / Table | Ý nghĩa |
|---|---|---|---|
| LDA topics | HDFS | `/user/root/results/lda/topics/` | Danh sách topic LDA |
| LDA post-topic assignment | HDFS | `/user/root/results/lda/post_topic_assignment/` | Gán topic cho từng post |
| LDA model | HDFS | `/user/root/results/lda/lda_model/` | Model LDA đã train |
| LDA vectorizer | HDFS | `/user/root/results/lda/cv_model/` | CountVectorizerModel để map vector về keyword |
| BERTopic topics | HDFS | `/user/root/results/bertopic/topics/topics.parquet` | Danh sách topic BERTopic |
| BERTopic post-topic assignment | HDFS | `/user/root/results/bertopic/post_topics/post_topic_assignment.parquet` | Gán topic BERTopic cho từng post |
| Silver post topics | HDFS | `/data/silver/post_topics/date=YYYY-MM-DD/` | Layer chuẩn hóa post-topic cho downstream |
| Topic lookup | ClickHouse | `tech_radar.stg_topics` | Topic label, keywords, model version |
| Post-topic mapping | ClickHouse | `tech_radar.stg_post_topics` | Mapping post sang topic |
| Keyword frequency | ClickHouse | `tech_radar.stg_keyword_freq` | Keyword frequency theo source/window |
| Sentiment prediction | ClickHouse | `tech_radar.stg_posts_nlp` | Sentiment PhoBERT cho từng post |
| Handoff summary | Local | `output/handoff/*.csv` | File summary gửi Member 5/dashboard |
| CMS fallback | Local | `output/cms/*.json`, `output/cms/*.pkl` | State và top keyword fallback |

## 2. LDA Output

LDA có nhiều output vì nó vừa sinh kết quả topic, vừa lưu model để dùng lại/debug.

```text
/user/root/results/lda/topics/
```

Danh sách topic LDA. Mỗi dòng là một topic gồm `topic_id`, `label`, `top_keywords`, `coherence_score`, `model_version`, `created_at`.

```text
/user/root/results/lda/post_topic_assignment/
```

Kết quả gán topic cho từng post. Mỗi dòng là một post với topic tốt nhất.

```text
/user/root/results/lda/lda_model/
```

Model LDA Spark đã train. Dùng để load lại model mà không cần train lại từ đầu.

```text
/user/root/results/lda/cv_model/
```

CountVectorizerModel. Đây là từ điển/vectorizer giúp map index trong vector về keyword thật.

Tóm gọn:

| Path | Ý nghĩa |
|---|---|
| `/user/root/results/lda/topics/` | Topic là gì |
| `/user/root/results/lda/post_topic_assignment/` | Post nào thuộc topic nào |
| `/user/root/results/lda/lda_model/` | Model LDA đã train |
| `/user/root/results/lda/cv_model/` | Từ điển/vectorizer để hiểu keyword |

Nếu chỉ phục vụ dashboard thì quan trọng nhất là:

```text
/user/root/results/lda/topics/
/user/root/results/lda/post_topic_assignment/
```

## 3. BERTopic Output

BERTopic inference ghi 2 file chính:

```text
/user/root/results/bertopic/topics/topics.parquet
/user/root/results/bertopic/post_topics/post_topic_assignment.parquet
```

Ý nghĩa:

| Path | Ý nghĩa |
|---|---|
| `/user/root/results/bertopic/topics/topics.parquet` | Danh sách topic BERTopic |
| `/user/root/results/bertopic/post_topics/post_topic_assignment.parquet` | Mapping post sang topic BERTopic |

Local dev nếu chạy local:

```text
output/bertopic_inference/topics.parquet
output/bertopic_inference/post_topic_assignment.parquet
```

## 4. Silver Layer

Silver layer lưu post-topic assignment theo ngày để downstream đọc chung.

```text
/data/silver/post_topics/date=YYYY-MM-DD/part-lda-0.parquet
/data/silver/post_topics/date=YYYY-MM-DD/part-bertopic-0.parquet
```

Ý nghĩa:

- `part-lda-0.parquet`: post-topic assignment từ LDA.
- `part-bertopic-0.parquet`: post-topic assignment từ BERTopic.
- `date=YYYY-MM-DD`: partition theo ngày chạy pipeline.

## 5. ClickHouse Tables

Các bảng ClickHouse là output phục vụ dashboard/dbt.

```text
tech_radar.stg_topics
tech_radar.stg_post_topics
tech_radar.stg_keyword_freq
tech_radar.stg_posts_nlp
```

Ý nghĩa:

| Table | Ý nghĩa |
|---|---|
| `tech_radar.stg_topics` | Topic metadata: label, keyword, model version |
| `tech_radar.stg_post_topics` | Mapping post sang topic |
| `tech_radar.stg_keyword_freq` | Keyword frequency theo source/window |
| `tech_radar.stg_posts_nlp` | Sentiment prediction |

## 6. Local / Handoff Output

Các file local thường dùng cho dev/test hoặc handoff cho Member 5.

```text
output/handoff/topic_summary_for_m5.csv
output/handoff/bertopic_summary_for_m5.csv
output/cms/cms_state.pkl
output/cms/top_keywords.json
```

Ý nghĩa:

| Path | Ý nghĩa |
|---|---|
| `output/handoff/topic_summary_for_m5.csv` | Summary LDA topics cho Member 5 |
| `output/handoff/bertopic_summary_for_m5.csv` | Summary BERTopic topics cho Member 5 |
| `output/cms/cms_state.pkl` | State Count-Min Sketch |
| `output/cms/top_keywords.json` | Top keywords fallback nếu ClickHouse insert lỗi |

## 7. Cach Xem Du Lieu HDFS

List file HDFS:

```powershell
docker compose exec namenode hdfs dfs -ls -R /user/root/results/lda
docker compose exec namenode hdfs dfs -ls -R /user/root/results/bertopic
docker compose exec namenode hdfs dfs -ls -R /data/silver/post_topics
```

Copy parquet từ HDFS về container rồi đọc bằng pandas:

```powershell
docker compose exec namenode hdfs dfs -copyToLocal /user/root/results/bertopic/topics/topics.parquet /tmp/topics.parquet
docker compose exec airflow-scheduler python -c "import pandas as pd; print(pd.read_parquet('/tmp/topics.parquet').head())"
```

Với LDA Spark parquet directory:

```powershell
docker compose exec namenode hdfs dfs -copyToLocal /user/root/results/lda/topics /tmp/lda_topics
docker compose exec airflow-scheduler python -c "import pandas as pd; print(pd.read_parquet('/tmp/lda_topics').head())"
```

Lưu ý: file parquet không xem bằng `cat` như text/json. Nên đọc bằng `pandas`, Spark hoặc công cụ hỗ trợ parquet.

## 8. Cach Xem Du Lieu ClickHouse

Mở UI ClickHouse:

```text
http://localhost:8123/play
```

Login:

```text
User: root
Password: root
Database: tech_radar
```

Show tables:

```sql
SHOW DATABASES;
```

```sql
SHOW TABLES FROM tech_radar;
```

Xem số dòng:

```sql
SELECT count()
FROM tech_radar.stg_topics;
```

```sql
SELECT model_type, count()
FROM tech_radar.stg_post_topics
GROUP BY model_type;
```

```sql
SELECT count()
FROM tech_radar.stg_keyword_freq;
```

```sql
SELECT count()
FROM tech_radar.stg_posts_nlp;
```

Xem dữ liệu mẫu:

```sql
SELECT
    topic_id,
    label,
    top_keywords,
    model_version,
    created_at
FROM tech_radar.stg_topics
LIMIT 20;
```

```sql
SELECT
    post_id,
    topic_id,
    topic_probability,
    model_type,
    predicted_at
FROM tech_radar.stg_post_topics
LIMIT 20;
```

```sql
SELECT
    keyword,
    window_start,
    window_end,
    estimated_count,
    source
FROM tech_radar.stg_keyword_freq
ORDER BY keyword, source
LIMIT 100;
```

```sql
SELECT
    post_id,
    sentiment_label,
    sentiment_score,
    model_version,
    predicted_at
FROM tech_radar.stg_posts_nlp
LIMIT 20;
```

Xem riêng LDA và BERTopic:

```sql
SELECT *
FROM tech_radar.stg_post_topics
WHERE model_type = 'lda'
LIMIT 20;
```

```sql
SELECT *
FROM tech_radar.stg_post_topics
WHERE model_type = 'bertopic'
LIMIT 20;
```

Join post-topic với topic label:

```sql
SELECT
    p.post_id,
    p.model_type,
    p.topic_id,
    t.label,
    t.top_keywords,
    p.topic_probability
FROM tech_radar.stg_post_topics AS p
LEFT JOIN tech_radar.stg_topics AS t
    ON p.topic_id = t.topic_id
ORDER BY p.predicted_at DESC
LIMIT 20;
```

Check duplicate CMS keyword/source/window:

```sql
SELECT
    keyword,
    source,
    window_start,
    window_end,
    estimated_count,
    count() AS n
FROM tech_radar.stg_keyword_freq
GROUP BY
    keyword,
    source,
    window_start,
    window_end,
    estimated_count
HAVING n > 1
ORDER BY n DESC
LIMIT 20;
```

Check keyword bị lặp vì rerun batch khác `window_end`:

```sql
SELECT
    keyword,
    source,
    estimated_count,
    count() AS n,
    min(window_end) AS first_window_end,
    max(window_end) AS last_window_end
FROM tech_radar.stg_keyword_freq
GROUP BY
    keyword,
    source,
    estimated_count
HAVING n > 1
ORDER BY n DESC, keyword
LIMIT 20;
```

## 9. Cach Xem Du Lieu Local

List file local:

```powershell
Get-ChildItem -Recurse output
```

Đọc parquet local:

```powershell
python -c "import pandas as pd; print(pd.read_parquet('output/bertopic_inference/topics.parquet').head())"
python -c "import pandas as pd; print(pd.read_parquet('output/bertopic_inference/post_topic_assignment.parquet').head())"
```

Đọc CSV handoff:

```powershell
python -c "import pandas as pd; print(pd.read_csv('output/handoff/topic_summary_for_m5.csv').head())"
python -c "import pandas as pd; print(pd.read_csv('output/handoff/bertopic_summary_for_m5.csv').head())"
```

## 10. Y Nghia Truong Du Lieu

### `stg_topics` / `topics.parquet`

| Trường | Ý nghĩa |
|---|---|
| `topic_id` | ID topic do model sinh ra |
| `label` | Tên topic, thường ghép từ top keywords |
| `top_keywords` | Danh sách keyword đại diện topic |
| `coherence_score` | Điểm coherence nếu có; BERTopic inference thường để null |
| `model_version` | Version model, ví dụ `lda_v1`, `bertopic_v1` |
| `created_at` | Thời điểm tạo topic output |

### `stg_post_topics` / `post_topic_assignment.parquet`

| Trường | Ý nghĩa |
|---|---|
| `post_id` | ID bài viết/comment |
| `topic_id` | Topic được gán cho post |
| `topic_probability` | Độ tin cậy/xác suất post thuộc topic |
| `model_type` | Loại model: `lda` hoặc `bertopic` |
| `predicted_at` | Thời điểm model gán topic |
| `loaded_at` | Thời điểm insert vào ClickHouse |

Lưu ý: `stg_post_topics` phải lưu theo key logic `(post_id, model_type)`. Cùng một `post_id` có thể có 2 dòng: một dòng từ LDA và một dòng từ BERTopic. Không được chỉ key theo `post_id`, vì BERTopic sẽ đè mất LDA hoặc ngược lại.

### `stg_keyword_freq`

| Trường | Ý nghĩa |
|---|---|
| `keyword` | Từ khóa |
| `window_start` | Thời điểm bắt đầu cửa sổ thống kê |
| `window_end` | Thời điểm kết thúc cửa sổ thống kê |
| `estimated_count` | Count ước lượng từ Count-Min Sketch |
| `source` | Nguồn dữ liệu: `voz`, `tinhte`, `vnexpress`, `youtube` |

### `stg_posts_nlp`

| Trường | Ý nghĩa |
|---|---|
| `post_id` | ID bài viết/comment |
| `sentiment_label` | Sentiment label: `positive`, `negative`, `neutral` |
| `sentiment_score` | Confidence của PhoBERT |
| `model_version` | Version sentiment model |
| `predicted_at` | Thời điểm predict sentiment |
| `loaded_at` | Thời điểm insert vào ClickHouse |

## 11. Tom Tat Nhanh

Nếu muốn xem output thật cho dashboard:

```text
ClickHouse:
- tech_radar.stg_topics
- tech_radar.stg_post_topics
- tech_radar.stg_keyword_freq
- tech_radar.stg_posts_nlp
```

Nếu muốn debug pipeline/model:

```text
HDFS:
- /user/root/results/lda/
- /user/root/results/bertopic/
- /data/silver/post_topics/
```

Nếu muốn xem file dev/handoff:

```text
Local:
- output/handoff/
- output/cms/
- output/bertopic_inference/
- output/lda/
```
