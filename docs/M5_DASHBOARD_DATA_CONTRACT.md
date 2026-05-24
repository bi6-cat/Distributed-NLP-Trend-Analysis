# M5 Dashboard Data Contract

Muc tieu cua file nay la chot mot contract du lieu on dinh cho Member 5 de lam dashboard, tu HDFS den ClickHouse, va uu tien nhung bang da duoc chuan hoa cho UI.

## 1. Nguyen tac dung du lieu cho dashboard

- Dashboard khong query truc tiep HDFS.
- Dashboard uu tien query 3 bang cuoi trong ClickHouse:
  - `tech_radar.fct_topic_activity`
  - `tech_radar.fct_crisis_events`
  - `tech_radar.dim_topics`
- `stg_*` chi dung cho debug, drill-through, hoac kiem tra pipeline.
- Mot `post_id` chi nen co 1 topic assignment tot nhat trong `stg_post_topics`.
- `fct_topic_activity` la bang fact rong, da embed `topic_label` va `source_type`, khong can join them cho man hinh chinh.

## 2. Duong di du lieu end-to-end

### 2.1 HDFS / local staged input

Nguon chung sau cleaning:

- HDFS: `hdfs://namenode:9000/user/<HDFS_USER>/staged/stg_posts_core`
- Local bind path: `/opt/airflow/data/preprocessed/stg_posts_core`

Y nghia:

- Day la output cua Spark cleaning job.
- Day la nguon dau vao chung cho sentiment, topic modeling, CMS keyword, crisis detection.
- Co the co them `clean_text` trong parquet/HDFS de phuc vu ML, nhung `clean_text` khong phai contract ClickHouse cho dashboard.

Cac cot staging quan trong o HDFS:

- `post_id`
- `source`
- `author`
- `title`
- `body`
- `segmented_text`
- `parent_id`
- `reaction_count`
- `comment_count`
- `view_count`
- `created_at`
- `crawled_at`

## 3. ClickHouse staging layer

### 3.1 `tech_radar.stg_posts_core`

Muc dich:

- Bang core sau khi ingest tu HDFS vao ClickHouse.
- Chua du lieu post/comment va engagement atomic.

Cot co the lay:

- `post_id`: khoa chinh xuyen suot pipeline
- `source`: `voz`, `tinhte`, `vnexpress`, `youtube`
- `author`
- `title`
- `body`
- `segmented_text`
- `parent_id`: `NULL` neu la post goc, co gia tri neu la comment
- `reaction_count`
- `comment_count`
- `view_count`
- `created_at`
- `crawled_at`
- `loaded_at`

Dung khi nao:

- Debug raw/staging
- Drill-through tu dashboard ve bai viet goc
- Kiem tra volume theo source

Khong dung lam fact dashboard chinh.

### 3.2 `tech_radar.stg_posts_nlp`

Muc dich:

- Ket qua sentiment cua M4.

Cot:

- `post_id`
- `sentiment_label`: `positive`, `negative`, `neutral`
- `sentiment_score`
- `model_version`
- `predicted_at`
- `loaded_at`

Dung khi nao:

- Debug sentiment
- Kiem tra quality output model

Khong query truc tiep cho chart chinh neu da co `fct_topic_activity`.

### 3.3 `tech_radar.stg_post_topics`

Muc dich:

- Topic assignment tot nhat cho moi `post_id`.

Cot:

- `post_id`
- `topic_id`
- `topic_probability`
- `model_type`: `lda` hoac `bertopic`
- `predicted_at`
- `loaded_at`

Contract dashboard-first:

- M5 coi day la 1 dong / `post_id`.
- Neu staging co duplicate cu, dbt `stg_posts` da tu lay row moi nhat.

### 3.4 `tech_radar.stg_topics`

Muc dich:

- Bang tra cuu topic.

Cot:

- `topic_id`
- `label`
- `top_keywords`
- `coherence_score`
- `model_version`
- `created_at`

Dung khi nao:

- Debug labels
- Kiem tra keyword cua tung topic

### 3.5 `tech_radar.stg_keyword_freq`

Muc dich:

- Top-K keyword theo cua so thoi gian va theo source.

Cot:

- `keyword`
- `window_start`
- `window_end`
- `estimated_count`
- `source`

Dung khi nao:

- Neu dashboard co module keyword monitoring / top keyword per source

### 3.6 `tech_radar.stg_crisis_events`

Muc dich:

- Dau vao anomaly/crisis cho mart cuoi.

Cot:

- `event_id`
- `detected_at`
- `severity`
- `anomaly_score`
- `trigger_conditions` as `Array(String)`
- `affected_topics` as `Array(Int32)`
- `neg_ratio`
- `mention_velocity`
- `evidence_post_ids` as `Array(String)`

Dung khi nao:

- Debug crisis pipeline
- Drill-through cho event

### 3.7 `tech_radar.stg_posts`

Muc dich:

- View staging hop nhat `core + sentiment + topic`.
- Da duoc dbt chuan hoa de lay topic/sentiment moi nhat theo `post_id`.

Cot:

- `post_id`
- `source`
- `author`
- `title`
- `body`
- `segmented_text`
- `parent_id`
- `reaction_count`
- `comment_count`
- `view_count`
- `engagement`
- `sentiment_label`
- `sentiment_score`
- `topic_id`
- `topic_probability`
- `source_type`
- `created_at`
- `created_hour`
- `created_date`
- `crawled_at`

Dung khi nao:

- Drill-through / explore record-level
- Man hinh detail theo bai viet

Khong nen dung de ve chart aggregate chinh neu da co mart.

## 4. ClickHouse mart layer: day la lop M5 nen dung

### 4.1 `tech_radar.fct_topic_activity`

Day la bang chinh cho Trend Explorer, leaderboard, timeseries, sentiment-by-topic.

Grain:

- `topic_id x source x hour_bucket`

Cot can dung:

- Keys:
  - `topic_id`
  - `source`
  - `hour_bucket`
  - `bucket_date`
- Embedded dimensions:
  - `source_type`
  - `topic_label`
- Volume:
  - `mention_count`
  - `unique_authors`
  - `engagement_sum`
- Atomic sums:
  - `reaction_sum`
  - `comment_sum`
  - `view_sum`
- Trend:
  - `velocity`
  - `acceleration`
  - `engagement_normalized`
  - `trend_score`
  - `trend_rank`
- Sentiment:
  - `pos_count`
  - `neg_count`
  - `neu_count`
  - `neg_ratio`
- Baselines/anomaly:
  - `neg_ratio_24h_avg`
  - `mention_7d_avg`
  - `mention_7d_stddev`
  - `z_score_neg_ratio`
  - `volume_zscore`
- Metadata:
  - `computed_at`

Dashboard use cases:

- Top trending topics
- Topic timeseries by hour/day
- Source filter: forum/news/video
- Sentiment stacked chart
- Volume anomaly / negative sentiment anomaly

### 4.2 `tech_radar.fct_crisis_events`

Day la bang chinh cho Crisis Monitor.

Cot:

- `event_id`
- `detected_at`
- `detected_date`
- `severity`
- `anomaly_score`
- `trigger_conditions`
- `neg_ratio`
- `mention_velocity`
- `evidence_post_ids`
- `affected_topics`
- `affected_topic_labels`
- `severity_rank`

Dashboard use cases:

- Bang danh sach crisis event
- Filter theo `severity`
- Card tong hop event 24h / 7d
- Event detail drawer

### 4.3 `tech_radar.dim_topics`

Day la bang dimension cho topic selector, lookup, metadata card.

Cot:

- `topic_id`
- `label`
- `top_keywords`
- `coherence_score`
- `model_version`
- `total_mentions`
- `first_seen`
- `last_seen`

Dashboard use cases:

- Selectbox / multiselect topic
- Topic metadata panel
- Sort topic theo total mentions

## 5. Mapping man hinh dashboard -> bang du lieu

### 5.1 Topic selector / filter panel

Dung:

- `dim_topics.topic_id`
- `dim_topics.label`
- `dim_topics.top_keywords`
- `dim_topics.total_mentions`

### 5.2 Top trending topics

Dung:

- `fct_topic_activity`

Goi y:

- Filter theo `bucket_date` hoac `hour_bucket`
- Sort theo `trend_score DESC`
- Co the them `source` hoac `source_type`

### 5.3 Trend timeseries

Dung:

- `fct_topic_activity.hour_bucket`
- `fct_topic_activity.mention_count`
- `fct_topic_activity.trend_score`

### 5.4 Sentiment by topic

Dung:

- `fct_topic_activity.pos_count`
- `fct_topic_activity.neg_count`
- `fct_topic_activity.neu_count`
- `fct_topic_activity.neg_ratio`

### 5.5 Crisis monitor

Dung:

- `fct_crisis_events`

### 5.6 Keyword panel neu co

Dung:

- `stg_keyword_freq`

Luu y:

- Day la bang staging/phu tro, chua co mart rieng.

### 5.7 Drill-through post level

Dung:

- `stg_posts`
- Neu can text goc/record-level detail thi join logic khong can tu dashboard, chi query thang `stg_posts` theo `topic_id`, `source`, `created_date`

## 6. Thu tu uu tien query cho M5

1. `fct_topic_activity`
2. `fct_crisis_events`
3. `dim_topics`
4. `stg_keyword_freq` neu lam keyword module
5. `stg_posts` neu lam detail drawer / record explorer

Khong nen dung truc tiep:

- `stg_posts_core`
- `stg_posts_nlp`
- `stg_post_topics`
- `stg_topics`

tru khi dang debug pipeline.

## 7. Query mau cho M5

### 7.1 Topic activity trong 30 ngay

```sql
SELECT *
FROM tech_radar.fct_topic_activity
WHERE bucket_date >= today() - 30
ORDER BY hour_bucket DESC, trend_score DESC
```

### 7.2 Topic selector

```sql
SELECT *
FROM tech_radar.dim_topics
ORDER BY total_mentions DESC
```

### 7.3 Crisis 7 ngay

```sql
SELECT *
FROM tech_radar.fct_crisis_events
WHERE detected_at >= now() - INTERVAL 168 HOUR
ORDER BY detected_at DESC
```

### 7.4 Drill-through bai viet theo topic

```sql
SELECT
    post_id,
    source,
    author,
    title,
    body,
    sentiment_label,
    topic_id,
    topic_probability,
    engagement,
    created_at
FROM tech_radar.stg_posts
WHERE topic_id = 10
ORDER BY created_at DESC
LIMIT 100
```

## 8. Luu y de M5 khong bi query nham

- DB dung cho dashboard that la `tech_radar`.
- Neu app dang tro vao DB khac nhu `dwh_prod` thi se khong thay du lieu pipeline that.
- ClickHouse `/play` khong cho multi-statement; chay tung query mot.
- Neu `fct_topic_activity` rong, kiem tra:
  - `stg_posts` co row `topic_id != 0` hay khong
  - filter thoi gian 90 ngay trong dbt staging
- Neu `fct_crisis_events` rong, kiem tra:
  - `stg_crisis_events` co row hay khong

## 9. Chot contract cho M5

Neu chi can mot danh sach ngan de code:

- Trend page: `fct_topic_activity`
- Crisis page: `fct_crisis_events`
- Topic filter / lookup: `dim_topics`
- Detail posts: `stg_posts`
- Optional keyword widget: `stg_keyword_freq`

Do day la lop da duoc chuan hoa phu hop nhat de keo ra dashboard.
