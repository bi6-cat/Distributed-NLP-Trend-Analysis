-- ClickHouse ingestion from HDFS Silver to tech_radar staging.
--
-- This file is meant to be rendered by Airflow/Jinja.
-- Required parameters:
--   {{ ds }}                 Airflow logical date, YYYY-MM-DD
--   {{ params.next_ds }}      Next logical date, YYYY-MM-DD
--   {{ params.hdfs_base }}    Example: hdfs://namenode:9000
--
-- Paths:
--   {{ params.hdfs_base }}/data/silver/posts_core/date={{ ds }}/*.parquet
--   {{ params.hdfs_base }}/data/silver/posts_nlp/date={{ ds }}/*.parquet
--   {{ params.hdfs_base }}/data/silver/post_topics/date={{ ds }}/*.parquet
--   {{ params.hdfs_base }}/data/silver/topics/date={{ ds }}/*.parquet
--   {{ params.hdfs_base }}/data/silver/keyword_freq/date={{ ds }}/*.parquet
--   {{ params.hdfs_base }}/data/silver/crisis_events/date={{ ds }}/*.parquet

-- ---------------------------------------------------------------------------
-- 1. stg_posts_core, MergeTree: force idempotency by deleting the logical day.
-- ---------------------------------------------------------------------------
ALTER TABLE tech_radar.stg_posts_core
    DELETE WHERE created_at >= toDateTime('{{ ds }} 00:00:00')
      AND created_at <  toDateTime('{{ params.next_ds }} 00:00:00')
SETTINGS mutations_sync = 1;

INSERT INTO tech_radar.stg_posts_core
(
    post_id,
    source,
    author_id,
    author_name,
    title,
    body,
    segmented_text,
    parent_id,
    reaction_count,
    comment_count,
    view_count,
    created_at,
    crawled_at,
    loaded_at
)
SELECT
    post_id,
    source,
    CAST(NULL AS Nullable(String)) AS author_id,
    ifNull(author, 'unknown') AS author_name,
    title,
    ifNull(body, '') AS body,
    ifNull(segmented_text, '') AS segmented_text,
    parent_id,
    reaction_count,
    comment_count,
    view_count,
    created_at,
    crawled_at,
    now() AS loaded_at
FROM hdfs(
    '{{ params.hdfs_base }}/data/silver/posts_core/date={{ ds }}/*.parquet',
    'Parquet',
    'post_id String,
     source String,
     author Nullable(String),
     title Nullable(String),
     body Nullable(String),
     clean_text Nullable(String),
     segmented_text Nullable(String),
     parent_id Nullable(String),
     reaction_count Nullable(Int32),
     view_count Nullable(Int32),
     comment_count Nullable(Int32),
     created_at DateTime,
     crawled_at DateTime'
);

-- ---------------------------------------------------------------------------
-- 2. stg_posts_nlp, ReplacingMergeTree(loaded_at): reruns insert newer versions.
-- ---------------------------------------------------------------------------
INSERT INTO tech_radar.stg_posts_nlp
(
    post_id,
    sentiment_label,
    sentiment_score,
    model_version,
    predicted_at,
    loaded_at
)
SELECT
    post_id,
    sentiment_label,
    sentiment_score,
    model_version,
    predicted_at,
    now() AS loaded_at
FROM hdfs(
    '{{ params.hdfs_base }}/data/silver/posts_nlp/date={{ ds }}/*.parquet',
    'Parquet',
    'post_id String,
     sentiment_label String,
     sentiment_score Float32,
     model_version String,
     predicted_at DateTime'
);

-- ---------------------------------------------------------------------------
-- 3. stg_post_topics, ReplacingMergeTree(loaded_at).
-- ---------------------------------------------------------------------------
INSERT INTO tech_radar.stg_post_topics
(
    post_id,
    topic_id,
    topic_probability,
    model_type,
    predicted_at,
    loaded_at
)
SELECT
    post_id,
    topic_id,
    topic_probability,
    model_type,
    predicted_at,
    now() AS loaded_at
FROM hdfs(
    '{{ params.hdfs_base }}/data/silver/post_topics/date={{ ds }}/*.parquet',
    'Parquet',
    'post_id String,
     topic_id Int32,
     topic_probability Float32,
     model_type String,
     predicted_at DateTime'
);

-- ---------------------------------------------------------------------------
-- 4. stg_topics, ReplacingMergeTree(created_at). Usually daily for LDA,
-- weekly for BERTopic; the Airflow sensor can be soft-failed if absent.
-- ---------------------------------------------------------------------------
INSERT INTO tech_radar.stg_topics
(
    topic_id,
    label,
    top_keywords,
    coherence_score,
    model_version,
    created_at
)
SELECT
    topic_id,
    label,
    top_keywords,
    coherence_score,
    model_version,
    created_at
FROM hdfs(
    '{{ params.hdfs_base }}/data/silver/topics/date={{ ds }}/*.parquet',
    'Parquet',
    'topic_id Int32,
     label String,
     top_keywords Array(String),
     coherence_score Nullable(Float32),
     model_version String,
     created_at DateTime'
);

-- ---------------------------------------------------------------------------
-- 5. stg_keyword_freq, MergeTree: delete the logical day before append.
-- ---------------------------------------------------------------------------
ALTER TABLE tech_radar.stg_keyword_freq
    DELETE WHERE window_start >= toDateTime('{{ ds }} 00:00:00')
      AND window_start <  toDateTime('{{ params.next_ds }} 00:00:00')
SETTINGS mutations_sync = 1;

INSERT INTO tech_radar.stg_keyword_freq
(
    keyword,
    window_start,
    window_end,
    estimated_count,
    source
)
SELECT
    keyword,
    window_start,
    window_end,
    estimated_count,
    source
FROM hdfs(
    '{{ params.hdfs_base }}/data/silver/keyword_freq/date={{ ds }}/*.parquet',
    'Parquet',
    'keyword String,
     window_start DateTime,
     window_end DateTime,
     estimated_count Int64,
     source String'
);

-- ---------------------------------------------------------------------------
-- 6. stg_crisis_events, MergeTree: delete the logical day before append.
-- ---------------------------------------------------------------------------
ALTER TABLE tech_radar.stg_crisis_events
    DELETE WHERE detected_at >= toDateTime('{{ ds }} 00:00:00')
      AND detected_at <  toDateTime('{{ params.next_ds }} 00:00:00')
SETTINGS mutations_sync = 1;

INSERT INTO tech_radar.stg_crisis_events
(
    event_id,
    detected_at,
    severity,
    anomaly_score,
    trigger_conditions,
    affected_topics,
    neg_ratio,
    mention_velocity,
    evidence_post_ids
)
SELECT
    event_id,
    detected_at,
    severity,
    anomaly_score,
    trigger_conditions,
    affected_topics,
    neg_ratio,
    mention_velocity,
    evidence_post_ids
FROM hdfs(
    '{{ params.hdfs_base }}/data/silver/crisis_events/date={{ ds }}/*.parquet',
    'Parquet',
    'event_id String,
     detected_at DateTime,
     severity String,
     anomaly_score Float64,
     trigger_conditions Array(String),
     affected_topics Array(Int32),
     neg_ratio Float32,
     mention_velocity Float32,
     evidence_post_ids Array(String)'
);
