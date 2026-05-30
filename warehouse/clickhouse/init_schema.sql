CREATE DATABASE IF NOT EXISTS tech_radar;

CREATE TABLE IF NOT EXISTS tech_radar.stg_posts_core (
    post_id         String,
    source          LowCardinality(String),
    author_id       Nullable(String),
    author_name     String,
    title           Nullable(String),
    body            String,
    segmented_text  String,
    parent_id       Nullable(String),
    reaction_count  Nullable(Int32),
    comment_count   Nullable(Int32),
    view_count      Nullable(Int32),
    created_at      DateTime,
    crawled_at      DateTime,
    loaded_at       DateTime DEFAULT now()
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(created_at)
ORDER BY (source, created_at, post_id)
TTL created_at + INTERVAL 1 YEAR;

CREATE TABLE IF NOT EXISTS tech_radar.stg_posts_nlp (
    post_id         String,
    sentiment_label LowCardinality(String),
    sentiment_score Float32,
    model_version   LowCardinality(String),
    predicted_at    DateTime,
    loaded_at       DateTime DEFAULT now()
) ENGINE = ReplacingMergeTree(loaded_at)
PARTITION BY toYYYYMM(predicted_at)
ORDER BY (post_id);

CREATE TABLE IF NOT EXISTS tech_radar.stg_post_topics (
    post_id           String,
    topic_id          Int32,
    topic_probability Float32,
    model_type        LowCardinality(String),
    predicted_at      DateTime,
    loaded_at         DateTime DEFAULT now()
) ENGINE = ReplacingMergeTree(loaded_at)
ORDER BY (post_id);

CREATE TABLE IF NOT EXISTS tech_radar.stg_topics (
    topic_id        Int32,
    label           String,
    top_keywords    Array(String),
    coherence_score Nullable(Float32),
    model_version   String,
    created_at      DateTime DEFAULT now()
) ENGINE = ReplacingMergeTree(created_at)
ORDER BY (topic_id, model_version);

CREATE TABLE IF NOT EXISTS tech_radar.stg_keyword_freq (
    keyword         String,
    window_start    DateTime,
    window_end      DateTime,
    estimated_count Int64,
    source          LowCardinality(String)
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(window_start)
ORDER BY (keyword, window_start);

CREATE TABLE IF NOT EXISTS tech_radar.stg_crisis_events (
    event_id           String,
    detected_at        DateTime,
    severity           LowCardinality(String),
    anomaly_score      Float64,
    trigger_conditions Array(String),
    affected_topics    Array(Int32),
    neg_ratio          Float32,
    mention_velocity   Float32,
    evidence_post_ids  Array(String)
) ENGINE = MergeTree()
ORDER BY (detected_at, event_id);
