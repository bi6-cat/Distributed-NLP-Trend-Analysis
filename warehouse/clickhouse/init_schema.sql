CREATE DATABASE IF NOT EXISTS tech_radar;

CREATE USER IF NOT EXISTS app IDENTIFIED WITH no_password;
GRANT ALL ON tech_radar.* TO app;

-- 1. stg_posts_core
CREATE TABLE IF NOT EXISTS tech_radar.stg_posts_core (
    post_id         String,                  -- Canonical PK across all stages
    source          LowCardinality(String),  -- 'voz','tinhte','vnexpress','youtube'
    author          String,
    title           Nullable(String),
    body            String,
    segmented_text  String,                  -- VnCoreNLP word-segmented output
    parent_id       Nullable(String),        -- NULL if top-level post
    reaction_count  Int32 DEFAULT 0,         -- Atomic: likes/reactions only
    comment_count   Int32 DEFAULT 0,         -- Atomic: direct replies count
    view_count      Nullable(Int32),         -- Atomic: YouTube views; NULL for others
    created_at      DateTime,
    crawled_at      DateTime,
    loaded_at       DateTime DEFAULT now()
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(created_at)
ORDER BY (source, created_at, post_id)
TTL created_at + INTERVAL 1 YEAR;

-- 2. stg_posts_nlp
CREATE TABLE IF NOT EXISTS tech_radar.stg_posts_nlp (
    post_id         String,                  -- FK → stg_posts_core.post_id
    sentiment_label LowCardinality(String),  -- 'positive','negative','neutral'
    sentiment_score Float32,                 -- Model confidence 0.0–1.0
    model_version   LowCardinality(String),  -- e.g., 'phobert_v1.2'
    predicted_at    DateTime,
    loaded_at       DateTime DEFAULT now()
) ENGINE = ReplacingMergeTree(loaded_at)
ORDER BY (post_id);

-- 3. stg_post_topics
CREATE TABLE IF NOT EXISTS tech_radar.stg_post_topics (
    post_id           String,                -- FK → stg_posts_core.post_id
    topic_id          Int32,                 -- LDA/BERTopic assignment
    topic_probability Float32,               -- Assignment confidence 0.0–1.0
    model_type        LowCardinality(String), -- 'lda' | 'bertopic'
    predicted_at      DateTime,
    loaded_at         DateTime DEFAULT now()
) ENGINE = ReplacingMergeTree(loaded_at)
ORDER BY (post_id, model_type);

-- 4. stg_topics
CREATE TABLE IF NOT EXISTS tech_radar.stg_topics (
    topic_id        Int32,
    label           String,
    top_keywords    Array(String),
    coherence_score Nullable(Float32),
    model_version   String,
    created_at      DateTime DEFAULT now()
) ENGINE = ReplacingMergeTree(created_at)
ORDER BY (topic_id, model_version);

-- 5. stg_keyword_freq
CREATE TABLE IF NOT EXISTS tech_radar.stg_keyword_freq (
    keyword         String,
    window_start    DateTime,
    window_end      DateTime,
    estimated_count Int64,
    source          LowCardinality(String)
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(window_start)
ORDER BY (keyword, window_start);

-- 6. stg_crisis_events
CREATE TABLE IF NOT EXISTS tech_radar.stg_crisis_events (
    event_id           String,
    detected_at        DateTime,
    severity           LowCardinality(String),  -- 'LOW','MEDIUM','HIGH'
    anomaly_score      Float64,
    trigger_conditions Array(String),
    affected_topics    Array(Int32),
    neg_ratio          Float32,
    mention_velocity   Float32,
    evidence_post_ids  Array(String)            -- References stg_posts_core.post_id
) ENGINE = MergeTree()
ORDER BY (detected_at, event_id);

-- 7. stg_crisis_hourly
CREATE TABLE IF NOT EXISTS tech_radar.stg_crisis_hourly (
    date          Date,
    hour          UInt8,
    comment_count UInt32,
    z_score       Nullable(Float32),
    global_spike  UInt8,
    if_spike      UInt8,
    is_spike      UInt8,
    is_crisis     UInt8,
    neg_ratio     Nullable(Float32),
    neg_score_avg Nullable(Float32)
) ENGINE = MergeTree()
ORDER BY (date, hour);

-- 8. hourly_baseline
CREATE TABLE IF NOT EXISTS tech_radar.hourly_baseline (
    hour            UInt8,
    baseline_median Float32,
    baseline_std    Float32
) ENGINE = ReplacingMergeTree()
ORDER BY hour;
