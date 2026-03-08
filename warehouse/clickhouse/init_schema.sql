CREATE DATABASE IF NOT EXISTS dwh_prod;

-- 1. stg_posts: Flattened, deduplicated, sentiment-labeled documents
CREATE TABLE IF NOT EXISTS dwh_prod.stg_posts (
    post_id         String,
    source          LowCardinality(String),
    author_id       String,
    author_name     String,
    title           String DEFAULT '',
    body            String,
    segmented_text  String,
    sentiment_label LowCardinality(String),
    sentiment_score Float32,
    topic_id        UInt32 DEFAULT 0,
    engagement      UInt32 DEFAULT 0,
    created_at      DateTime,
    loaded_at       DateTime DEFAULT now()
) ENGINE = ReplacingMergeTree(loaded_at)
PARTITION BY toYYYYMM(created_at)
ORDER BY (topic_id, created_at, source, post_id);

-- 2. stg_interactions: Reply/quote edges for PageRank graph
CREATE TABLE IF NOT EXISTS dwh_prod.stg_interactions (
    source_author_id    String,                  
    target_author_id    String,                  
    interaction_type LowCardinality(String),
    weight           UInt32 DEFAULT 1,
    source           LowCardinality(String),  
    created_at       DateTime
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(created_at)
ORDER BY (source_author_id, target_author_id, created_at);

-- 3. stg_topics: Topic labels from LDA/BERTopic
CREATE TABLE IF NOT EXISTS dwh_prod.stg_topics (
    topic_id        UInt32,
    label           String,
    top_keywords    Array(String),
    coherence_score Float32 DEFAULT 0.0,
    model_version   String,
    created_at      DateTime DEFAULT now()
) ENGINE = ReplacingMergeTree(created_at)
ORDER BY (topic_id, model_version);

-- 4. stg_influencers: PageRank/HITS scores per author
CREATE TABLE IF NOT EXISTS dwh_prod.stg_influencers (
    author_id          String,
    author_name        String,
    source          LowCardinality(String),
    pagerank_score  Float32,
    hub_score       Float32,
    authority_score Float32,
    total_posts     UInt32,
    total_replies   UInt32,
    computed_at     DateTime DEFAULT now()
) ENGINE = ReplacingMergeTree(computed_at)
ORDER BY (author_id, source);

-- 5. stg_keyword_freq: Count-Min Sketch keyword frequencies
CREATE TABLE IF NOT EXISTS dwh_prod.stg_keyword_freq (
    keyword         String,
    window_start    DateTime,
    window_end      DateTime,
    estimated_count UInt64,
    source          LowCardinality(String)
) ENGINE = ReplacingMergeTree(window_start)
PARTITION BY toYYYYMM(window_start)
ORDER BY (keyword, window_start);

-- 6. stg_crisis_events: Anomaly detection output
CREATE TABLE IF NOT EXISTS dwh_prod.stg_crisis_events (
    event_id           String,
    detected_at        DateTime,
    severity           LowCardinality(String),
    anomaly_score      Float32,
    trigger_conditions Array(String),
    affected_topics    Array(UInt32),
    neg_ratio          Float32,
    mention_velocity   Float32,
    evidence_doc_ids   Array(String)
) ENGINE = ReplacingMergeTree(detected_at)
ORDER BY (detected_at, event_id);