{{
    config(
        materialized='table'
    )
}}

/*
    FACT: Topic Activity (Hourly)

    Design rationale (ClickHouse OLAP):
      - Merges the previous fct_topic_trends_hourly AND
        fct_sentiment_timeseries into ONE table.
      - Both had the same grain (topic_id × source × hour_bucket),
        so joining them at query time was redundant overhead.
      - dim_sources (only 4 rows) is embedded as a CASE expression
        to eliminate that JOIN entirely.
      - Topic label is embedded at write time from stg_topics.
      - Result: dashboard queries for Trending Topics and Sentiment
        Explorer need ZERO joins on this table.

    Grain: topic_id × source × hour_bucket

    Trend Score formula:
        TrendScore(t) = α·V(t) + β·A(t) + γ·E(t)
          V = Velocity (mentions/hour)        α = 0.40
          A = Acceleration (ΔV/Δt)            β = 0.30
          E = Normalized engagement           γ = 0.30
*/

WITH hourly AS (
    SELECT *
    FROM {{ ref('int_topic_sentiment_hourly') }}
),

-- Velocity + Acceleration via window lag
with_velocity AS (
    SELECT
        *,
        mention_count AS velocity,
        mention_count - lagInFrame(mention_count, 1, 0)
            OVER (PARTITION BY topic_id, source ORDER BY hour_bucket)
            AS acceleration
    FROM hourly
),

-- Min-max normalise engagement within the same calendar day
with_engagement_norm AS (
    SELECT
        *,
        CASE
            WHEN (max(engagement_sum) OVER (PARTITION BY bucket_date)
                - min(engagement_sum) OVER (PARTITION BY bucket_date)) = 0
            THEN 0.0
            ELSE (engagement_sum - min(engagement_sum) OVER (PARTITION BY bucket_date))
                / (max(engagement_sum) OVER (PARTITION BY bucket_date)
                 - min(engagement_sum) OVER (PARTITION BY bucket_date))
        END AS engagement_normalized
    FROM with_velocity
),

-- 7-day rolling stats for Z-Score anomaly detection
with_rolling AS (
    SELECT
        *,
        avg(neg_ratio) OVER (
            PARTITION BY topic_id, source
            ORDER BY hour_bucket
            ROWS BETWEEN 24 PRECEDING AND CURRENT ROW
        ) AS neg_ratio_24h_avg,

        avg(mention_count) OVER (
            PARTITION BY topic_id, source
            ORDER BY hour_bucket
            ROWS BETWEEN 168 PRECEDING AND CURRENT ROW
        ) AS mention_7d_avg,

        stddevPop(mention_count) OVER (
            PARTITION BY topic_id, source
            ORDER BY hour_bucket
            ROWS BETWEEN 168 PRECEDING AND CURRENT ROW
        ) AS mention_7d_stddev

    FROM with_engagement_norm
)

SELECT
    -- Keys
    topic_id,
    source,
    hour_bucket,
    bucket_date,

    -- Embed source type inline — eliminates dim_sources JOIN
    multiIf(
        source = 'voz',       'forum',
        source = 'tinhte',    'forum',
        source = 'vnexpress', 'news',
        source = 'youtube',   'video',
        'unknown'
    ) AS source_type,

    -- Topic label embedded at write time — eliminates dim_topics JOIN
    topic_label,

    -- Volume
    mention_count,
    unique_authors,
    engagement_sum,

    -- Trend components
    velocity,
    acceleration,
    engagement_normalized,

    -- Trend Score
    {{ trend_score_calc('velocity', 'acceleration', 'engagement_normalized') }}
        AS trend_score,

    -- Rank within this hour (for Top-N queries)
    row_number() OVER (
        PARTITION BY hour_bucket
        ORDER BY {{ trend_score_calc('velocity', 'acceleration', 'engagement_normalized') }} DESC
    ) AS trend_rank,

    -- Sentiment distribution
    pos_count,
    neg_count,
    neu_count,
    neg_ratio,
    pos_ratio,
    avg_sentiment_confidence,

    -- Rolling stats for crisis detection / Sentiment Explorer
    neg_ratio_24h_avg,
    mention_7d_avg,
    mention_7d_stddev,

    -- Z-Score: standard deviations from 7-day rolling mean
    CASE
        WHEN mention_7d_stddev > 0
        THEN (mention_count - mention_7d_avg) / mention_7d_stddev
        ELSE 0.0
    END AS volume_zscore,

    now() AS computed_at

FROM with_rolling
ORDER BY hour_bucket DESC, trend_score DESC
