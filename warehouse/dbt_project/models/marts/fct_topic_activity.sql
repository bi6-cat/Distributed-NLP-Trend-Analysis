{{
    config(
        materialized='table',
        engine='MergeTree()',
        partition_by='toYYYYMM(bucket_date)',
        order_by=['hour_bucket', 'source_type', 'trend_score']
    )
}}

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

-- Min-max normalise engagement
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

-- Rolling stats for anomaly detection + crisis thresholds
with_rolling AS (
    SELECT
        *,
        -- 24h rolling average of neg_ratio
        avg(neg_ratio) OVER (
            PARTITION BY topic_id, source
            ORDER BY hour_bucket
            ROWS BETWEEN 24 PRECEDING AND CURRENT ROW
        ) AS neg_ratio_24h_avg,

        -- 24h rolling stddev of neg_ratio
        stddevPop(neg_ratio) OVER (
            PARTITION BY topic_id, source
            ORDER BY hour_bucket
            ROWS BETWEEN 24 PRECEDING AND CURRENT ROW
        ) AS neg_ratio_24h_stddev,

        -- 7-day rolling average of mention_count
        avg(mention_count) OVER (
            PARTITION BY topic_id, source
            ORDER BY hour_bucket
            ROWS BETWEEN 168 PRECEDING AND CURRENT ROW
        ) AS mention_7d_avg,

        -- 7-day rolling stddev of mention_count
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

    -- Embedded dimensions
    source_type,
    topic_label,

    -- Volume metrics
    mention_count,
    unique_authors,
    engagement_sum,

    -- Atomic metric sums
    reaction_sum,
    comment_sum,
    view_sum,

    -- Trend components
    velocity,
    acceleration,
    engagement_normalized,

    -- Trend Score
    {{ trend_score_calc('velocity', 'acceleration', 'engagement_normalized') }}
        AS trend_score,

    -- Rank within this hour
    row_number() OVER (
        PARTITION BY hour_bucket
        ORDER BY {{ trend_score_calc('velocity', 'acceleration', 'engagement_normalized') }} DESC
    ) AS trend_rank,

    -- Sentiment distribution
    pos_count,
    neg_count,
    neu_count,
    neg_ratio,

    -- Rolling baselines for crisis detection / Sentiment Explorer
    neg_ratio_24h_avg,
    mention_7d_avg,
    mention_7d_stddev,

    -- Z-Score: neg_ratio anomaly signal
    CASE
        WHEN neg_ratio_24h_stddev > 0
        THEN (neg_ratio - neg_ratio_24h_avg) / neg_ratio_24h_stddev
        ELSE 0.0
    END AS z_score_neg_ratio,

    -- Z-Score: volume anomaly signal
    CASE
        WHEN mention_7d_stddev > 0
        THEN (mention_count - mention_7d_avg) / mention_7d_stddev
        ELSE 0.0
    END AS volume_zscore,

    now() AS computed_at

FROM with_rolling
ORDER BY hour_bucket DESC, trend_score DESC
