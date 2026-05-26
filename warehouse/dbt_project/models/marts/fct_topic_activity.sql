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

-- Compare each hourly bucket with the previous bucket per topic/source.
with_velocity AS (
    SELECT
        *,
        mention_count AS velocity,
        mention_count - lagInFrame(mention_count, 1, 0)
            OVER (PARTITION BY topic_id, source ORDER BY hour_bucket)
            AS acceleration
    FROM hourly
),

-- Avoid division by zero when every topic has the same daily engagement.
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

-- Rolling baselines feed anomaly and crisis scoring.
with_rolling AS (
    SELECT
        *,
        avg(neg_ratio) OVER (
            PARTITION BY topic_id, source
            ORDER BY hour_bucket
            ROWS BETWEEN 24 PRECEDING AND CURRENT ROW
        ) AS neg_ratio_24h_avg,

        stddevPop(neg_ratio) OVER (
            PARTITION BY topic_id, source
            ORDER BY hour_bucket
            ROWS BETWEEN 24 PRECEDING AND CURRENT ROW
        ) AS neg_ratio_24h_stddev,

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
    topic_id,
    source,
    hour_bucket,
    bucket_date,

    source_type,
    topic_label,

    mention_count,
    unique_authors,
    engagement_sum,

    reaction_sum,
    comment_sum,
    view_sum,

    velocity,
    acceleration,
    engagement_normalized,

    {{ trend_score_calc('velocity', 'acceleration', 'engagement_normalized') }}
        AS trend_score,

    -- Rank is scoped to each hour so spikes are comparable over time.
    row_number() OVER (
        PARTITION BY hour_bucket
        ORDER BY {{ trend_score_calc('velocity', 'acceleration', 'engagement_normalized') }} DESC
    ) AS trend_rank,

    pos_count,
    neg_count,
    neu_count,
    neg_ratio,

    neg_ratio_24h_avg,
    mention_7d_avg,
    mention_7d_stddev,

    -- Z-scores are zeroed until enough variance exists.
    CASE
        WHEN neg_ratio_24h_stddev > 0
        THEN (neg_ratio - neg_ratio_24h_avg) / neg_ratio_24h_stddev
        ELSE 0.0
    END AS z_score_neg_ratio,

    CASE
        WHEN mention_7d_stddev > 0
        THEN (mention_count - mention_7d_avg) / mention_7d_stddev
        ELSE 0.0
    END AS volume_zscore,

    now() AS computed_at

FROM with_rolling
ORDER BY hour_bucket DESC, trend_score DESC
