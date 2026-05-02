{{
    config(
        materialized='table'
    )
}}

SELECT
    t.topic_id,
    t.label,
    t.top_keywords,
    t.coherence_score,
    t.model_version,

    coalesce(sum(h.mention_count), 0)                   AS total_mentions,
    min(h.hour_bucket)                                  AS first_seen,
    max(h.hour_bucket)                                  AS last_seen

FROM {{ source('tech_radar', 'stg_topics') }}           AS t
LEFT JOIN {{ ref('int_topic_sentiment_hourly') }}       AS h
    ON t.topic_id = h.topic_id
GROUP BY
    t.topic_id,
    t.label,
    t.top_keywords,
    t.coherence_score,
    t.model_version