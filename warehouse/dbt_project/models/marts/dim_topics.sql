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

    -- Aggregate from the pre-aggregated intermediate table (avoids stg_posts view recursion)
    coalesce(sum(h.mention_count), 0)                           AS total_mentions,
    coalesce(max(h.unique_authors), 0)                          AS unique_authors,
    toNullable(min(h.bucket_date))                              AS first_seen,
    toNullable(max(h.bucket_date))                              AS last_seen,
    dateDiff('day',
        coalesce(min(h.bucket_date), today()),
        coalesce(max(h.bucket_date), today())
    )                                                           AS active_days

FROM dwh_prod.stg_topics AS t
LEFT JOIN {{ ref('int_topic_sentiment_hourly') }} AS h
    ON t.topic_id = h.topic_id
GROUP BY
    t.topic_id,
    t.label,
    t.top_keywords,
    t.coherence_score,
    t.model_version
