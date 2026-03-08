{{
    config(
        materialized='table'
    )
}}

SELECT
    author_id,
    source,
    count(*)                                        AS total_posts,
    avg(engagement)                                 AS avg_engagement,
    max(engagement)                                 AS max_engagement,
    sum(engagement)                                 AS total_engagement,

    countIf(sentiment_label = 'positive')           AS positive_posts,
    countIf(sentiment_label = 'negative')           AS negative_posts,
    countIf(sentiment_label = 'neutral')            AS neutral_posts,

    min(created_at)                                 AS first_seen,
    max(created_at)                                 AS last_seen,
    dateDiff('day', min(created_at), max(created_at)) AS active_days,

    uniqExactIf(topic_id, topic_id != 0)            AS distinct_topics

FROM {{ ref('stg_posts') }}
GROUP BY author_id, source
