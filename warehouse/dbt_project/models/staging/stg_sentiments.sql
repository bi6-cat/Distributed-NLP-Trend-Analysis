{{
    config(
        materialized='view'
    )
}}

SELECT
    post_id,
    source,
    author_id,
    topic_id,
    created_at,
    toStartOfHour(created_at)                       AS hour_bucket,
    toDate(created_at)                              AS created_date,
    sentiment_label,
    sentiment_score,

    -- One-hot encoded sentiment for SUM() aggregations
    if(sentiment_label = 'positive', 1, 0)          AS is_positive,
    if(sentiment_label = 'negative', 1, 0)          AS is_negative,
    if(sentiment_label = 'neutral',  1, 0)          AS is_neutral,

    engagement

FROM {{ source('dwh_prod', 'stg_posts') }}
WHERE sentiment_label != ''
