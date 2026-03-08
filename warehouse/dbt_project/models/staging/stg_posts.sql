{{
    config(
        materialized='view'
    )
}}

SELECT
    post_id,
    source,
    author_id,
    author_name,
    title,
    body,
    segmented_text,
    sentiment_label,
    sentiment_score,
    topic_id,
    engagement,
    created_at,
    loaded_at,

    -- Derived time dimensions
    toDate(created_at)                 AS created_date,
    toStartOfHour(created_at)          AS created_hour,
    toHour(created_at)                 AS hour_of_day,
    toDayOfWeek(created_at)            AS day_of_week,

    -- Derived metrics
    length(body)                       AS body_length

FROM {{ source('dwh_prod', 'stg_posts') }}
