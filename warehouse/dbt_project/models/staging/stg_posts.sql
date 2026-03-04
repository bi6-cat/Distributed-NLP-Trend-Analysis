{{
    config(
        materialized='view'
    )
}}

/*
    Staging layer: light cleaning & computed columns on top of raw stg_posts.
    Filters out empty bodies and limits to the last 90 days for performance.
*/

WITH source AS (
    SELECT *
    FROM dwh_prod.stg_posts
)

SELECT
    doc_id,
    source,
    author,
    coalesce(title, '')                AS title,
    body,
    segmented_text,
    parent_id,
    sentiment_label,
    sentiment_score,
    topic_id,
    engagement,
    created_at,
    crawled_at,
    loaded_at,

    -- Derived time dimensions
    toDate(created_at)                 AS created_date,
    toStartOfHour(created_at)          AS created_hour,
    toHour(created_at)                 AS hour_of_day,
    toDayOfWeek(created_at)            AS day_of_week,

    -- Derived flags
    if(parent_id IS NOT NULL, 1, 0)    AS is_reply,
    length(body)                       AS body_length

FROM source
WHERE body != ''
  AND created_at >= today() - INTERVAL 90 DAY
