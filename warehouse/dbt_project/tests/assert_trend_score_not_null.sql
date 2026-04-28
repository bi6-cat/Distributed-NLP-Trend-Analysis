SELECT
    topic_id,
    source,
    hour_bucket
FROM {{ ref('fct_topic_activity') }}
WHERE trend_score IS NULL
