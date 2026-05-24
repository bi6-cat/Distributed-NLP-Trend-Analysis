{{
    config(
        materialized='view'
    )
}}

WITH latest_sentiment AS (
    SELECT
        post_id,
        sentiment_label,
        sentiment_score
    FROM (
        SELECT
            post_id,
            sentiment_label,
            sentiment_score,
            row_number() OVER (
                PARTITION BY post_id
                ORDER BY predicted_at DESC, loaded_at DESC
            ) AS rn
        FROM {{ source('tech_radar', 'stg_posts_nlp') }}
    )
    WHERE rn = 1
),
latest_topics AS (
    SELECT
        post_id,
        topic_id,
        topic_probability
    FROM (
        SELECT
            post_id,
            topic_id,
            topic_probability,
            row_number() OVER (
                PARTITION BY post_id
                ORDER BY predicted_at DESC, loaded_at DESC, topic_probability DESC
            ) AS rn
        FROM {{ source('tech_radar', 'stg_post_topics') }}
    )
    WHERE rn = 1
)

SELECT
    c.post_id           AS post_id,
    c.source            AS source,
    c.author            AS author,
    c.title             AS title,
    c.body              AS body,
    c.segmented_text    AS segmented_text,
    c.parent_id         AS parent_id,

    -- METRICS
    c.reaction_count    AS reaction_count,
    c.comment_count     AS comment_count,
    c.view_count        AS view_count,

    -- ENGAGEMENT
    (c.reaction_count + c.comment_count * 2
     + coalesce(c.view_count, 0) / 100) AS engagement,

    -- NLP LABELS
    coalesce(n.sentiment_label, 'neutral') AS sentiment_label,
    coalesce(n.sentiment_score, 0.0) AS sentiment_score,

    -- TOPIC
    coalesce(t.topic_id, 0) AS topic_id,
    coalesce(t.topic_probability, 0.0) AS topic_probability,

    -- SOURCE TYPE
    multiIf(
        c.source = 'voz',       'forum',
        c.source = 'tinhte',    'forum',
        c.source = 'vnexpress', 'news',
        c.source = 'youtube',   'video',
        'unknown'
    ) AS source_type,

    -- COMPUTED TIME COLUMNS
    c.created_at        AS created_at,
    toStartOfHour(c.created_at) AS created_hour,
    toDate(c.created_at)        AS created_date,
    c.crawled_at        AS crawled_at

FROM {{ source('tech_radar', 'stg_posts_core') }}       AS c
LEFT JOIN latest_sentiment                               AS n ON c.post_id = n.post_id
LEFT JOIN latest_topics                                  AS t ON c.post_id = t.post_id

WHERE c.body != ''
  AND c.created_at >= today() - INTERVAL 90 DAY
