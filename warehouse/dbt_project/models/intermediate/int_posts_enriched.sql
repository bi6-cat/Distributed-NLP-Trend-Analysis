{{
    config(
        materialized='view'
    )
}}

-- depends_on: {{ ref('stg_posts_core') }}
-- depends_on: {{ ref('stg_posts_nlp') }}
-- depends_on: {{ ref('stg_post_topics') }}

SELECT
    c.post_id           AS post_id,
    c.source            AS source,
    c.author_id         AS author_id,
    c.author_name       AS author_name,
    c.title             AS title,
    c.body              AS body,
    c.segmented_text    AS segmented_text,
    c.parent_id         AS parent_id,

    c.reaction_count    AS reaction_count,
    c.comment_count     AS comment_count,
    c.view_count        AS view_count,

    -- Weights are dbt vars so scoring can change without rerunning Spark.
    (coalesce(c.reaction_count, 0) * {{ var('weight_reaction') }} + coalesce(c.comment_count, 0) * {{ var('weight_comment') }}
     + coalesce(c.view_count, 0) * {{ var('weight_view') }}) AS engagement,

    coalesce(n.sentiment_label, 'neutral') AS sentiment_label,
    coalesce(n.sentiment_score, 0.0) AS sentiment_score,

    coalesce(t.topic_id, 0) AS topic_id,
    coalesce(t.topic_probability, 0.0) AS topic_probability,

    multiIf(
        c.source = 'voz',       'forum',
        c.source = 'tinhte',    'forum',
        c.source = 'vnexpress', 'news',
        c.source = 'youtube',   'video',
        c.source = 'vatvo',     'forum',
        'unknown'
    ) AS source_type,

    c.created_at        AS created_at,
    toStartOfHour(c.created_at) AS created_hour,
    toDate(c.created_at)        AS created_date,
    c.crawled_at        AS crawled_at

FROM {{ ref('stg_posts_core') }}       AS c
LEFT JOIN {{ ref('stg_posts_nlp') }}    AS n ON c.post_id = n.post_id
LEFT JOIN {{ ref('stg_post_topics') }}  AS t ON c.post_id = t.post_id

WHERE c.body != ''
  AND c.created_at >= today() - INTERVAL 90 DAY
