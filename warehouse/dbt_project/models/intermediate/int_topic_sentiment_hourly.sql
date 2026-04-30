{{
    config(
        materialized='view'
    )
}}

SELECT
    s.topic_id                                      AS topic_id,
    t.label                                         AS topic_label,
    s.source                                        AS source,
    s.source_type                                   AS source_type,
    toStartOfHour(s.created_at)                     AS hour_bucket,
    toDate(s.created_at)                            AS bucket_date,

    -- Volume metrics
    count(*)                                        AS mention_count,
    sum(s.engagement)                               AS engagement_sum,
    uniqExact(s.author_id)                          AS unique_authors,

    -- Atomic metric sums
    sum(s.reaction_count)                           AS reaction_sum,
    sum(s.comment_count)                            AS comment_sum,
    sum(coalesce(s.view_count, 0))                  AS view_sum,

    -- Sentiment distribution
    countIf(s.sentiment_label = 'positive')         AS pos_count,
    countIf(s.sentiment_label = 'negative')         AS neg_count,
    countIf(s.sentiment_label = 'neutral')          AS neu_count,

    -- Derived sentiment ratios
    countIf(s.sentiment_label = 'negative')
        / greatest(count(*), 1)                     AS neg_ratio

FROM (
    SELECT
        topic_id, source, source_type, created_at,
        engagement, author_id, reaction_count, comment_count,
        view_count, sentiment_label
    FROM {{ ref('stg_posts') }}
) AS s
LEFT JOIN {{ source('tech_radar', 'stg_topics') }}  AS t
    ON s.topic_id = t.topic_id
WHERE s.topic_id IS NOT NULL
  AND s.topic_id != 0
GROUP BY
    s.topic_id,
    t.label,
    s.source,
    s.source_type,
    hour_bucket,
    bucket_date
