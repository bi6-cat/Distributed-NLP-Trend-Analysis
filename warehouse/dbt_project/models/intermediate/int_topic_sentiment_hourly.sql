{{
    config(
        materialized='view'
    )
}}

-- depends_on: {{ ref('int_posts_enriched') }}

SELECT
    s.topic_id                                      AS topic_id,
    t.label                                         AS topic_label,
    s.source                                        AS source,
    s.source_type                                   AS source_type,
    toStartOfHour(s.created_at)                     AS hour_bucket,
    toDate(s.created_at)                            AS bucket_date,

    count(*)                                        AS mention_count,
    sum(s.engagement)                               AS engagement_sum,
    uniqExact(s.author_id)                          AS unique_authors,

    sum(coalesce(s.reaction_count, 0))              AS reaction_sum,
    sum(coalesce(s.comment_count, 0))               AS comment_sum,
    sum(coalesce(s.view_count, 0))                  AS view_sum,

    countIf(s.sentiment_label = 'positive')         AS pos_count,
    countIf(s.sentiment_label = 'negative')         AS neg_count,
    countIf(s.sentiment_label = 'neutral')          AS neu_count,

    -- Guard sparse buckets from divide-by-zero.
    countIf(s.sentiment_label = 'negative')
        / greatest(count(*), 1)                     AS neg_ratio

FROM (
    SELECT
        topic_id, source, source_type, created_at,
        engagement, author_id, reaction_count, comment_count,
        view_count, sentiment_label
    FROM {{ ref('int_posts_enriched') }}
) AS s
LEFT JOIN {{ source('tech_radar', 'stg_topics') }} AS t FINAL
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
