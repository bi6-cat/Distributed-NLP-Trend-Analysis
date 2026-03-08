{{
    config(
        materialized='table'
    )
}}

SELECT
    s.topic_id,
    t.label                                         AS topic_label,
    s.source,
    toStartOfHour(s.created_at)                     AS hour_bucket,
    toDate(s.created_at)                            AS bucket_date,

    count(*)                                        AS mention_count,
    sum(s.engagement)                               AS engagement_sum,
    uniqExact(s.author_id)                          AS unique_authors,

    countIf(s.sentiment_label = 'positive')         AS pos_count,
    countIf(s.sentiment_label = 'negative')         AS neg_count,
    countIf(s.sentiment_label = 'neutral')          AS neu_count,

    countIf(s.sentiment_label = 'negative')
        / greatest(count(*), 1)                     AS neg_ratio,
    countIf(s.sentiment_label = 'positive')
        / greatest(count(*), 1)                     AS pos_ratio,

    avg(s.sentiment_score)                          AS avg_sentiment_confidence

FROM {{ ref('stg_posts') }}            AS s
LEFT JOIN dwh_prod.stg_topics          AS t
    ON s.topic_id = t.topic_id
WHERE s.topic_id != 0
  AND s.body != ''
  AND s.created_at >= today() - INTERVAL 90 DAY
GROUP BY
    s.topic_id,
    t.label,
    s.source,
    hour_bucket,
    bucket_date
