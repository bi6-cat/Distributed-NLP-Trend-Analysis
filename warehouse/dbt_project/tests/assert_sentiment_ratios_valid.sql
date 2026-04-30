SELECT
    topic_id,
    source,
    hour_bucket,
    pos_count,
    neg_count,
    neu_count,
    mention_count
FROM {{ ref('fct_topic_activity') }}
WHERE (pos_count + neg_count + neu_count) != mention_count
