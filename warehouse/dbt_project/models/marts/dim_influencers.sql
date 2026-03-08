{{
    config(
        materialized='table'
    )
}}

SELECT
    i.author_id,
    i.source,
    i.author_name,
    -- Graph-based authority scores
    i.pagerank_score,
    i.hub_score,
    i.authority_score,

    -- Rank within each source
    row_number() OVER (
        PARTITION BY i.source
        ORDER BY i.pagerank_score DESC
    )                                       AS pagerank_rank,

    -- Activity metrics from intermediate model
    coalesce(a.total_posts, 0)              AS total_posts,
    coalesce(a.avg_engagement, 0)           AS avg_engagement,
    coalesce(a.total_engagement, 0)         AS total_engagement,
    coalesce(a.positive_posts, 0)           AS positive_posts,
    coalesce(a.negative_posts, 0)           AS negative_posts,
    coalesce(a.distinct_topics, 0)          AS distinct_topics,
    a.first_seen,
    a.last_seen,
    coalesce(a.active_days, 0)              AS active_days,

    i.computed_at

FROM dwh_prod.stg_influencers AS i
LEFT JOIN {{ ref('int_author_metrics') }} AS a
    ON i.author_id = a.author_id
   AND i.source = a.source
