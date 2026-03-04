{{
    config(
        materialized='view'
    )
}}

/*
    Staging layer: interaction edges for PageRank/HITS graph.
    Adds derived date columns and filters self-interactions.
*/

SELECT
    source_author,
    target_author,
    interaction_type,
    weight,
    source,
    created_at,
    toDate(created_at)             AS interaction_date,
    toStartOfHour(created_at)      AS interaction_hour

FROM dwh_prod.stg_interactions
WHERE source_author != target_author   -- Remove self-loops
  AND source_author != ''
  AND target_author != ''
