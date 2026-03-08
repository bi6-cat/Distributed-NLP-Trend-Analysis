{{
    config(
        materialized='view'
    )
}}

SELECT
    source_author_id,
    target_author_id,
    interaction_type,
    weight,
    source,
    created_at,
    toDate(created_at)                 AS interaction_date,
    toStartOfHour(created_at)          AS interaction_hour

FROM {{ source('dwh_prod', 'stg_interactions') }}
WHERE source_author_id != target_author_id
  AND source_author_id != ''
  AND target_author_id != ''
