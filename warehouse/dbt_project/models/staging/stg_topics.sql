{{
    config(
        materialized='view'
    )
}}

SELECT
    topic_id,
    label,
    top_keywords,
    coherence_score,
    model_version,
    created_at
FROM {{ source('tech_radar', 'stg_topics') }}
FINAL
WHERE startsWith(model_version, 'bertopic')
  AND label NOT IN (
    'phim_chế_độ_máy',
    'chuyên_nghiệp_chuyên_nghiệp chuyên_nghiệp_dòng chuyên_nghiệp'
  )
