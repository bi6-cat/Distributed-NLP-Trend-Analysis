{{
    config(
        materialized='view'
    )
}}

SELECT
    post_id,
    topic_id,
    topic_probability,
    model_type,
    predicted_at
FROM {{ source('tech_radar', 'stg_post_topics') }}
FINAL
WHERE model_type = 'bertopic'
