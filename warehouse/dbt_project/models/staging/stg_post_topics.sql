{{
    config(
        materialized='view'
    )
}}

SELECT
    post_id,
    topic_id,
    topic_probability
FROM {{ source('tech_radar', 'stg_post_topics') }}
