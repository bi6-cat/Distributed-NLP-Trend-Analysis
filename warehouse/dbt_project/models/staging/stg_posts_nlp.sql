{{
    config(
        materialized='view'
    )
}}

SELECT
    post_id,
    sentiment_label,
    sentiment_score
FROM {{ source('tech_radar', 'stg_posts_nlp') }}
