{{
    config(
        materialized='view'
    )
}}

SELECT
    post_id,
    source,
    author_id,
    author_name,
    title,
    body,
    segmented_text,
    parent_id,
    reaction_count,
    comment_count,
    view_count,
    created_at,
    crawled_at
FROM {{ source('tech_radar', 'stg_posts_core') }}
