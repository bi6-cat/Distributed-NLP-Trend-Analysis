{{
    config(
        materialized='table',
        engine='MergeTree()',
        order_by=['detected_date', 'event_id']
    )
}}

WITH exploded_topics AS (
    -- Split array of topic IDs into separate rows
    SELECT
        event_id,
        arrayJoin(affected_topics) AS tid
    FROM {{ source('tech_radar', 'stg_crisis_events') }}
),

with_labels AS (
    -- Join with topics table to get topic labels
    SELECT
        ex.event_id,
        ex.tid,
        coalesce(t.label, concat('topic_', toString(ex.tid))) AS tid_label
    FROM exploded_topics AS ex
    LEFT JOIN {{ source('tech_radar', 'stg_topics') }} AS t 
        ON ex.tid = t.topic_id
),

collapsed_labels AS (
    -- Collapse topic labels back into an array per event
    SELECT
        event_id,
        groupArray(tid_label) AS affected_topic_labels
    FROM with_labels
    GROUP BY event_id
)

SELECT
    e.event_id,
    e.detected_at,
    toDate(e.detected_at) AS detected_date,
    e.severity,
    e.anomaly_score,
    e.trigger_conditions,
    e.neg_ratio,
    e.mention_velocity,
    e.evidence_post_ids,
    e.affected_topics,

    c.affected_topic_labels,

    -- Rank for sorting on Dashboard
    multiIf(
        e.severity = 'HIGH',   3,
        e.severity = 'MEDIUM', 2,
        e.severity = 'LOW',    1,
        0
    ) AS severity_rank

FROM {{ source('tech_radar', 'stg_crisis_events') }} AS e
LEFT JOIN collapsed_labels AS c
    ON e.event_id = c.event_id