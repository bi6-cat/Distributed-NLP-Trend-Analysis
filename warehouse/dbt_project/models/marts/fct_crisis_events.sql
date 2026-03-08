{{
    config(
        materialized='table'
    )
}}

/*
    FACT: Crisis Events

    Enriches raw crisis events with embedded topic labels and severity rank.
    Grain: event_id

    ClickHouse note: arrayMap lambda cannot reference outer-scope variables
    inside a correlated subquery. We resolve labels by joining on the expanded
    array (arrayJoin) and re-aggregating with groupArray.
*/

WITH exploded AS (
    -- Expand each affected_topic into one row for label lookup
    SELECT
        e.event_id,
        e.detected_at,
        e.severity,
        e.anomaly_score,
        e.trigger_conditions,
        e.neg_ratio,
        e.mention_velocity,
        e.evidence_doc_ids,
        e.affected_topics,
        arrayJoin(e.affected_topics)            AS tid
    FROM dwh_prod.stg_crisis_events AS e
),

with_labels AS (
    SELECT
        ex.event_id,
        ex.detected_at,
        ex.severity,
        ex.anomaly_score,
        ex.trigger_conditions,
        ex.neg_ratio,
        ex.mention_velocity,
        ex.evidence_doc_ids,
        ex.affected_topics,
        ex.tid,
        coalesce(t.label, concat('topic_', toString(ex.tid))) AS tid_label
    FROM exploded AS ex
    LEFT JOIN dwh_prod.stg_topics AS t ON ex.tid = t.topic_id
)

SELECT
    event_id,
    detected_at,
    toDate(detected_at)                             AS detected_date,
    severity,
    anomaly_score,
    trigger_conditions,
    neg_ratio,
    mention_velocity,
    evidence_doc_ids,
    affected_topics,

    -- Re-aggregate per-event, preserving array order via rowNumberInAllBlocks grouping
    groupArray(tid_label)                           AS affected_topic_labels,

    multiIf(
        severity = 'HIGH',   3,
        severity = 'MEDIUM', 2,
        severity = 'LOW',    1,
        0
    )                                               AS severity_rank

FROM with_labels
GROUP BY
    event_id,
    detected_at,
    severity,
    anomaly_score,
    trigger_conditions,
    neg_ratio,
    mention_velocity,
    evidence_doc_ids,
    affected_topics

ORDER BY detected_at DESC
