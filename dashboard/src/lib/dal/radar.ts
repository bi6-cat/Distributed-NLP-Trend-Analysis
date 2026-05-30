import "server-only";
import { queryClickhouse } from "@/lib/clickhouse";
import type { ResolvedTimeRange } from "@/lib/time-range";

const SCHEMA = process.env.CLICKHOUSE_DATABASE ?? "tech_radar";
const POSTS_ENRICHED_TABLE = `${SCHEMA}.dbt_int_posts_enriched`;

export type OverviewKpis = {
  daily_mentions: number;
  yesterday_mentions: number;
  active_crises: number;
  mention_delta_pct: number;
};

type OverviewMentionsRow = {
  daily_mentions?: number | string;
  yesterday_mentions?: number | string;
};

type OverviewCrisesRow = {
  active_crises?: number | string;
};

export type TrendingTopic = {
  id: number;
  label: string;
  score: number;
  volume: number;
  delta: number;
};

export type SentimentBreakdown = {
  positive: number;
  neutral: number;
  negative: number;
};

export type ActiveTopic = {
  id: number;
  label: string;
  first_seen: string;
  last_seen: string;
  mentions: number;
  score: number;
};

export type TrendPoint = {
  time: string;
  score: number;
};

export type SentimentPoint = {
  time: string;
  pos: number;
  neu: number;
  neg: number;
};

export type TopicEvidencePost = {
  id: string;
  author_name: string;
  source_type: string;
  sentiment: string;
  body: string;
  engagement: number;
};

export type CrisisEvent = {
  event_id: string;
  severity: "HIGH" | "MEDIUM" | "LOW";
  detected_at: string;
  time: string;
  affected_topics: string[];
  trigger_conditions: string[];
  neg_ratio: number;
  mention_velocity: string;
  anomaly_score: number;
  evidence_posts: {
    id: string;
    author: string;
    text: string;
    type: string;
    engagement: number;
  }[];
  severityRank: number;
};

export type CrisisStats = {
  total_24h: number;
  high_severity_count: number;
  avg_velocity: number;
};

function timeParams(timeRange: ResolvedTimeRange) {
  return {
    start: timeRange.start,
    end: timeRange.end,
  };
}

export async function getOverviewKPIsFromCH(timeRange: ResolvedTimeRange): Promise<OverviewKpis> {
  const queryMentions = `
      WITH
        toDateTime({start: String}) AS range_start,
        toDateTime({end: String}) AS range_end,
        dateDiff('second', range_start, range_end) + 1 AS range_seconds
      SELECT 
        sumIf(mention_count, hour_bucket >= range_start AND hour_bucket <= range_end) AS daily_mentions,
        sumIf(
          mention_count,
          hour_bucket >= range_start - toIntervalSecond(range_seconds)
            AND hour_bucket < range_start
        ) AS yesterday_mentions
      FROM ${SCHEMA}.dbt_fct_topic_activity
      WHERE hour_bucket >= range_start - toIntervalSecond(range_seconds)
        AND hour_bucket <= range_end
    `;

  const queryCrises = `
      WITH
        toDateTime({start: String}) AS range_start,
        toDateTime({end: String}) AS range_end
      SELECT count(DISTINCT event_id) AS active_crises 
      FROM ${SCHEMA}.dbt_fct_crisis_events 
      WHERE severity = 'HIGH'
        AND parseDateTimeBestEffortOrNull(toString(detected_at)) >= range_start
        AND parseDateTimeBestEffortOrNull(toString(detected_at)) <= range_end
    `;

  const params = timeParams(timeRange);
  const [mentionsRows, crisesRows] = await Promise.all([
    queryClickhouse<OverviewMentionsRow>(queryMentions, params),
    queryClickhouse<OverviewCrisesRow>(queryCrises, params)
  ]);

  const daily = Number(mentionsRows[0]?.daily_mentions) || 0;
  const yesterday = Number(mentionsRows[0]?.yesterday_mentions) || 0;
  const activeCrises = Number(crisesRows[0]?.active_crises) || 0;

  let deltaPct = 0;
  if (yesterday > 0) {
    deltaPct = ((daily - yesterday) / yesterday) * 100;
  } else if (daily > 0) {
    deltaPct = 100;
  }

  return {
    daily_mentions: daily,
    yesterday_mentions: yesterday,
    active_crises: activeCrises,
    mention_delta_pct: Number(deltaPct.toFixed(2)),
  };
}

export async function getTrendingTopicsFromCH(timeRange: ResolvedTimeRange): Promise<TrendingTopic[]> {
  const query = `
      WITH
        toDateTime({start: String}) AS range_start,
        toDateTime({end: String}) AS range_end
      SELECT
        topic_id AS id,
        topic_label AS label,
        max(trend_score) AS score,
        sum(mention_count) AS volume,
        sum(acceleration) AS delta
      FROM ${SCHEMA}.dbt_fct_topic_activity
      WHERE hour_bucket >= range_start
        AND hour_bucket <= range_end
      GROUP BY topic_id, topic_label
      ORDER BY score DESC
      LIMIT 10
    `;
  const rows = await queryClickhouse<Partial<TrendingTopic>>(query, timeParams(timeRange));
  return rows.map((row) => ({
    id: Number(row.id) || 0,
    label: String(row.label ?? ""),
    score: Number(row.score) || 0,
    volume: Number(row.volume) || 0,
    delta: Number(row.delta) || 0,
  }));
}

export async function getOverallSentimentFromCH(timeRange: ResolvedTimeRange): Promise<SentimentBreakdown> {
  const query = `
      WITH
        toDateTime({start: String}) AS range_start,
        toDateTime({end: String}) AS range_end
      SELECT
        sum(pos_count) AS positive,
        sum(neu_count) AS neutral,
        sum(neg_count) AS negative
      FROM ${SCHEMA}.dbt_fct_topic_activity
      WHERE hour_bucket >= range_start
        AND hour_bucket <= range_end
    `;
  const rows = await queryClickhouse<Partial<SentimentBreakdown>>(query, timeParams(timeRange));
  const row = rows[0] ?? {};
  return {
    positive: Number(row.positive) || 0,
    neutral: Number(row.neutral) || 0,
    negative: Number(row.negative) || 0,
  };
}

export async function getActiveTopicsFromCH(timeRange: ResolvedTimeRange): Promise<ActiveTopic[]> {
  const query = `
      WITH
        toDateTime({start: String}) AS range_start,
        toDateTime({end: String}) AS range_end,
        ranged_activity AS (
        SELECT
          topic_id,
          anyLast(topic_label) AS topic_label,
          min(hour_bucket) AS first_seen,
          max(hour_bucket) AS last_seen,
          sum(mention_count) AS mentions,
          max(trend_score) AS score
        FROM ${SCHEMA}.dbt_fct_topic_activity
        WHERE hour_bucket >= range_start
          AND hour_bucket <= range_end
        GROUP BY topic_id
      )
      SELECT
        ra.topic_id AS id,
        ifNull(t.label, ra.topic_label) AS label,
        formatDateTime(ra.first_seen, '%b %d') AS first_seen,
        formatDateTime(ra.last_seen, '%b %d') AS last_seen,
        ra.mentions AS mentions,
        ra.score AS score
      FROM ranged_activity ra
      LEFT JOIN ${SCHEMA}.dbt_dim_topics t ON t.topic_id = ra.topic_id
      ORDER BY score DESC, mentions DESC
      LIMIT 50
    `;
  const rows = await queryClickhouse<Partial<ActiveTopic>>(query, timeParams(timeRange));
  return rows.map((row) => ({
    id: Number(row.id) || 0,
    label: String(row.label ?? ""),
    first_seen: String(row.first_seen ?? "-"),
    last_seen: String(row.last_seen ?? "-"),
    mentions: Number(row.mentions) || 0,
    score: Number(row.score) || 0,
  }));
}

export async function getTopicTrendScoreFromCH(
  topicId: number,
  timeRange: ResolvedTimeRange,
): Promise<TrendPoint[]> {
  const query = `
    WITH
      toDateTime({start: String}) AS range_start,
      toDateTime({end: String}) AS range_end
    SELECT
      formatDateTime(hour_bucket, '%m-%d %H:%M') AS time,
      max(trend_score) AS score
    FROM ${SCHEMA}.dbt_fct_topic_activity
    WHERE topic_id = {topicId: UInt32}
      AND hour_bucket >= range_start
      AND hour_bucket <= range_end
    GROUP BY hour_bucket
    ORDER BY hour_bucket ASC
  `;
  const rows = await queryClickhouse<Partial<TrendPoint>>(query, { topicId, ...timeParams(timeRange) });
  return rows.map((row) => ({
    time: String(row.time ?? ""),
    score: Number(row.score) || 0,
  }));
}

export async function getTopicSentimentFromCH(
  topicId: number,
  timeRange: ResolvedTimeRange,
): Promise<SentimentPoint[]> {
  const query = `
    WITH
      toDateTime({start: String}) AS range_start,
      toDateTime({end: String}) AS range_end
    SELECT
      formatDateTime(hour_bucket, '%m-%d %H:%M') AS time,
      sum(pos_count) AS pos,
      sum(neu_count) AS neu,
      sum(neg_count) AS neg
    FROM ${SCHEMA}.dbt_fct_topic_activity
    WHERE topic_id = {topicId: UInt32}
      AND hour_bucket >= range_start
      AND hour_bucket <= range_end
    GROUP BY hour_bucket
    ORDER BY hour_bucket ASC
  `;
  const rows = await queryClickhouse<Partial<SentimentPoint>>(query, { topicId, ...timeParams(timeRange) });
  return rows.map((row) => ({
    time: String(row.time ?? ""),
    pos: Number(row.pos) || 0,
    neu: Number(row.neu) || 0,
    neg: Number(row.neg) || 0,
  }));
}

export async function getTopicKeywordsFromCH(topicId: number) {
  const query = `
    SELECT top_keywords
    FROM ${SCHEMA}.dbt_dim_topics
    WHERE topic_id = {topicId: UInt32}
  `;
  const rows = await queryClickhouse<{ top_keywords?: string[] }>(query, { topicId });
  const keywords = rows[0]?.top_keywords ?? [];
  return keywords.map((word, i) => ({
    word,
    weight: Math.max(40, 100 - i * 5),
  }));
}

export async function getTopicEvidencePostsFromCH(
  topicId: number,
  timeRange: ResolvedTimeRange,
): Promise<TopicEvidencePost[]> {
  const query = `
    WITH
      toDateTime({start: String}) AS range_start,
      toDateTime({end: String}) AS range_end
    SELECT
      toString(post_id) AS id,
      ifNull(author_name, 'unknown') AS author_name,
      ifNull(source_type, 'unknown') AS source_type,
      ifNull(sentiment_label, 'neutral') AS sentiment,
      ifNull(body, '') AS body,
      toFloat64(ifNull(engagement, 0)) AS engagement
    FROM ${POSTS_ENRICHED_TABLE}
    WHERE topic_id = {topicId: UInt32}
      AND created_at >= range_start
      AND created_at <= range_end
    ORDER BY engagement DESC, created_at DESC
    LIMIT 25
  `;

  const rows = await queryClickhouse<Partial<TopicEvidencePost>>(query, { topicId, ...timeParams(timeRange) });
  return rows.map((row) => ({
    id: String(row.id ?? ""),
    author_name: String(row.author_name ?? "unknown"),
    source_type: String(row.source_type ?? "unknown"),
    sentiment: String(row.sentiment ?? "neutral"),
    body: String(row.body ?? ""),
    engagement: Number(row.engagement) || 0,
  }));
}

export async function getRecentCrisesFromCH(timeRange: ResolvedTimeRange): Promise<CrisisEvent[]> {
  const query = `
    WITH
      toDateTime({start: String}) AS range_start,
      toDateTime({end: String}) AS range_end,
      recent_crises AS (
      SELECT
        event_id,
        severity,
        parseDateTimeBestEffortOrNull(toString(detected_at)) AS parsed_detected_at,
        ifNull(affected_topic_labels, []) AS affected_topics,
        ifNull(trigger_conditions, []) AS trigger_conditions,
        neg_ratio,
        mention_velocity,
        anomaly_score,
        ifNull(evidence_post_ids, []) AS evidence_post_ids
      FROM ${SCHEMA}.dbt_fct_crisis_events
      WHERE parseDateTimeBestEffortOrNull(toString(detected_at)) >= range_start
        AND parseDateTimeBestEffortOrNull(toString(detected_at)) <= range_end
    )
    SELECT
      rc.event_id,
      rc.severity,
      formatDateTime(rc.parsed_detected_at, '%b %d, %H:%M') AS detected_at,
      formatDateTime(rc.parsed_detected_at, '%H:%M') AS time,
      rc.affected_topics,
      rc.trigger_conditions,
      round(rc.neg_ratio * 100, 1) AS neg_ratio,
      concat('+', toString(rc.mention_velocity), '/hr') AS mention_velocity,
      rc.anomaly_score,
      groupArray(
        map(
          'id', toString(p.post_id),
          'author', ifNull(p.author_name, 'unknown'),
          'text', ifNull(p.body, ''),
          'type', ifNull(p.source_type, 'unknown'),
          'engagement', toString(toInt64(ifNull(p.engagement, 0)))
        )
      ) AS evidence_posts
    FROM recent_crises rc
    LEFT ARRAY JOIN rc.evidence_post_ids AS evidence_post_id
    LEFT JOIN ${POSTS_ENRICHED_TABLE} p ON toString(p.post_id) = toString(evidence_post_id)
    GROUP BY
      rc.event_id,
      rc.severity,
      rc.parsed_detected_at,
      rc.affected_topics,
      rc.trigger_conditions,
      rc.neg_ratio,
      rc.mention_velocity,
      rc.anomaly_score
    ORDER BY rc.parsed_detected_at DESC
  `;

  const rows = await queryClickhouse<
    Omit<CrisisEvent, "severityRank"> & {
      evidence_posts: Array<Record<string, string>>;
    }
  >(query, timeParams(timeRange));

  return rows.map((row) => ({
    ...row,
    severityRank: row.severity === "HIGH" ? 3 : row.severity === "MEDIUM" ? 2 : 1,
    evidence_posts: (row.evidence_posts ?? [])
      .filter((post) => post.id)
      .map((post) => ({
        id: post.id,
        author: post.author ?? "unknown",
        text: post.text ?? "",
        type: post.type ?? "unknown",
        engagement: Number(post.engagement) || 0,
      })),
  }));
}

export async function getCrisisStatsFromCH(timeRange: ResolvedTimeRange): Promise<CrisisStats> {
  const query = `
      WITH
        toDateTime({start: String}) AS range_start,
        toDateTime({end: String}) AS range_end
      SELECT
        count(*) AS total_24h,
        countIf(severity = 'HIGH') AS high_severity_count,
        avg(mention_velocity) AS avg_velocity
      FROM (
        SELECT
          severity,
          mention_velocity,
          parseDateTimeBestEffortOrNull(toString(detected_at)) AS parsed_detected_at
        FROM ${SCHEMA}.dbt_fct_crisis_events
      )
      WHERE parsed_detected_at IS NOT NULL
        AND parsed_detected_at >= range_start
        AND parsed_detected_at <= range_end
    `;

  const rows = await queryClickhouse<Partial<CrisisStats>>(query, timeParams(timeRange));
  const row = rows[0] ?? {};
  return {
    total_24h: Number(row.total_24h) || 0,
    high_severity_count: Number(row.high_severity_count) || 0,
    avg_velocity: Number(row.avg_velocity) || 0,
  };
}
