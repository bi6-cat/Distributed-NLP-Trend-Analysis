import "server-only";
import { unstable_cache } from "next/cache";
import { queryClickhouse } from "@/lib/clickhouse";

const SCHEMA = process.env.CLICKHOUSE_DATABASE ?? "tech_radar";
const ONE_HOUR = 3600;
const POSTS_ENRICHED_TABLE = `${SCHEMA}.dbt_int_posts_enriched`;

const CACHE_KEY_REVISION = process.env.DASHBOARD_CACHE_REVISION ?? "";

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

const getOverviewKPIsCached = unstable_cache(
  async (): Promise<OverviewKpis> => {
    
    const queryMentions = `
      SELECT 
        sumIf(mention_count, bucket_date = today()) AS daily_mentions,
        sumIf(mention_count, bucket_date = yesterday()) AS yesterday_mentions
      FROM ${SCHEMA}.dbt_fct_topic_activity
      WHERE bucket_date >= yesterday()
    `;

    const queryCrises = `
      SELECT count(DISTINCT event_id) AS active_crises 
      FROM ${SCHEMA}.dbt_fct_crisis_events 
      WHERE severity = 'HIGH' AND detected_at >= now() - INTERVAL 24 HOUR
    `;

    const [mentionsRows, crisesRows] = await Promise.all([
      queryClickhouse<OverviewMentionsRow>(queryMentions),
      queryClickhouse<OverviewCrisesRow>(queryCrises)
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
  },
  ["overview-kpis", CACHE_KEY_REVISION],
  { revalidate: ONE_HOUR },
);

const getTrendingTopicsCached = unstable_cache(
  async (): Promise<TrendingTopic[]> => {
    const query = `
      SELECT
        topic_id AS id,
        topic_label AS label,
        max(trend_score) AS score,
        sum(mention_count) AS volume,
        sum(acceleration) AS delta
      FROM ${SCHEMA}.dbt_fct_topic_activity
      WHERE hour_bucket >= now() - INTERVAL 24 HOUR
      GROUP BY topic_id, topic_label
      ORDER BY score DESC
      LIMIT 10
    `;
    const rows = await queryClickhouse<Partial<TrendingTopic>>(query);
    return rows.map((row) => ({
      id: Number(row.id) || 0,
      label: String(row.label ?? ""),
      score: Number(row.score) || 0,
      volume: Number(row.volume) || 0,
      delta: Number(row.delta) || 0,
    }));
  },
  ["overview-trending-topics", CACHE_KEY_REVISION],
  { revalidate: ONE_HOUR },
);

const getOverallSentimentCached = unstable_cache(
  async (): Promise<SentimentBreakdown> => {
    const query = `
      SELECT
        sum(pos_count) AS positive,
        sum(neu_count) AS neutral,
        sum(neg_count) AS negative
      FROM ${SCHEMA}.dbt_fct_topic_activity
      WHERE hour_bucket >= now() - INTERVAL 24 HOUR
    `;
    const rows = await queryClickhouse<Partial<SentimentBreakdown>>(query);
    const row = rows[0] ?? {};
    return {
      positive: Number(row.positive) || 0,
      neutral: Number(row.neutral) || 0,
      negative: Number(row.negative) || 0,
    };
  },
  ["overview-sentiment", CACHE_KEY_REVISION],
  { revalidate: ONE_HOUR },
);

const getActiveTopicsCached = unstable_cache(
  async (): Promise<ActiveTopic[]> => {
    const query = `
      WITH recent_scores AS (
        SELECT
          topic_id,
          max(trend_score) AS score
        FROM ${SCHEMA}.dbt_fct_topic_activity
        WHERE hour_bucket >= now() - INTERVAL 24 HOUR
        GROUP BY topic_id
      )
      SELECT
        t.topic_id AS id,
        t.label,
        formatDateTime(parseDateTimeBestEffort(toString(t.first_seen)), '%b %d') AS first_seen,
        formatDateTime(parseDateTimeBestEffort(toString(t.last_seen)), '%b %d') AS last_seen,
        t.total_mentions AS mentions,
        ifNull(rs.score, 0) AS score
      FROM ${SCHEMA}.dbt_dim_topics t
      LEFT JOIN recent_scores rs ON rs.topic_id = t.topic_id
      WHERE parseDateTimeBestEffortOrNull(toString(t.last_seen)) >= now() - INTERVAL 7 DAY
      ORDER BY score DESC, mentions DESC
      LIMIT 50
    `;
    const rows = await queryClickhouse<Partial<ActiveTopic>>(query);
    return rows.map((row) => ({
      id: Number(row.id) || 0,
      label: String(row.label ?? ""),
      first_seen: String(row.first_seen ?? "-"),
      last_seen: String(row.last_seen ?? "-"),
      mentions: Number(row.mentions) || 0,
      score: Number(row.score) || 0,
    }));
  },
  ["trends-active-topics", CACHE_KEY_REVISION],
  { revalidate: ONE_HOUR },
);

export async function getOverviewKPIsFromCH() {
  return getOverviewKPIsCached();
}

export async function getTrendingTopicsFromCH() {
  return getTrendingTopicsCached();
}

export async function getOverallSentimentFromCH() {
  return getOverallSentimentCached();
}

export async function getActiveTopicsFromCH() {
  return getActiveTopicsCached();
}

export async function getTopicTrendScoreFromCH(topicId: number): Promise<TrendPoint[]> {
  const query = `
    SELECT
      formatDateTime(hour_bucket, '%m-%d %H:%M') AS time,
      max(trend_score) AS score
    FROM ${SCHEMA}.dbt_fct_topic_activity
    WHERE topic_id = {topicId: UInt32}
      AND hour_bucket >= now() - INTERVAL 7 DAY
    GROUP BY hour_bucket
    ORDER BY hour_bucket ASC
  `;
  const rows = await queryClickhouse<Partial<TrendPoint>>(query, { topicId });
  return rows.map((row) => ({
    time: String(row.time ?? ""),
    score: Number(row.score) || 0,
  }));
}

export async function getTopicSentimentFromCH(topicId: number): Promise<SentimentPoint[]> {
  const query = `
    SELECT
      formatDateTime(hour_bucket, '%m-%d %H:%M') AS time,
      sum(pos_count) AS pos,
      sum(neu_count) AS neu,
      sum(neg_count) AS neg
    FROM ${SCHEMA}.dbt_fct_topic_activity
    WHERE topic_id = {topicId: UInt32}
      AND hour_bucket >= now() - INTERVAL 7 DAY
    GROUP BY hour_bucket
    ORDER BY hour_bucket ASC
  `;
  const rows = await queryClickhouse<Partial<SentimentPoint>>(query, { topicId });
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

export async function getTopicEvidencePostsFromCH(topicId: number): Promise<TopicEvidencePost[]> {
  const query = `
    SELECT
      toString(post_id) AS id,
      ifNull(author_name, 'unknown') AS author_name,
      ifNull(source_type, 'unknown') AS source_type,
      ifNull(sentiment_label, 'neutral') AS sentiment,
      ifNull(body, '') AS body,
      toFloat64(ifNull(engagement, 0)) AS engagement
    FROM ${POSTS_ENRICHED_TABLE}
    WHERE topic_id = {topicId: UInt32}
      AND created_at >= now() - INTERVAL 7 DAY
    ORDER BY engagement DESC, created_at DESC
    LIMIT 25
  `;

  const rows = await queryClickhouse<Partial<TopicEvidencePost>>(query, { topicId });
  return rows.map((row) => ({
    id: String(row.id ?? ""),
    author_name: String(row.author_name ?? "unknown"),
    source_type: String(row.source_type ?? "unknown"),
    sentiment: String(row.sentiment ?? "neutral"),
    body: String(row.body ?? ""),
    engagement: Number(row.engagement) || 0,
  }));
}

export async function getRecentCrisesFromCH(): Promise<CrisisEvent[]> {
  const query = `
    WITH recent_crises AS (
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
      WHERE detected_at >= now() - INTERVAL 7 DAY
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
  >(query);

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

const getCrisisStatsCached = unstable_cache(
  async (): Promise<CrisisStats> => {
    const query = `
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
        WHERE detected_at >= now() - INTERVAL 24 HOUR
      )
      WHERE parsed_detected_at IS NOT NULL
    `;

    const rows = await queryClickhouse<Partial<CrisisStats>>(query);
    const row = rows[0] ?? {};
    return {
      total_24h: Number(row.total_24h) || 0,
      high_severity_count: Number(row.high_severity_count) || 0,
      avg_velocity: Number(row.avg_velocity) || 0,
    };
  },
  ["crises-stats", CACHE_KEY_REVISION],
  { revalidate: ONE_HOUR },
);

export async function getCrisisStatsFromCH() { 
  return getCrisisStatsCached();
}
