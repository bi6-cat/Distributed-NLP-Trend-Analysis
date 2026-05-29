import "server-only";
import { queryClickhouse } from "@/lib/clickhouse";
import {
  DEFAULT_TIME_RANGE_MODE,
  TimeRangeMode,
  TimeRangeSearchParams,
  firstParam,
  getTimeRangePresetLabel,
  isDateOnly,
  normalizeTimeRangeMode,
  type ResolvedTimeRange,
} from "@/lib/time-range";

const SCHEMA = process.env.CLICKHOUSE_DATABASE ?? "tech_radar";

type RangeRow = {
  start: string;
  end: string;
};

function customRange(from: string, to: string): ResolvedTimeRange {
  const [startDate, endDate] = from <= to ? [from, to] : [to, from];
  return {
    mode: "custom",
    start: `${startDate} 00:00:00`,
    end: `${endDate} 23:59:59`,
    label: `${startDate} to ${endDate}`,
    fromDate: startDate,
    toDate: endDate,
  };
}

async function latestDataRange(mode: Extract<TimeRangeMode, "latest7d" | "latest30d">) {
  const days = mode === "latest30d" ? 30 : 7;
  const rows = await queryClickhouse<RangeRow>(`
    WITH ifNull(max(hour_bucket), now()) AS anchor
    SELECT
      formatDateTime(anchor - INTERVAL ${days} DAY, '%F %T') AS start,
      formatDateTime(anchor, '%F %T') AS end
    FROM ${SCHEMA}.dbt_fct_topic_activity
  `);

  return {
    mode,
    start: rows[0]?.start,
    end: rows[0]?.end,
    label: getTimeRangePresetLabel(mode),
  };
}

async function realTimeRange(mode: Extract<TimeRangeMode, "last24h" | "last7d" | "last30d">) {
  const interval =
    mode === "last24h"
      ? "24 HOUR"
      : mode === "last30d"
        ? "30 DAY"
        : "7 DAY";

  const rows = await queryClickhouse<RangeRow>(`
    SELECT
      formatDateTime(now() - INTERVAL ${interval}, '%F %T') AS start,
      formatDateTime(now(), '%F %T') AS end
  `);

  return {
    mode,
    start: rows[0]?.start,
    end: rows[0]?.end,
    label: getTimeRangePresetLabel(mode),
  };
}

export async function resolveTimeRange(
  searchParams?: TimeRangeSearchParams,
): Promise<ResolvedTimeRange> {
  const requestedMode = normalizeTimeRangeMode(firstParam(searchParams?.range));
  const from = firstParam(searchParams?.from);
  const to = firstParam(searchParams?.to);

  if (requestedMode === "custom" && isDateOnly(from) && isDateOnly(to)) {
    return customRange(from, to);
  }

  const mode = requestedMode === "custom" ? DEFAULT_TIME_RANGE_MODE : requestedMode;
  const resolved =
    mode === "latest7d" || mode === "latest30d"
      ? await latestDataRange(mode)
      : mode === "last24h" || mode === "last7d" || mode === "last30d"
        ? await realTimeRange(mode)
        : await latestDataRange(DEFAULT_TIME_RANGE_MODE);

  return {
    mode,
    start: resolved.start,
    end: resolved.end,
    label: resolved.label,
  };
}
