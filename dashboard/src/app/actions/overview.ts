"use server";

import {
  getOverviewKPIsFromCH,
  getTrendingTopicsFromCH,
  getOverallSentimentFromCH,
} from "@/lib/dal/radar";
import type { ResolvedTimeRange } from "@/lib/time-range";

/**
 * Overview KPIs
 */
export async function getOverviewKPIs(timeRange: ResolvedTimeRange) {
  return getOverviewKPIsFromCH(timeRange);
}

/**
 * Top 10 Trending Topics
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export async function getTrendingTopics(timeRange: ResolvedTimeRange): Promise<any[]> {
  return getTrendingTopicsFromCH(timeRange);
}

/**
 * Overall Sentiment Distribution
 */
export async function getOverallSentiment(timeRange: ResolvedTimeRange) {
  const row = await getOverallSentimentFromCH(timeRange);
  
  return [
    { name: "Positive", value: Number(row.positive) || 0, color: "#10b981", soft: "#d1fae5" },
    { name: "Neutral", value: Number(row.neutral) || 0, color: "#94a3b8", soft: "#e2e8f0" },
    { name: "Negative", value: Number(row.negative) || 0, color: "#f43f5e", soft: "#ffe4e6" },
  ];
}
