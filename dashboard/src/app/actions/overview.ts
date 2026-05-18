"use server";

import {
  getOverviewKPIsFromCH,
  getTrendingTopicsFromCH,
  getOverallSentimentFromCH,
} from "@/lib/dal/radar";

/**
 * Overview KPIs
 */
export async function getOverviewKPIs() {
  return getOverviewKPIsFromCH();
}

/**
 * Top 10 Trending Topics
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export async function getTrendingTopics(): Promise<any[]> {
  return getTrendingTopicsFromCH();
}

/**
 * Overall Sentiment Distribution
 */
export async function getOverallSentiment() {
  const row = await getOverallSentimentFromCH();
  
  return [
    { name: "Positive", value: Number(row.positive) || 0, color: "#10b981", soft: "#d1fae5" },
    { name: "Neutral", value: Number(row.neutral) || 0, color: "#94a3b8", soft: "#e2e8f0" },
    { name: "Negative", value: Number(row.negative) || 0, color: "#f43f5e", soft: "#ffe4e6" },
  ];
}
