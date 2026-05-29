"use server";

import {
  getActiveTopicsFromCH,
  getTopicEvidencePostsFromCH,
  getTopicKeywordsFromCH,
  getTopicSentimentFromCH,
  getTopicTrendScoreFromCH,
} from "@/lib/dal/radar";
import type { ResolvedTimeRange } from "@/lib/time-range";

/**
 * Get all active topics for the sidebar
 */
export async function getActiveTopics(timeRange: ResolvedTimeRange) {
  return getActiveTopicsFromCH(timeRange);
}

/**
 * Get trend score time-series for a topic
 */
export async function getTopicTrendScore(topicId: number, timeRange: ResolvedTimeRange) {
  return getTopicTrendScoreFromCH(topicId, timeRange);
}

/**
 * Get sentiment time-series for a topic
 */
export async function getTopicSentiment(topicId: number, timeRange: ResolvedTimeRange) {
  return getTopicSentimentFromCH(topicId, timeRange);
}

/**
 * Get word cloud keywords for a topic
 */
export async function getTopicKeywords(topicId: number) {
  return getTopicKeywordsFromCH(topicId);
}

/**
 * Get top evidence posts for a topic
 */
export async function getTopicEvidencePosts(topicId: number, timeRange: ResolvedTimeRange) {
  return getTopicEvidencePostsFromCH(topicId, timeRange);
}
