"use server";

import {
  getActiveTopicsFromCH,
  getTopicEvidencePostsFromCH,
  getTopicKeywordsFromCH,
  getTopicSentimentFromCH,
  getTopicTrendScoreFromCH,
} from "@/lib/dal/radar";

/**
 * Get all active topics for the sidebar
 */
export async function getActiveTopics() {
  return getActiveTopicsFromCH();
}

/**
 * Get trend score time-series for a topic
 */
export async function getTopicTrendScore(topicId: number) {
  return getTopicTrendScoreFromCH(topicId);
}

/**
 * Get sentiment time-series for a topic
 */
export async function getTopicSentiment(topicId: number) {
  return getTopicSentimentFromCH(topicId);
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
export async function getTopicEvidencePosts(topicId: number) {
  return getTopicEvidencePostsFromCH(topicId);
}
