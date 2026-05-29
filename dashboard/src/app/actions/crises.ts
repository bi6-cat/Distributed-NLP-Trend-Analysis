"use server";

import { getCrisisStatsFromCH, getRecentCrisesFromCH } from "@/lib/dal/radar";
import type { ResolvedTimeRange } from "@/lib/time-range";

/**
 * Get recent crisis events
 */
export async function getRecentCrises(timeRange: ResolvedTimeRange) {
  return getRecentCrisesFromCH(timeRange);
}

/**
 * Get aggregate crisis stats
 */
export async function getCrisisStats(timeRange: ResolvedTimeRange) {
  return getCrisisStatsFromCH(timeRange);
}
