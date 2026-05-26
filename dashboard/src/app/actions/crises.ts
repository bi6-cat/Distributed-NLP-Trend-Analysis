"use server";

import { getCrisisStatsFromCH, getRecentCrisesFromCH } from "@/lib/dal/radar";

/**
 * Get recent crisis events
 */
export async function getRecentCrises() {
  return getRecentCrisesFromCH();
}

/**
 * Get aggregate crisis stats
 */
export async function getCrisisStats() {
  return getCrisisStatsFromCH();
}
