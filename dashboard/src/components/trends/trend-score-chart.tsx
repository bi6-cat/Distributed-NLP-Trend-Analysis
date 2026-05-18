"use client";

import { useEffect, useState } from "react";
import { getTopicTrendScore } from "@/app/actions/trends";
import { TrendScoreChartClient } from "./trend-score-chart-client";

export function TrendScoreChart({ topicId }: { topicId: number }) {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const [data, setData] = useState<any[]>([]);
  
  useEffect(() => {
    getTopicTrendScore(topicId).then(setData).catch(console.error);
  }, [topicId]);

  return <TrendScoreChartClient data={data} />;
}
