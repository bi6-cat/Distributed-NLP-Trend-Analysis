"use client";

import { useEffect, useState } from "react";
import { getTopicSentiment } from "@/app/actions/trends";
import { SentimentTimelineChartClient } from "./sentiment-timeline-chart-client";

export function SentimentTimelineChart({ topicId }: { topicId: number }) {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const [data, setData] = useState<any[]>([]);
  
  useEffect(() => {
    getTopicSentiment(topicId).then(setData).catch(console.error);
  }, [topicId]);

  return <SentimentTimelineChartClient data={data} />;
}
