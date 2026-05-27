"use client";

import { useEffect, useState } from "react";
import { getTopicSentiment } from "@/app/actions/trends";
import { ChartCardLoading } from "@/components/ui/data-loading";
import { SentimentTimelineChartClient } from "./sentiment-timeline-chart-client";

export function SentimentTimelineChart({ topicId }: { topicId: number }) {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const [data, setData] = useState<any[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  
  useEffect(() => {
    let isMounted = true;

    setIsLoading(true);
    getTopicSentiment(topicId)
      .then((nextData) => {
        if (isMounted) setData(nextData);
      })
      .catch((error) => {
        console.error(error);
        if (isMounted) setData([]);
      })
      .finally(() => {
        if (isMounted) setIsLoading(false);
      });

    return () => {
      isMounted = false;
    };
  }, [topicId]);

  if (isLoading) {
    return <ChartCardLoading titleWidth="w-40" accentClassName="bg-emerald-100" />;
  }

  return <SentimentTimelineChartClient data={data} />;
}
