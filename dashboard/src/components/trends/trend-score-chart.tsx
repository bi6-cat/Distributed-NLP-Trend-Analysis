"use client";

import { useEffect, useState } from "react";
import { getTopicTrendScore } from "@/app/actions/trends";
import { ChartCardLoading } from "@/components/ui/data-loading";
import { TrendScoreChartClient } from "./trend-score-chart-client";

export function TrendScoreChart({ topicId }: { topicId: number }) {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const [data, setData] = useState<any[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  
  useEffect(() => {
    let isMounted = true;

    setIsLoading(true);
    getTopicTrendScore(topicId)
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
    return <ChartCardLoading titleWidth="w-28" accentClassName="bg-indigo-100" />;
  }

  return <TrendScoreChartClient data={data} />;
}
