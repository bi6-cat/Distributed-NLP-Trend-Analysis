"use client";

import { useEffect, useState } from "react";
import { getTopicKeywords } from "@/app/actions/trends";
import { WordCloudLoading } from "@/components/ui/data-loading";
import { WordCloudClient } from "./word-cloud-client";

export function WordCloud({ topicId }: { topicId: number }) {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const [keywords, setKeywords] = useState<any[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  
  useEffect(() => {
    let isMounted = true;

    setIsLoading(true);
    getTopicKeywords(topicId)
      .then((nextKeywords) => {
        if (isMounted) setKeywords(nextKeywords);
      })
      .catch((error) => {
        console.error(error);
        if (isMounted) setKeywords([]);
      })
      .finally(() => {
        if (isMounted) setIsLoading(false);
      });

    return () => {
      isMounted = false;
    };
  }, [topicId]);

  if (isLoading) {
    return <WordCloudLoading />;
  }

  return <WordCloudClient keywords={keywords} />;
}
