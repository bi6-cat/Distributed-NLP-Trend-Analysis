"use client";

import { useEffect, useState } from "react";
import { getTopicKeywords } from "@/app/actions/trends";
import { WordCloudClient } from "./word-cloud-client";

export function WordCloud({ topicId }: { topicId: number }) {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const [keywords, setKeywords] = useState<any[]>([]);
  
  useEffect(() => {
    getTopicKeywords(topicId).then(setKeywords).catch(console.error);
  }, [topicId]);

  return <WordCloudClient keywords={keywords} />;
}
