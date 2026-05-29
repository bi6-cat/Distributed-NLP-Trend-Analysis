"use client";

import { useEffect, useState } from "react";
import { getTopicEvidencePosts } from "@/app/actions/trends";
import { EvidencePostsLoading } from "@/components/ui/data-loading";
import type { ResolvedTimeRange } from "@/lib/time-range";
import { EvidencePostCardsClient } from "./evidence-post-cards-client";

type PostData = {
  id: string | number;
  author_name: string;
  source_type: string;
  sentiment: string;
  body: string;
  engagement: number;
};

export function EvidencePostCards({
  topicId,
  timeRange,
}: {
  topicId: number;
  timeRange: ResolvedTimeRange;
}) {
  const [posts, setPosts] = useState<PostData[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    let isMounted = true;

    setIsLoading(true);
    getTopicEvidencePosts(topicId, timeRange)
      .then((nextPosts) => {
        if (isMounted) setPosts(nextPosts);
      })
      .catch((error) => {
        console.error(error);
        if (isMounted) setPosts([]);
      })
      .finally(() => {
        if (isMounted) setIsLoading(false);
      });

    return () => {
      isMounted = false;
    };
  }, [topicId, timeRange]);

  if (isLoading) {
    return <EvidencePostsLoading />;
  }

  return <EvidencePostCardsClient posts={posts} />;
}
