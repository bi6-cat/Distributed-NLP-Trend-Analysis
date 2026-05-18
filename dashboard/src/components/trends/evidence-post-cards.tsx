"use client";

import { useEffect, useState } from "react";
import { getTopicEvidencePosts } from "@/app/actions/trends";
import { EvidencePostCardsClient } from "./evidence-post-cards-client";

type PostData = {
  id: string | number;
  author_name: string;
  source_type: string;
  sentiment: string;
  body: string;
  engagement: number;
};

export function EvidencePostCards({ topicId }: { topicId: number }) {
  const [posts, setPosts] = useState<PostData[]>([]);

  useEffect(() => {
    getTopicEvidencePosts(topicId).then(setPosts).catch(console.error);
  }, [topicId]);

  return <EvidencePostCardsClient posts={posts} />;
}
