import { getActiveTopics } from "@/app/actions/trends";
import TrendsClient from "./trends-client";

export const revalidate = 3600;

export default async function TrendsExplorerPage() {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const topics: any[] = await getActiveTopics();

  return <TrendsClient initialTopics={topics} />;
}
