import { getActiveTopics } from "@/app/actions/trends";
import { resolveTimeRange } from "@/lib/dal/time-range";
import type { TimeRangeSearchParams } from "@/lib/time-range";
import TrendsClient from "./trends-client";

export const revalidate = 3600;
export const dynamic = "force-dynamic";

export default async function TrendsExplorerPage({
  searchParams,
}: {
  searchParams?: TimeRangeSearchParams;
}) {
  const timeRange = await resolveTimeRange(searchParams);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const topics: any[] = await getActiveTopics(timeRange);

  return <TrendsClient initialTopics={topics} timeRange={timeRange} />;
}
