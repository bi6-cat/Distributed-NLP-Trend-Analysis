"use client";

import { useEffect, useMemo, useState } from "react";
import {
  Search,
  Sparkles,
  Calendar,
  Hash,
  FileText,
  TrendingUp,
} from "lucide-react";
import { TrendScoreChart } from "@/components/trends/trend-score-chart";
import { SentimentTimelineChart } from "@/components/trends/sentiment-timeline-chart";
import { WordCloud } from "@/components/trends/word-cloud";
import { EvidencePostCards } from "@/components/trends/evidence-post-cards";
import type { ResolvedTimeRange } from "@/lib/time-range";

type Topic = {
  id: number;
  label: string;
  first_seen: string;
  last_seen: string;
  mentions: number;
  score: number;
};

export default function TrendsClient({
  initialTopics,
  timeRange,
}: {
  initialTopics: Topic[];
  timeRange: ResolvedTimeRange;
}) {
  // Guard against empty array
  const topics = useMemo(
    () =>
      initialTopics.length > 0
        ? initialTopics
        : [{ id: 0, label: "No topics found", first_seen: "-", last_seen: "-", mentions: 0, score: 0 }],
    [initialTopics],
  );

  const [activeId, setActiveId] = useState(topics[0].id);
  const activeTopic = topics.find((t) => t.id === activeId) ?? topics[0];

  useEffect(() => {
    if (!topics.some((topic) => topic.id === activeId)) {
      setActiveId(topics[0].id);
    }
  }, [activeId, topics]);

  return (
    <div className="space-y-6">
      {/* Page header */}
      <div>
        <div className="flex flex-wrap items-center gap-2">
          <p className="text-xs font-semibold uppercase tracking-[0.08em] text-indigo-600">
            Analytics
          </p>
          <span className="rounded-md bg-slate-100 px-2 py-0.5 text-[11px] font-medium text-slate-600">
            {timeRange.label}: {timeRange.start} to {timeRange.end}
          </span>
        </div>
        <h1 className="mt-1 text-3xl font-bold tracking-tight text-slate-900">
          Trends Explorer
        </h1>
        <p className="mt-1.5 text-sm text-slate-500">
          Drill into topic-level trends, sentiment, and evidence over time.
        </p>
      </div>

      {initialTopics.length === 0 && (
        <div className="rounded-lg border border-amber-200 bg-amber-50 px-4 py-3 text-sm font-medium text-amber-800">
          No data in this range. Try Latest data 7d.
        </div>
      )}

      {/* Master-detail layout */}
      <div className="grid grid-cols-1 xl:grid-cols-12 gap-5">
        {/* Sidebar: Topics list */}
        <aside className="xl:col-span-3 flex flex-col bg-white rounded-xl border border-slate-200/80 shadow-[0_1px_2px_rgba(15,23,42,0.04)] overflow-hidden">
          <div className="p-4 border-b border-slate-100">
            <div className="flex items-center justify-between mb-3">
              <h2 className="text-sm font-semibold text-slate-900">Topics</h2>
              <span className="text-[10px] font-semibold uppercase tracking-[0.08em] text-slate-400">
                {topics.length} active
              </span>
            </div>
            <div className="relative">
              <Search className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-slate-400" />
              <input
                type="text"
                placeholder="Search topics..."
                className="w-full h-9 pl-9 pr-3 rounded-lg bg-slate-50 border border-slate-200/80 text-sm text-slate-700 placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-300 focus:bg-white transition-all"
              />
            </div>
          </div>
          <div className="flex-1 overflow-y-auto thin-scrollbar max-h-[640px]">
            <ul className="p-2 space-y-1">
              {topics.map((topic) => {
                const isActive = topic.id === activeId;
                const score = Number(topic.score) || 0;
                const mentions = Number(topic.mentions) || 0;

                return (
                  <li key={topic.id}>
                    <button
                      onClick={() => setActiveId(topic.id)}
                      className={`group relative w-full text-left rounded-lg px-3 py-2.5 transition-colors ${isActive
                          ? "bg-gradient-to-r from-indigo-50 to-violet-50/50"
                          : "hover:bg-slate-50"
                        }`}
                    >
                      {isActive && (
                        <span className="absolute inset-y-2 left-0 w-0.5 rounded-full bg-gradient-to-b from-indigo-500 to-violet-500" />
                      )}
                      <div className="flex items-start justify-between gap-2">
                        <span
                          className={`text-sm font-medium leading-tight line-clamp-2 ${isActive ? "text-indigo-900" : "text-slate-800"
                            }`}
                        >
                          {topic.label}
                        </span>
                        <span
                          className={`text-[10px] font-semibold tabular-nums shrink-0 px-1.5 py-0.5 rounded-md ${isActive
                              ? "bg-white text-indigo-700 ring-1 ring-inset ring-indigo-200/60"
                              : "bg-slate-100 text-slate-500"
                            }`}
                        >
                          {score.toFixed(0)}
                        </span>
                      </div>
                      <div className="mt-1.5 flex items-center gap-2 text-[11px] text-slate-500">
                        <span className="tabular-nums">
                          {mentions.toLocaleString()} mentions
                        </span>
                        <span className="text-slate-300">·</span>
                        <span>{topic.last_seen}</span>
                      </div>
                    </button>
                  </li>
                );
              })}
            </ul>
          </div>
        </aside>

        {/* Main content */}
        <div className="xl:col-span-9 space-y-5">
          {/* Topic header card */}
          <div className="rounded-xl bg-white border border-slate-200/80 shadow-[0_1px_2px_rgba(15,23,42,0.04)] p-6">
            <div className="flex flex-col md:flex-row md:items-start md:justify-between gap-4">
              <div>
                <div className="flex items-center gap-2 mb-1.5">
                  <span className="inline-flex items-center gap-1 text-[11px] font-semibold uppercase tracking-[0.08em] text-indigo-600">
                    <Sparkles className="h-3 w-3" strokeWidth={2.5} />
                    Trending now
                  </span>
                </div>
                <h2 className="text-2xl font-bold tracking-tight text-slate-900">
                  {activeTopic.label}
                </h2>
                <div className="mt-3 flex flex-wrap items-center gap-2">
                  <span className="inline-flex items-center gap-1.5 rounded-md bg-slate-50 px-2.5 py-1 text-xs font-medium text-slate-600 ring-1 ring-inset ring-slate-200/60">
                    <Calendar className="h-3 w-3 text-slate-400" />
                    First seen <span className="font-semibold text-slate-800">{activeTopic.first_seen}</span>
                  </span>
                  <span className="inline-flex items-center gap-1.5 rounded-md bg-slate-50 px-2.5 py-1 text-xs font-medium text-slate-600 ring-1 ring-inset ring-slate-200/60">
                    <Calendar className="h-3 w-3 text-slate-400" />
                    Last seen <span className="font-semibold text-slate-800">{activeTopic.last_seen}</span>
                  </span>
                </div>
              </div>

              <div className="flex gap-3">
                <div className="rounded-lg bg-gradient-to-br from-indigo-50 to-violet-50/60 border border-indigo-100/60 px-4 py-3 min-w-[120px]">
                  <p className="text-[10px] font-semibold uppercase tracking-[0.08em] text-indigo-700/80">
                    Trend Score
                  </p>
                  <p className="mt-0.5 text-2xl font-bold tabular-nums text-indigo-700">
                    {(Number(activeTopic.score) || 0).toFixed(1)}
                  </p>
                </div>
                <div className="rounded-lg bg-slate-50 border border-slate-200/60 px-4 py-3 min-w-[120px]">
                  <p className="text-[10px] font-semibold uppercase tracking-[0.08em] text-slate-500">
                    Mentions
                  </p>
                  <p className="mt-0.5 text-2xl font-bold tabular-nums text-slate-900">
                    {(Number(activeTopic.mentions) || 0).toLocaleString()}
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* Charts row */}
          <div className="grid grid-cols-1 xl:grid-cols-2 gap-5">
            <TrendScoreChart topicId={activeTopic.id} timeRange={timeRange} />
            <SentimentTimelineChart topicId={activeTopic.id} timeRange={timeRange} />
          </div>

          {/* Bottom split pane */}
          <div className="grid grid-cols-1 xl:grid-cols-12 gap-5">
            {activeTopic.id !== 0 && (
              <div className="xl:col-span-5 rounded-xl bg-white border border-slate-200/80 shadow-[0_1px_2px_rgba(15,23,42,0.04)] p-6">
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center gap-2.5">
                    <div className="flex h-7 w-7 items-center justify-center rounded-md bg-violet-50 text-violet-600">
                      <Hash className="h-4 w-4" strokeWidth={2} />
                    </div>
                    <h3 className="text-sm font-semibold text-slate-900">Top Keywords</h3>
                  </div>
                </div>
                <WordCloud topicId={activeTopic.id} />
              </div>
            )}
            <div className="xl:col-span-7 rounded-xl bg-white border border-slate-200/80 shadow-[0_1px_2px_rgba(15,23,42,0.04)] p-6 flex flex-col h-[480px]">
              <div className="flex items-center justify-between mb-4 shrink-0">
                <div className="flex items-center gap-2.5">
                  <div className="flex h-7 w-7 items-center justify-center rounded-md bg-emerald-50 text-emerald-600">
                    <FileText className="h-4 w-4" strokeWidth={2} />
                  </div>
                  <h3 className="text-sm font-semibold text-slate-900">Top Evidence Posts</h3>
                </div>
                <span className="inline-flex items-center gap-1 text-[11px] font-medium text-slate-500">
                  <TrendingUp className="h-3 w-3" />
                  Sorted by engagement
                </span>
              </div>
              <EvidencePostCards topicId={activeTopic.id} timeRange={timeRange} />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
