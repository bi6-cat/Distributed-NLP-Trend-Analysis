"use client";

import { Flame, ArrowUpRight } from "lucide-react";

const trendingTopics = [
  { id: 1, label: "Apple Silicon M4 Release", score: 98.5, volume: 15420, delta: 24 },
  { id: 2, label: "VinFast VF3 Launch", score: 92.1, volume: 12100, delta: 18 },
  { id: 3, label: "Shopee Data Breach Rumor", score: 88.4, volume: 8430, delta: 42 },
  { id: 4, label: "VNG Stock NYSE IPO", score: 85.0, volume: 7200, delta: -3 },
  { id: 5, label: "FPT Play Cyber Attack", score: 82.3, volume: 6890, delta: 31 },
  { id: 6, label: "Momo App Outage", score: 79.1, volume: 5400, delta: 12 },
  { id: 7, label: "Gojek New Fee Policy", score: 76.8, volume: 4900, delta: 8 },
  { id: 8, label: "TikTok Shop Vietnam Ban", score: 74.2, volume: 4100, delta: -5 },
  { id: 9, label: "Viettel 5G Coverage Expansion", score: 71.5, volume: 3800, delta: 6 },
  { id: 10, label: "Be Group Funding Round", score: 68.9, volume: 3100, delta: 2 },
];

interface TrendingTopicsTableProps {
  className?: string;
}

export function TrendingTopicsTable({ className }: TrendingTopicsTableProps) {
  const maxScore = Math.max(...trendingTopics.map((t) => t.score));

  return (
    <div
      className={`flex flex-col rounded-xl bg-white border border-slate-200/80 shadow-[0_1px_2px_rgba(15,23,42,0.04)] ${className ?? ""}`}
    >
      <div className="flex items-center justify-between px-6 pt-5 pb-4 border-b border-slate-100">
        <div className="flex items-center gap-2.5">
          <div className="flex h-7 w-7 items-center justify-center rounded-md bg-amber-50 text-amber-600">
            <Flame className="h-4 w-4" strokeWidth={2} />
          </div>
          <div>
            <h3 className="text-sm font-semibold text-slate-900">Top 10 Trending Topics</h3>
            <p className="text-[11px] text-slate-500">Ranked by trend score (velocity × engagement)</p>
          </div>
        </div>
        <button className="inline-flex items-center gap-1 text-xs font-medium text-indigo-600 hover:text-indigo-700">
          View all
          <ArrowUpRight className="h-3.5 w-3.5" />
        </button>
      </div>

      <div className="px-3 py-2">
        <div className="grid grid-cols-12 gap-3 px-3 py-2 text-[10px] font-semibold uppercase tracking-[0.08em] text-slate-400">
          <div className="col-span-1">#</div>
          <div className="col-span-5">Topic</div>
          <div className="col-span-4">Trend Score</div>
          <div className="col-span-1 text-right">Δ%</div>
          <div className="col-span-1 text-right">Volume</div>
        </div>

        <ul className="space-y-0.5">
          {trendingTopics.map((topic, idx) => {
            const widthPct = (topic.score / maxScore) * 100;
            const isPositive = topic.delta >= 0;
            return (
              <li
                key={topic.id}
                className="group grid grid-cols-12 gap-3 items-center px-3 py-2.5 rounded-lg hover:bg-slate-50 transition-colors cursor-pointer"
              >
                <div className="col-span-1">
                  <span
                    className={`inline-flex h-6 w-6 items-center justify-center rounded-md text-[11px] font-bold tabular-nums ${
                      idx < 3
                        ? "bg-gradient-to-br from-indigo-50 to-violet-50 text-indigo-700 ring-1 ring-inset ring-indigo-100"
                        : "bg-slate-50 text-slate-500 ring-1 ring-inset ring-slate-100"
                    }`}
                  >
                    {idx + 1}
                  </span>
                </div>
                <div className="col-span-5">
                  <span className="text-sm font-medium text-slate-900 group-hover:text-indigo-700 transition-colors">
                    {topic.label}
                  </span>
                </div>
                <div className="col-span-4 flex items-center gap-2.5">
                  <span className="text-xs font-semibold tabular-nums text-slate-700 w-9">
                    {topic.score.toFixed(1)}
                  </span>
                  <div className="h-1.5 flex-1 bg-slate-100 rounded-full overflow-hidden">
                    <div
                      className="h-full rounded-full bg-gradient-to-r from-indigo-500 to-violet-500"
                      style={{ width: `${widthPct}%` }}
                    />
                  </div>
                </div>
                <div className="col-span-1 text-right">
                  <span
                    className={`inline-flex items-center text-[11px] font-semibold tabular-nums ${
                      isPositive ? "text-emerald-600" : "text-rose-600"
                    }`}
                  >
                    {isPositive ? "+" : ""}
                    {topic.delta}%
                  </span>
                </div>
                <div className="col-span-1 text-right">
                  <span className="text-xs font-semibold tabular-nums text-slate-700">
                    {topic.volume.toLocaleString()}
                  </span>
                </div>
              </li>
            );
          })}
        </ul>
      </div>
    </div>
  );
}
