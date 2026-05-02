"use client";

import { TrendingUp } from "lucide-react";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import {
  chartTooltipStyle,
  chartTooltipLabelStyle,
  chartAxisTickStyle,
} from "@/lib/chart-styles";

const data = [
  { time: "00:00", score: 20 },
  { time: "04:00", score: 35 },
  { time: "08:00", score: 85 },
  { time: "12:00", score: 98 },
  { time: "16:00", score: 92 },
  { time: "20:00", score: 88 },
  { time: "23:59", score: 75 },
];

export function TrendScoreChart() {
  const peak = Math.max(...data.map((d) => d.score));

  return (
    <div className="rounded-xl bg-white border border-slate-200/80 shadow-[0_1px_2px_rgba(15,23,42,0.04)] flex flex-col h-[340px]">
      <div className="flex items-start justify-between px-6 pt-5 pb-2">
        <div className="flex items-center gap-2.5">
          <div className="flex h-7 w-7 items-center justify-center rounded-md bg-indigo-50 text-indigo-600">
            <TrendingUp className="h-4 w-4" strokeWidth={2} />
          </div>
          <div>
            <h3 className="text-sm font-semibold text-slate-900">Trend Score</h3>
            <p className="text-[11px] text-slate-500">Velocity × engagement (last 24h)</p>
          </div>
        </div>
        <div className="text-right">
          <p className="text-[10px] font-semibold uppercase tracking-[0.08em] text-slate-400">Peak</p>
          <p className="text-base font-bold tabular-nums text-slate-900">{peak.toFixed(1)}</p>
        </div>
      </div>

      <div className="flex-1 px-3 pb-4">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data} margin={{ top: 6, right: 12, left: -10, bottom: 0 }}>
            <defs>
              <linearGradient id="trendScoreGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#6366f1" stopOpacity={0.35} />
                <stop offset="95%" stopColor="#6366f1" stopOpacity={0} />
              </linearGradient>
            </defs>
            <XAxis
              dataKey="time"
              tick={chartAxisTickStyle}
              axisLine={false}
              tickLine={false}
              dy={4}
            />
            <YAxis
              tick={chartAxisTickStyle}
              axisLine={false}
              tickLine={false}
              width={40}
            />
            <CartesianGrid strokeDasharray="2 4" vertical={false} stroke="#f1f5f9" />
            <Tooltip
              contentStyle={chartTooltipStyle}
              labelStyle={chartTooltipLabelStyle}
              cursor={{ stroke: "#cbd5e1", strokeWidth: 1, strokeDasharray: "3 3" }}
            />
            <Area
              type="monotone"
              dataKey="score"
              stroke="#6366f1"
              strokeWidth={2.25}
              fillOpacity={1}
              fill="url(#trendScoreGradient)"
              activeDot={{ r: 5, fill: "#6366f1", stroke: "#fff", strokeWidth: 2 }}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
