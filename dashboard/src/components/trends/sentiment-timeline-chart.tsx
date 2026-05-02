"use client";

import { BarChart3 } from "lucide-react";
import {
  BarChart,
  Bar,
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
  { time: "00:00", pos: 200, neg: 50, neu: 150 },
  { time: "04:00", pos: 250, neg: 80, neu: 200 },
  { time: "08:00", pos: 800, neg: 400, neu: 500 },
  { time: "12:00", pos: 1200, neg: 600, neu: 800 },
  { time: "16:00", pos: 900, neg: 500, neu: 700 },
  { time: "20:00", pos: 700, neg: 350, neu: 600 },
  { time: "23:59", pos: 500, neg: 200, neu: 400 },
];

const series = [
  { key: "pos", name: "Positive", color: "#10b981" },
  { key: "neu", name: "Neutral", color: "#cbd5e1" },
  { key: "neg", name: "Negative", color: "#f43f5e" },
];

export function SentimentTimelineChart() {
  return (
    <div className="rounded-xl bg-white border border-slate-200/80 shadow-[0_1px_2px_rgba(15,23,42,0.04)] flex flex-col h-[340px]">
      <div className="flex items-start justify-between px-6 pt-5 pb-2">
        <div className="flex items-center gap-2.5">
          <div className="flex h-7 w-7 items-center justify-center rounded-md bg-emerald-50 text-emerald-600">
            <BarChart3 className="h-4 w-4" strokeWidth={2} />
          </div>
          <div>
            <h3 className="text-sm font-semibold text-slate-900">Sentiment Over Time</h3>
            <p className="text-[11px] text-slate-500">Volume by polarity (last 24h)</p>
          </div>
        </div>
        <div className="flex items-center gap-3">
          {series.map((s) => (
            <div key={s.key} className="flex items-center gap-1.5">
              <span className="h-2 w-2 rounded-full" style={{ backgroundColor: s.color }} />
              <span className="text-[11px] font-medium text-slate-600">{s.name}</span>
            </div>
          ))}
        </div>
      </div>

      <div className="flex-1 px-3 pb-4">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={{ top: 6, right: 12, left: -10, bottom: 0 }}>
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
              cursor={{ fill: "rgba(241, 245, 249, 0.6)" }}
            />
            {series.map((s, idx) => (
              <Bar
                key={s.key}
                dataKey={s.key}
                name={s.name}
                stackId="a"
                fill={s.color}
                radius={idx === series.length - 1 ? [4, 4, 0, 0] : 0}
                maxBarSize={36}
              />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
