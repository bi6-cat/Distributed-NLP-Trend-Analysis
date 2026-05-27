"use client";

import { PieChart as PieChartIcon } from "lucide-react";
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from "recharts";
import { chartTooltipStyle } from "@/lib/chart-styles";

type SentimentData = {
  name: string;
  value: number;
  color: string;
  soft: string;
};

interface SentimentDonutChartProps {
  className?: string;
  data: SentimentData[];
}

export function SentimentDonutChart({ className, data }: SentimentDonutChartProps) {
  // Guard against empty data
  const safeData = data?.length > 0 ? data : [
    { name: "Positive", value: 1, color: "#10b981", soft: "#d1fae5" },
    { name: "Neutral", value: 1, color: "#94a3b8", soft: "#e2e8f0" },
    { name: "Negative", value: 1, color: "#f43f5e", soft: "#ffe4e6" },
  ];
  
  const total = safeData.reduce((sum, d) => sum + d.value, 0) || 1;
  const dominant = safeData.reduce((max, d) => (d.value > max.value ? d : max), safeData[0]);
  const dominantPct = ((dominant.value / total) * 100).toFixed(0);

  return (
    <div
      className={`flex flex-col rounded-xl bg-white border border-slate-200/80 shadow-[0_1px_2px_rgba(15,23,42,0.04)] ${className ?? ""}`}
    >
      <div className="flex items-center justify-between px-6 pt-5 pb-3">
        <div className="flex items-center gap-2.5">
          <div className="flex h-7 w-7 items-center justify-center rounded-md bg-slate-100 text-slate-600">
            <PieChartIcon className="h-4 w-4" strokeWidth={2} />
          </div>
          <h3 className="text-sm font-semibold text-slate-900">Sentiment Distribution</h3>
        </div>
        <span className="text-[11px] font-medium text-slate-500">Today</span>
      </div>

      <div className="relative px-6 pb-2 flex-1">
        <div className="h-[240px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie
                data={safeData}
                cx="50%"
                cy="50%"
                innerRadius={70}
                outerRadius={95}
                paddingAngle={3}
                dataKey="value"
                stroke="none"
              >
                {safeData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip
                contentStyle={chartTooltipStyle}
                formatter={(value, name) => {
                  const numValue = Number(value);
                  return [
                    `${numValue.toLocaleString()} (${((numValue / total) * 100).toFixed(1)}%)`,
                    name,
                  ];
                }}
              />
            </PieChart>
          </ResponsiveContainer>
        </div>
        {/* Center stat */}
        <div className="pointer-events-none absolute inset-x-0 top-[110px] flex flex-col items-center">
          <span className="text-[11px] font-semibold uppercase tracking-[0.08em] text-slate-500">
            Dominant
          </span>
          <span className="mt-0.5 text-2xl font-bold tabular-nums tracking-tight" style={{ color: dominant.color }}>
            {dominantPct}%
          </span>
          <span className="text-xs font-medium text-slate-500">{dominant.name}</span>
        </div>
      </div>

      {/* Legend with bars */}
      <div className="px-6 pb-5 pt-2 space-y-2">
        {safeData.map((d) => {
          const pct = (d.value / total) * 100;
          return (
            <div key={d.name} className="flex items-center gap-3">
              <span
                className="h-2 w-2 shrink-0 rounded-full"
                style={{ backgroundColor: d.color }}
              />
              <span className="text-xs font-medium text-slate-600 w-16">{d.name}</span>
              <div className="relative h-1.5 flex-1 rounded-full bg-slate-100 overflow-hidden">
                <div
                  className="h-full rounded-full"
                  style={{ width: `${pct}%`, backgroundColor: d.color }}
                />
              </div>
              <span className="text-xs font-semibold tabular-nums text-slate-700 w-10 text-right">
                {pct.toFixed(0)}%
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
