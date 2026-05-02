"use client";

import {
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  ZAxis,
  Tooltip,
  CartesianGrid,
  Cell,
} from "recharts";
import {
  chartTooltipStyle,
  chartTooltipLabelStyle,
  chartAxisTickStyle,
} from "@/lib/chart-styles";

const data = [
  { id: "EVT-1", time: "08:30", severity: 3, label: "Shopee Data Breach Rumor", color: "#f43f5e" },
  { id: "EVT-2", time: "10:15", severity: 2, label: "Momo App Outage", color: "#f59e0b" },
  { id: "EVT-3", time: "14:45", severity: 1, label: "Be Group Service Fee", color: "#eab308" },
  { id: "EVT-4", time: "16:20", severity: 3, label: "FPT Play Cyber Attack", color: "#f43f5e" },
];

export function ActiveEventsTimeline() {
  return (
    <div className="h-[260px] w-full">
      <ResponsiveContainer width="100%" height="100%">
        <ScatterChart margin={{ top: 20, right: 24, bottom: 8, left: 0 }}>
          <CartesianGrid strokeDasharray="2 4" vertical={false} stroke="#f1f5f9" />
          <XAxis
            type="category"
            dataKey="time"
            tick={chartAxisTickStyle}
            axisLine={false}
            tickLine={false}
            dy={6}
          />
          <YAxis
            type="number"
            dataKey="severity"
            domain={[0, 4]}
            ticks={[1, 2, 3]}
            tickFormatter={(val) => {
              if (val === 1) return "Low";
              if (val === 2) return "Medium";
              if (val === 3) return "High";
              return "";
            }}
            tick={chartAxisTickStyle}
            axisLine={false}
            tickLine={false}
            width={60}
          />
          <ZAxis type="number" range={[280, 280]} />
          <Tooltip
            cursor={{ strokeDasharray: "3 3", stroke: "#cbd5e1" }}
            contentStyle={chartTooltipStyle}
            labelStyle={chartTooltipLabelStyle}
            formatter={(value, name) => {
              if (name === "severity") {
                const s = value === 3 ? "High" : value === 2 ? "Medium" : "Low";
                return [s, "Severity"];
              }
              return [value, name];
            }}
          />
          <Scatter name="Incidents" data={data} fill="#8884d8" shape="circle">
            {data.map((entry, index) => (
              <Cell
                key={`cell-${index}`}
                fill={entry.color}
                stroke="#ffffff"
                strokeWidth={3}
              />
            ))}
          </Scatter>
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}
