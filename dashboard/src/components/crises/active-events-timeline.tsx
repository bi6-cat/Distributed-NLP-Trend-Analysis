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

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function ActiveEventsTimeline({ events }: { events: any[] }) {
  // Guard against empty array and map properties
  const safeData = events.length > 0 ? events.map(e => ({
    id: e.event_id,
    time: e.time,
    severity: e.severityRank,
    label: e.affected_topics[0] || "Unknown Event",
    color: e.severity === 'HIGH' ? '#f43f5e' : e.severity === 'MEDIUM' ? '#f59e0b' : '#eab308'
  })) : [
    { id: "No-events", time: "12:00", severity: 0, label: "No active events", color: "#cbd5e1" }
  ];

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
          <Scatter name="Incidents" data={safeData} fill="#8884d8" shape="circle">
            {safeData.map((entry, index) => (
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
