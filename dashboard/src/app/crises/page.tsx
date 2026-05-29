import { Activity, Siren, Radio } from "lucide-react";
import { ActiveEventsTimeline } from "@/components/crises/active-events-timeline";
import { IncidentDetailCards } from "@/components/crises/incident-detail-cards";
import { getRecentCrises, getCrisisStats } from "@/app/actions/crises";
import { resolveTimeRange } from "@/lib/dal/time-range";
import type { TimeRangeSearchParams } from "@/lib/time-range";

export const revalidate = 3600;
export const dynamic = "force-dynamic";

export default async function CrisisMonitorPage({
  searchParams,
}: {
  searchParams?: TimeRangeSearchParams;
}) {
  const timeRange = await resolveTimeRange(searchParams);
  const [crises, stats] = await Promise.all([
    getRecentCrises(timeRange),
    getCrisisStats(timeRange)
  ]);

  const activeCount = crises.filter(c => (c as Record<string, unknown>).severity === 'HIGH').length;
  const total24h = Number(stats.total_24h) || 0;
  const highSeverity = Number(stats.high_severity_count) || 0;
  const avgVelocity = Number(stats.avg_velocity) || 0;

  return (
    <div className="space-y-6">
      {/* Page header */}
      <div className="flex flex-col sm:flex-row sm:items-end sm:justify-between gap-4">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.08em] text-rose-600">
            Operations
          </p>
          <h1 className="mt-1 text-3xl font-bold tracking-tight text-slate-900">
            Crisis Monitor
          </h1>
          <p className="mt-1.5 text-sm text-slate-500">
            Real-time anomaly detection
          </p>
          <p className="mt-2 text-xs font-medium text-slate-500">
            {timeRange.label}: {timeRange.start} to {timeRange.end}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <span className="inline-flex items-center gap-2 rounded-full bg-gradient-to-r from-rose-50 to-orange-50 ring-1 ring-inset ring-rose-200/60 px-3 py-1.5 text-xs font-semibold text-rose-700">
            <span className="relative flex h-2 w-2">
              <span className={`absolute inline-flex h-full w-full rounded-full bg-rose-400 opacity-75 ${activeCount > 0 ? "animate-ping" : ""}`} />
              <span className="relative inline-flex h-2 w-2 rounded-full bg-rose-500" />
            </span>
            {activeCount} Active Events
          </span>
          <span className="inline-flex items-center gap-1.5 rounded-full bg-slate-100 px-3 py-1.5 text-xs font-medium text-slate-600">
            <Radio className="h-3.5 w-3.5 text-slate-400" />
            Streaming
          </span>
        </div>
      </div>

      {crises.length === 0 && total24h === 0 && (
        <div className="rounded-lg border border-amber-200 bg-amber-50 px-4 py-3 text-sm font-medium text-amber-800">
          No data in this range. Try Latest data 7d.
        </div>
      )}

      {/* Stat strip */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {[
          { label: "Selected range", value: total24h.toString(), sub: "events detected", tone: "slate" },
          { label: "High severity", value: highSeverity.toString(), sub: "requires action", tone: "rose" },
          { label: "Avg. velocity", value: `+${avgVelocity.toFixed(0)}/hr`, sub: "across active", tone: "amber" },
          { label: "Resolved today", value: "0", sub: "auto-cleared", tone: "emerald" }, // Mocked resolved
        ].map((s) => {
          const toneClasses: Record<string, string> = {
            slate: "text-slate-900",
            rose: "text-rose-700",
            amber: "text-amber-700",
            emerald: "text-emerald-700",
          };
          return (
            <div
              key={s.label}
              className="rounded-xl bg-white border border-slate-200/80 shadow-[0_1px_2px_rgba(15,23,42,0.04)] px-4 py-3.5"
            >
              <p className="text-[10px] font-semibold uppercase tracking-[0.08em] text-slate-500">
                {s.label}
              </p>
              <p
                className={`mt-1 text-2xl font-bold tabular-nums tracking-tight ${
                  toneClasses[s.tone]
                }`}
              >
                {s.value}
              </p>
              <p className="text-[11px] text-slate-500">{s.sub}</p>
            </div>
          );
        })}
      </div>

      {/* Timeline section */}
      <section className="rounded-xl bg-white border border-slate-200/80 shadow-[0_1px_2px_rgba(15,23,42,0.04)]">
        <div className="px-6 pt-5 pb-3 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="flex h-7 w-7 items-center justify-center rounded-md bg-indigo-50 text-indigo-600">
              <Activity className="h-4 w-4" strokeWidth={2} />
            </div>
            <div>
              <h2 className="text-sm font-semibold text-slate-900">Active Events Timeline</h2>
              <p className="text-[11px] text-slate-500">
                When and how severely anomalies were detected
              </p>
            </div>
          </div>
          <div className="flex items-center gap-3 text-[11px] font-medium text-slate-500">
            <Legend color="#f43f5e" label="High" />
            <Legend color="#f59e0b" label="Medium" />
            <Legend color="#eab308" label="Low" />
          </div>
        </div>
        <div className="px-3 pb-5">
          <ActiveEventsTimeline events={crises} />
        </div>
      </section>

      {/* Incidents */}
      <section className="space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="flex h-7 w-7 items-center justify-center rounded-md bg-rose-50 text-rose-600">
              <Siren className="h-4 w-4" strokeWidth={2} />
            </div>
            <h2 className="text-sm font-semibold text-slate-900">Detected Incidents</h2>
            <span className="inline-flex items-center rounded-full bg-slate-100 px-2 py-0.5 text-[10px] font-semibold tabular-nums text-slate-600">
              {crises.length}
            </span>
          </div>
        </div>
        <IncidentDetailCards incidents={crises} />
      </section>
    </div>
  );
}

function Legend({ color, label }: { color: string; label: string }) {
  return (
    <div className="flex items-center gap-1.5">
      <span className="h-2 w-2 rounded-full" style={{ backgroundColor: color }} />
      {label}
    </div>
  );
}
