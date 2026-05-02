import {
  TrendingUp,
  TrendingDown,
  CheckCircle2,
  MessagesSquare,
  ShieldAlert,
  Flame,
} from "lucide-react";
import { SentimentDonutChart } from "@/components/dashboard/sentiment-donut";
import { TrendingTopicsTable } from "@/components/dashboard/trending-topics-table";

const isAlert = true;

export default function OverviewPage() {
  return (
    <div className="space-y-8">
      {/* Page header */}
      <div className="flex flex-col sm:flex-row sm:items-end sm:justify-between gap-4">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.08em] text-indigo-600">
            Dashboard
          </p>
          <h1 className="mt-1 text-3xl font-bold tracking-tight text-slate-900">
            Overview
          </h1>
          <p className="mt-1.5 text-sm text-slate-500">
            Real-time pulse on Vietnamese tech conversation across forums, news, and video.
          </p>
        </div>
        <div className="flex items-center gap-2 text-xs text-slate-500">
          <span className="inline-flex h-2 w-2 rounded-full bg-emerald-500 animate-pulse" />
          Live
          <span className="text-slate-300">·</span>
          Updated 2 min ago
        </div>
      </div>

      {/* KPI strip */}
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-5">
        {/* Daily Mention Count */}
        <div className="relative overflow-hidden rounded-xl bg-white border border-slate-200/80 shadow-[0_1px_2px_rgba(15,23,42,0.04)] hover:shadow-[0_4px_16px_-4px_rgba(15,23,42,0.06)] transition-shadow p-6">
          <div className="flex items-start justify-between">
            <div>
              <p className="text-[11px] font-semibold uppercase tracking-[0.08em] text-slate-500">
                Daily Mentions
              </p>
              <p className="mt-3 text-4xl font-bold tabular-nums tracking-tight text-slate-900">
                24,531
              </p>
              <div className="mt-2 inline-flex items-center gap-1.5 rounded-full bg-emerald-50 px-2 py-0.5 text-xs font-semibold text-emerald-700">
                <TrendingUp className="h-3.5 w-3.5" strokeWidth={2.5} />
                +12.5%
                <span className="font-medium text-emerald-600/80">vs yesterday</span>
              </div>
            </div>
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-indigo-50 text-indigo-600 ring-1 ring-inset ring-indigo-100">
              <MessagesSquare className="h-5 w-5" strokeWidth={2} />
            </div>
          </div>
        </div>

        {/* Total Crisis Count */}
        <div className="relative overflow-hidden rounded-xl bg-white border border-slate-200/80 shadow-[0_1px_2px_rgba(15,23,42,0.04)] hover:shadow-[0_4px_16px_-4px_rgba(15,23,42,0.06)] transition-shadow p-6">
          <div className="flex items-start justify-between">
            <div>
              <p className="text-[11px] font-semibold uppercase tracking-[0.08em] text-slate-500">
                Active Crises
              </p>
              <p className="mt-3 text-4xl font-bold tabular-nums tracking-tight text-slate-900">
                3
              </p>
              <div className="mt-2 inline-flex items-center gap-1.5 rounded-full bg-rose-50 px-2 py-0.5 text-xs font-semibold text-rose-700">
                <TrendingDown className="h-3.5 w-3.5 rotate-180" strokeWidth={2.5} />
                +1 detected today
              </div>
            </div>
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-amber-50 text-amber-600 ring-1 ring-inset ring-amber-100">
              <ShieldAlert className="h-5 w-5" strokeWidth={2} />
            </div>
          </div>
        </div>

        {/* System Heat Indicator */}
        <div
          className={`relative overflow-hidden rounded-xl border shadow-[0_1px_2px_rgba(15,23,42,0.04)] hover:shadow-[0_4px_16px_-4px_rgba(15,23,42,0.06)] transition-shadow p-6 ${
            isAlert
              ? "bg-gradient-to-br from-rose-50 via-orange-50/70 to-white border-rose-200/60"
              : "bg-gradient-to-br from-emerald-50 via-teal-50/70 to-white border-emerald-200/60"
          }`}
        >
          {/* Decorative orb */}
          <div
            className={`absolute -top-8 -right-8 h-28 w-28 rounded-full blur-2xl opacity-50 ${
              isAlert ? "bg-rose-200" : "bg-emerald-200"
            }`}
          />
          <div className="relative flex items-start justify-between">
            <div>
              <p
                className={`text-[11px] font-semibold uppercase tracking-[0.08em] ${
                  isAlert ? "text-rose-700/80" : "text-emerald-700/80"
                }`}
              >
                System Heat
              </p>
              <div className="mt-3 flex items-baseline gap-2">
                <p
                  className={`text-3xl font-bold tracking-tight ${
                    isAlert ? "text-rose-700" : "text-emerald-700"
                  }`}
                >
                  {isAlert ? "Alert" : "Normal"}
                </p>
                <span className="text-xs font-medium text-slate-500">level</span>
              </div>
              <p className="mt-2 max-w-[16rem] text-xs font-medium text-slate-600">
                {isAlert
                  ? "Crisis volume above 2σ baseline. Review incidents."
                  : "All signals are within healthy thresholds."}
              </p>
            </div>
            <div
              className={`flex h-10 w-10 items-center justify-center rounded-lg ring-1 ring-inset ${
                isAlert
                  ? "bg-rose-100/80 text-rose-600 ring-rose-200/60"
                  : "bg-emerald-100/80 text-emerald-600 ring-emerald-200/60"
              }`}
            >
              {isAlert ? (
                <Flame className="h-5 w-5" strokeWidth={2} />
              ) : (
                <CheckCircle2 className="h-5 w-5" strokeWidth={2} />
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Charts row */}
      <div className="grid grid-cols-1 xl:grid-cols-12 gap-5">
        <SentimentDonutChart className="xl:col-span-4" />
        <TrendingTopicsTable className="xl:col-span-8" />
      </div>
    </div>
  );
}
