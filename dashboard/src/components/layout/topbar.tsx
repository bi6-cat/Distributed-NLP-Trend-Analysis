"use client";

import { useEffect, useMemo, useState, useTransition } from "react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { Search, Bell, Calendar } from "lucide-react";
import {
  TIME_RANGE_PRESETS,
  isDateOnly,
  normalizeTimeRangeMode,
  type TimeRangeMode,
} from "@/lib/time-range";

export function Topbar() {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const [isPending, startTransition] = useTransition();

  const selectedMode = normalizeTimeRangeMode(searchParams.get("range") ?? undefined);
  const [draftMode, setDraftMode] = useState<TimeRangeMode>(selectedMode);
  const [fromDate, setFromDate] = useState(searchParams.get("from") ?? "");
  const [toDate, setToDate] = useState(searchParams.get("to") ?? "");

  useEffect(() => {
    setDraftMode(selectedMode);
    setFromDate(searchParams.get("from") ?? "");
    setToDate(searchParams.get("to") ?? "");
  }, [searchParams, selectedMode]);

  const selectedLabel = useMemo(
    () => TIME_RANGE_PRESETS.find((preset) => preset.value === selectedMode)?.label ?? TIME_RANGE_PRESETS[0].label,
    [selectedMode],
  );

  function updateRange(mode: TimeRangeMode, from?: string, to?: string) {
    const params = new URLSearchParams(searchParams.toString());
    params.set("range", mode);

    if (mode === "custom") {
      params.set("from", from ?? "");
      params.set("to", to ?? "");
    } else {
      params.delete("from");
      params.delete("to");
    }

    const query = params.toString();
    startTransition(() => {
      router.push(query ? `${pathname}?${query}` : pathname);
    });
  }

  function handleModeChange(nextMode: TimeRangeMode) {
    setDraftMode(nextMode);
    if (nextMode !== "custom") {
      updateRange(nextMode);
    }
  }

  const canApplyCustom = isDateOnly(fromDate) && isDateOnly(toDate);

  return (
    <header className="sticky top-0 z-20 bg-white/80 backdrop-blur supports-[backdrop-filter]:bg-white/70 border-b border-slate-200">
      <div className="h-16 px-8 lg:px-10 flex items-center gap-4">
        <div className="relative flex-1 max-w-md">
          <Search className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-400" />
          <input
            type="text"
            placeholder="Search topics, posts, events..."
            className="w-full h-9 pl-9 pr-4 rounded-lg bg-slate-50 border border-slate-200/80 text-sm text-slate-700 placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-300 focus:bg-white transition-all"
          />
        </div>

        <div className="ml-auto flex items-center gap-2">
          <div className="flex items-center gap-2">
            <div className="relative">
              <Calendar className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-500" />
              <select
                value={draftMode}
                onChange={(event) => handleModeChange(event.target.value as TimeRangeMode)}
                aria-label="Time range"
                className="h-9 rounded-lg border border-slate-200 bg-white pl-9 pr-8 text-sm font-medium text-slate-700 outline-none transition-colors hover:bg-slate-50 focus:border-indigo-300 focus:ring-2 focus:ring-indigo-500/20"
              >
                {TIME_RANGE_PRESETS.map((preset) => (
                  <option key={preset.value} value={preset.value}>
                    {preset.label}
                  </option>
                ))}
              </select>
            </div>

            {draftMode === "custom" && (
              <div className="flex items-center gap-1.5">
                <input
                  type="date"
                  value={fromDate}
                  onChange={(event) => setFromDate(event.target.value)}
                  aria-label="Start date"
                  className="h-9 rounded-lg border border-slate-200 bg-white px-2 text-sm font-medium text-slate-700 outline-none focus:border-indigo-300 focus:ring-2 focus:ring-indigo-500/20"
                />
                <span className="text-xs font-medium text-slate-400">to</span>
                <input
                  type="date"
                  value={toDate}
                  onChange={(event) => setToDate(event.target.value)}
                  aria-label="End date"
                  className="h-9 rounded-lg border border-slate-200 bg-white px-2 text-sm font-medium text-slate-700 outline-none focus:border-indigo-300 focus:ring-2 focus:ring-indigo-500/20"
                />
                <button
                  type="button"
                  disabled={!canApplyCustom || isPending}
                  onClick={() => updateRange("custom", fromDate, toDate)}
                  className="h-9 rounded-lg bg-indigo-600 px-3 text-sm font-semibold text-white transition-colors hover:bg-indigo-700 disabled:cursor-not-allowed disabled:bg-slate-300"
                >
                  Apply
                </button>
              </div>
            )}
          </div>

          <div className="hidden items-center gap-2 h-9 rounded-lg border border-slate-200 bg-white px-3 text-sm font-medium text-slate-700 lg:inline-flex">
            <Calendar className="h-4 w-4 text-slate-500" />
            {selectedLabel}
          </div>
          <button className="relative inline-flex items-center justify-center h-9 w-9 rounded-lg border border-slate-200 bg-white text-slate-600 hover:bg-slate-50 transition-colors">
            <Bell className="h-4 w-4" />
            <span className="absolute top-1.5 right-1.5 h-2 w-2 rounded-full bg-rose-500 ring-2 ring-white" />
          </button>
        </div>
      </div>
    </header>
  );
}
