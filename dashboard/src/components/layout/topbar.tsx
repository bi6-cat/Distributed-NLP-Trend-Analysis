"use client";

import { Search, Bell, Calendar } from "lucide-react";

export function Topbar() {
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
          <button className="inline-flex items-center gap-2 h-9 rounded-lg border border-slate-200 bg-white px-3 text-sm font-medium text-slate-700 hover:bg-slate-50 transition-colors">
            <Calendar className="h-4 w-4 text-slate-500" />
            Last 24 hours
          </button>
          <button className="relative inline-flex items-center justify-center h-9 w-9 rounded-lg border border-slate-200 bg-white text-slate-600 hover:bg-slate-50 transition-colors">
            <Bell className="h-4 w-4" />
            <span className="absolute top-1.5 right-1.5 h-2 w-2 rounded-full bg-rose-500 ring-2 ring-white" />
          </button>
        </div>
      </div>
    </header>
  );
}
