"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  Radar,
  LayoutDashboard,
  LineChart,
  ShieldAlert,
  Settings,
  CircleHelp,
} from "lucide-react";

const primaryNav = [
  { href: "/", label: "Overview", icon: LayoutDashboard },
  { href: "/trends", label: "Trends Explorer", icon: LineChart },
  { href: "/crises", label: "Crisis Monitor", icon: ShieldAlert },
];

const secondaryNav = [
  { href: "#", label: "Settings", icon: Settings },
  { href: "#", label: "Help", icon: CircleHelp },
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="w-64 bg-white border-r border-slate-200 flex flex-col h-screen fixed left-0 top-0 z-30">
      {/* Brand */}
      <div className="h-16 flex items-center px-6 border-b border-slate-200">
        <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-gradient-to-br from-indigo-500 to-violet-600 shadow-sm shadow-indigo-200/60">
          <Radar className="h-5 w-5 text-white" strokeWidth={2.25} />
        </div>
        <div className="ml-3 leading-tight">
          <div className="text-[15px] font-semibold tracking-tight text-slate-900">
            TechRadar
          </div>
          <div className="text-[11px] font-medium text-slate-500">VN Tech Pulse</div>
        </div>
      </div>

      {/* Nav */}
      <nav className="flex-1 px-3 py-4 space-y-0.5">
        <p className="px-3 py-2 text-[10px] font-semibold uppercase tracking-[0.08em] text-slate-400">
          Workspace
        </p>
        {primaryNav.map((item) => {
          const isActive =
            item.href === "/"
              ? pathname === "/"
              : pathname.startsWith(item.href);
          const Icon = item.icon;
          return (
            <Link
              key={item.href}
              href={item.href}
              className={`group relative flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-colors ${
                isActive
                  ? "bg-gradient-to-r from-indigo-50 to-violet-50/60 text-indigo-700"
                  : "text-slate-600 hover:bg-slate-50 hover:text-slate-900"
              }`}
            >
              {isActive && (
                <span className="absolute inset-y-1.5 left-0 w-0.5 rounded-full bg-gradient-to-b from-indigo-500 to-violet-500" />
              )}
              <Icon
                className={`h-4 w-4 ${
                  isActive ? "text-indigo-600" : "text-slate-500 group-hover:text-slate-700"
                }`}
                strokeWidth={2}
              />
              {item.label}
            </Link>
          );
        })}

        <p className="mt-6 px-3 py-2 text-[10px] font-semibold uppercase tracking-[0.08em] text-slate-400">
          System
        </p>
        {secondaryNav.map((item) => {
          const Icon = item.icon;
          return (
            <Link
              key={item.label}
              href={item.href}
              className="group flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium text-slate-600 hover:bg-slate-50 hover:text-slate-900 transition-colors"
            >
              <Icon className="h-4 w-4 text-slate-500 group-hover:text-slate-700" strokeWidth={2} />
              {item.label}
            </Link>
          );
        })}
      </nav>

      {/* Footer profile */}
      <div className="border-t border-slate-200 p-3">
        <div className="flex items-center gap-3 rounded-lg p-2 hover:bg-slate-50 transition-colors cursor-pointer">
          <div className="flex h-9 w-9 items-center justify-center rounded-full bg-gradient-to-br from-slate-200 to-slate-100 text-sm font-semibold text-slate-700">
            HT
          </div>
          <div className="min-w-0 flex-1 leading-tight">
            <div className="truncate text-sm font-medium text-slate-900">Huy Tran</div>
            <div className="truncate text-xs text-slate-500">Analyst</div>
          </div>
        </div>
      </div>
    </aside>
  );
}
