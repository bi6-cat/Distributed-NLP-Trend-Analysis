"use client";

import { Button } from "@/components/ui/button";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Activity,
  Gauge,
  Eye,
  MessageSquare,
  Newspaper,
  Video,
  ChevronRight,
  Zap,
  type LucideIcon,
} from "lucide-react";

const incidents = [
  {
    event_id: "EVT-1001",
    severity: "HIGH",
    detected_at: "Today, 08:30",
    affected_topics: ["Shopee Data Breach Rumor", "E-commerce Security"],
    trigger_conditions: ["volume_zscore > 3.0", "neg_ratio > 0.8"],
    neg_ratio: 85,
    mention_velocity: "+450/hr",
    anomaly_score: 0.92,
    evidence_posts: [
      { id: 1, author: "Security_Expert", text: "Dữ liệu hơn 10 triệu thẻ tín dụng nghi bị rò rỉ từ sàn S.", type: "forum", engagement: 5200 },
      { id: 2, author: "Báo Pháp Luật", text: "Tin đồn: Một nền tảng TMĐT lớn bị hacker tấn công.", type: "news", engagement: 3100 },
    ],
  },
  {
    event_id: "EVT-1002",
    severity: "HIGH",
    detected_at: "Today, 16:20",
    affected_topics: ["FPT Play Cyber Attack", "Streaming Service"],
    trigger_conditions: ["volume_zscore > 2.5", "acceleration > 50"],
    neg_ratio: 72,
    mention_velocity: "+320/hr",
    anomaly_score: 0.84,
    evidence_posts: [
      { id: 3, author: "TinTucCongNghe", text: "FPT Play sập toàn hệ thống tối chủ nhật ngay lúc có bóng đá.", type: "news", engagement: 4500 },
      { id: 4, author: "nguoidung_fpt", text: "Không thể đăng nhập được từ web hay app TV.", type: "forum", engagement: 1200 },
    ],
  },
  {
    event_id: "EVT-1003",
    severity: "MEDIUM",
    detected_at: "Today, 10:15",
    affected_topics: ["Momo App Outage", "Digital Payment"],
    trigger_conditions: ["neg_ratio_zscore > 2.0"],
    neg_ratio: 65,
    mention_velocity: "+150/hr",
    anomaly_score: 0.61,
    evidence_posts: [
      { id: 5, author: "ThanhToanLoi", text: "Chuyển tiền qua Momo bị trừ tiền nhưng bên kia chưa nhận được.", type: "forum", engagement: 890 },
      { id: 6, author: "TechVlogger", text: "Lỗi kết nối ngân hàng của Momo chiều nay.", type: "video", engagement: 670 },
    ],
  },
];

const severityMeta = {
  HIGH: {
    cardClass: "border-rose-200/70",
    accentBar: "bg-gradient-to-b from-rose-500 to-rose-400",
    badgeClass: "bg-gradient-to-r from-rose-50 to-orange-50 text-rose-700 ring-rose-200/60",
    valueText: "text-rose-700",
    icon: "text-rose-500",
  },
  MEDIUM: {
    cardClass: "border-amber-200/70",
    accentBar: "bg-gradient-to-b from-amber-500 to-amber-400",
    badgeClass: "bg-gradient-to-r from-amber-50 to-yellow-50 text-amber-700 ring-amber-200/60",
    valueText: "text-amber-700",
    icon: "text-amber-500",
  },
  LOW: {
    cardClass: "border-yellow-200/70",
    accentBar: "bg-gradient-to-b from-yellow-400 to-yellow-300",
    badgeClass: "bg-yellow-50 text-yellow-800 ring-yellow-200/60",
    valueText: "text-yellow-800",
    icon: "text-yellow-500",
  },
} as const;

const sourceMeta = {
  forum: { Icon: MessageSquare, label: "Forum", className: "bg-blue-50 text-blue-600 ring-blue-100" },
  video: { Icon: Video, label: "Video", className: "bg-rose-50 text-rose-600 ring-rose-100" },
  news: { Icon: Newspaper, label: "News", className: "bg-emerald-50 text-emerald-600 ring-emerald-100" },
};

export function IncidentDetailCards() {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-5">
      {incidents.map((incident) => {
        const meta = severityMeta[incident.severity as keyof typeof severityMeta];
        return (
          <article
            key={incident.event_id}
            className={`relative overflow-hidden rounded-xl bg-white border ${meta.cardClass} shadow-[0_1px_2px_rgba(15,23,42,0.04)] hover:shadow-[0_8px_24px_-8px_rgba(15,23,42,0.12)] transition-all`}
          >
            {/* Severity bar */}
            <span className={`absolute inset-y-0 left-0 w-1 ${meta.accentBar}`} />

            {/* Header */}
            <div className="px-5 pt-5 pb-3">
              <div className="flex items-start justify-between mb-3">
                <span
                  className={`inline-flex items-center gap-1 rounded-full px-2.5 py-0.5 text-[10px] font-bold uppercase tracking-[0.08em] ring-1 ring-inset ${meta.badgeClass}`}
                >
                  <Zap className="h-3 w-3" strokeWidth={2.5} />
                  {incident.severity}
                </span>
                <div className="text-right">
                  <p className="text-[10px] font-mono text-slate-400">{incident.event_id}</p>
                  <p className="text-[10px] text-slate-400">{incident.detected_at}</p>
                </div>
              </div>

              <div className="space-y-1.5">
                {incident.affected_topics.map((topic, i) => (
                  <h3
                    key={i}
                    className="flex items-center gap-1.5 text-sm font-semibold text-slate-900 leading-snug"
                  >
                    <ChevronRight className="h-3 w-3 text-slate-400 shrink-0" strokeWidth={2.5} />
                    {topic}
                  </h3>
                ))}
              </div>
            </div>

            {/* Body */}
            <div className="px-5 pb-4 space-y-4">
              {/* Trigger conditions */}
              <div>
                <h4 className="text-[10px] font-semibold uppercase tracking-[0.08em] text-slate-500 mb-1.5">
                  Trigger Conditions
                </h4>
                <div className="flex flex-wrap gap-1.5">
                  {incident.trigger_conditions.map((cond, i) => (
                    <span
                      key={i}
                      className="inline-flex items-center rounded-md bg-slate-50 ring-1 ring-inset ring-slate-200/60 px-2 py-0.5 text-[11px] font-mono font-medium text-slate-600"
                    >
                      {cond}
                    </span>
                  ))}
                </div>
              </div>

              {/* Metrics */}
              <div className="grid grid-cols-3 gap-2">
                <Metric
                  Icon={Activity}
                  label="Neg ratio"
                  value={`${incident.neg_ratio}%`}
                  iconClassName={meta.icon}
                  valueClassName={meta.valueText}
                />
                <Metric
                  Icon={Gauge}
                  label="Velocity"
                  value={incident.mention_velocity}
                  iconClassName={meta.icon}
                  valueClassName={meta.valueText}
                />
                <Metric
                  Icon={Zap}
                  label="Anomaly"
                  value={incident.anomaly_score.toFixed(2)}
                  iconClassName={meta.icon}
                  valueClassName={meta.valueText}
                />
              </div>
            </div>

            {/* Footer */}
            <div className="px-5 py-3 bg-slate-50/60 border-t border-slate-100 flex items-center justify-between">
              <span className="text-[11px] font-medium text-slate-500">
                {incident.evidence_posts.length} evidence posts
              </span>
              <Sheet>
                <SheetTrigger
                  render={
                    <Button
                      variant="ghost"
                      size="sm"
                      className="h-7 px-2.5 text-xs font-semibold text-indigo-600 hover:text-indigo-700 hover:bg-indigo-50"
                    />
                  }
                >
                  <Eye className="h-3.5 w-3.5 mr-1.5" />
                  View Evidence
                </SheetTrigger>
                <SheetContent className="sm:max-w-md w-full">
                  <SheetHeader className="border-b border-slate-100 pb-4">
                    <span
                      className={`inline-flex w-fit items-center gap-1 rounded-full px-2.5 py-0.5 text-[10px] font-bold uppercase tracking-[0.08em] ring-1 ring-inset ${meta.badgeClass}`}
                    >
                      <Zap className="h-3 w-3" strokeWidth={2.5} />
                      {incident.severity} Severity
                    </span>
                    <SheetTitle className="text-base font-semibold text-slate-900">
                      Evidence: {incident.event_id}
                    </SheetTitle>
                    <SheetDescription className="text-xs text-slate-500">
                      Posts that triggered the anomaly detector for{" "}
                      {incident.affected_topics[0]}
                    </SheetDescription>
                  </SheetHeader>
                  <ScrollArea className="h-[calc(100vh-180px)] px-1">
                    <div className="space-y-3 py-4 pr-3">
                      {incident.evidence_posts.map((post) => {
                        const src = sourceMeta[post.type as keyof typeof sourceMeta];
                        const SourceIcon = src.Icon;
                        return (
                          <div
                            key={post.id}
                            className="rounded-lg border border-slate-200/80 bg-white p-4 hover:border-slate-300 transition-colors"
                          >
                            <div className="flex items-start justify-between gap-3 mb-2">
                              <div className="flex items-center gap-2.5 min-w-0">
                                <div
                                  className={`flex h-7 w-7 items-center justify-center rounded-md ring-1 ring-inset shrink-0 ${src.className}`}
                                >
                                  <SourceIcon className="h-3.5 w-3.5" strokeWidth={2} />
                                </div>
                                <div className="min-w-0">
                                  <p className="truncate text-sm font-semibold text-slate-900">
                                    {post.author}
                                  </p>
                                  <p className="text-[11px] font-medium text-slate-500">
                                    {src.label}
                                  </p>
                                </div>
                              </div>
                              <span className="text-[11px] font-semibold tabular-nums text-slate-500 shrink-0">
                                {post.engagement.toLocaleString()}
                              </span>
                            </div>
                            <p className="text-sm leading-relaxed text-slate-700">
                              {post.text}
                            </p>
                          </div>
                        );
                      })}
                    </div>
                  </ScrollArea>
                </SheetContent>
              </Sheet>
            </div>
          </article>
        );
      })}
    </div>
  );
}

function Metric({
  Icon,
  label,
  value,
  iconClassName,
  valueClassName,
}: {
  Icon: LucideIcon;
  label: string;
  value: string;
  iconClassName: string;
  valueClassName: string;
}) {
  return (
    <div className="rounded-lg bg-slate-50/60 ring-1 ring-inset ring-slate-200/40 px-2.5 py-2">
      <div className="flex items-center gap-1 mb-0.5">
        <Icon className={`h-3 w-3 ${iconClassName}`} strokeWidth={2.5} />
        <span className="text-[10px] font-semibold uppercase tracking-[0.06em] text-slate-500">
          {label}
        </span>
      </div>
      <p className={`text-sm font-bold tabular-nums ${valueClassName}`}>{value}</p>
    </div>
  );
}
