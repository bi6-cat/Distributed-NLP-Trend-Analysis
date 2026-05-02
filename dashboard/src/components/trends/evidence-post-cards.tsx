"use client";

import { MessageSquare, Video, Newspaper, Heart } from "lucide-react";

const posts = [
  {
    id: 1,
    author_name: "nguyen_van_a",
    source_type: "forum",
    sentiment: "positive",
    body: "Apple M4 năm nay quá khủng khiếp, hiệu năng render video vượt xa cả M3 Max trên một thiết bị mỏng nhẹ như vậy.",
    engagement: 1450,
  },
  {
    id: 2,
    author_name: "tech_review_vn",
    source_type: "video",
    sentiment: "neutral",
    body: "Đánh giá chi tiết iPad Pro M4 mới: Màn hình OLED kép ấn tượng, nhưng giá bán còn khá cao so với mặt bằng chung.",
    engagement: 890,
  },
  {
    id: 3,
    author_name: "Tin Tức Số",
    source_type: "news",
    sentiment: "negative",
    body: "Nhiều người dùng phàn nàn máy dễ bị quá nhiệt khi chạy các tác vụ AI liên tục trong thời gian dài.",
    engagement: 620,
  },
  {
    id: 4,
    author_name: "ifam_vietnam",
    source_type: "forum",
    sentiment: "positive",
    body: "Apple Intelligence thực sự thay đổi cách mình làm việc hàng ngày, tóm tắt email và viết nháp siêu nhanh.",
    engagement: 530,
  },
  {
    id: 5,
    author_name: "Hoàng Minh",
    source_type: "forum",
    sentiment: "negative",
    body: "Nâng cấp từ M2 lên M4 không thấy khác biệt nhiều với người dùng văn phòng cơ bản, tốn tiền vô ích.",
    engagement: 410,
  },
];

const sourceMeta = {
  forum: { Icon: MessageSquare, label: "Forum", className: "bg-blue-50 text-blue-600 ring-blue-100" },
  video: { Icon: Video, label: "Video", className: "bg-rose-50 text-rose-600 ring-rose-100" },
  news: { Icon: Newspaper, label: "News", className: "bg-emerald-50 text-emerald-600 ring-emerald-100" },
};

const sentimentMeta = {
  positive: {
    label: "Positive",
    className: "bg-emerald-50 text-emerald-700 ring-emerald-200/60",
    accent: "border-l-emerald-400",
  },
  negative: {
    label: "Negative",
    className: "bg-rose-50 text-rose-700 ring-rose-200/60",
    accent: "border-l-rose-400",
  },
  neutral: {
    label: "Neutral",
    className: "bg-slate-100 text-slate-700 ring-slate-200/60",
    accent: "border-l-slate-300",
  },
};

export function EvidencePostCards() {
  return (
    <div className="flex-1 overflow-y-auto thin-scrollbar -mr-2 pr-2">
      <div className="space-y-3">
        {posts.map((post) => {
          const src = sourceMeta[post.source_type as keyof typeof sourceMeta];
          const sent = sentimentMeta[post.sentiment as keyof typeof sentimentMeta];
          const SourceIcon = src.Icon;
          return (
            <article
              key={post.id}
              className={`group rounded-lg border border-slate-200/80 border-l-[3px] ${sent.accent} bg-white p-4 hover:shadow-[0_4px_12px_-4px_rgba(15,23,42,0.08)] hover:border-slate-300/80 transition-all`}
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
                      {post.author_name}
                    </p>
                    <p className="text-[11px] font-medium text-slate-500">{src.label}</p>
                  </div>
                </div>
                <span
                  className={`inline-flex items-center rounded-full px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide ring-1 ring-inset ${sent.className}`}
                >
                  {sent.label}
                </span>
              </div>
              <p className="text-sm leading-relaxed text-slate-700 line-clamp-2">
                {post.body}
              </p>
              <div className="mt-3 flex items-center gap-1.5 text-xs text-slate-500">
                <Heart className="h-3.5 w-3.5 text-slate-400" />
                <span className="font-medium tabular-nums">
                  {post.engagement.toLocaleString()}
                </span>
                <span className="text-slate-300">·</span>
                <span>engagement</span>
              </div>
            </article>
          );
        })}
      </div>
    </div>
  );
}
