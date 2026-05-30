import type { CSSProperties } from "react";

export function ChartCardLoading({
  titleWidth = "w-36",
  accentClassName = "bg-indigo-100",
}: {
  titleWidth?: string;
  accentClassName?: string;
}) {
  return (
    <div className="flex h-[340px] flex-col rounded-xl border border-slate-200/80 bg-white shadow-[0_1px_2px_rgba(15,23,42,0.04)]" aria-busy="true">
      <div className="flex items-start justify-between px-6 pb-2 pt-5">
        <div className="flex items-center gap-2.5">
          <Skeleton className={`h-7 w-7 rounded-md ${accentClassName}`} />
          <div>
            <Skeleton className={`h-4 ${titleWidth}`} />
            <Skeleton className="mt-2 h-3 w-44" />
          </div>
        </div>
        <Skeleton className="h-6 w-14" />
      </div>
      <div className="flex-1 px-6 pb-5 pt-4">
        <div className="relative h-full overflow-hidden rounded-lg border border-slate-100 bg-slate-50/80 p-4">
          <div className="absolute inset-x-4 bottom-4 top-4 flex items-end gap-2">
            {Array.from({ length: 14 }).map((_, idx) => (
              <Skeleton
                key={idx}
                className="flex-1 rounded-t-md"
                style={{ height: `${28 + ((idx * 17) % 58)}%` }}
              />
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

export function WordCloudLoading() {
  const items = [
    { width: "6rem", height: "2rem" },
    { width: "4rem", height: "1.25rem" },
    { width: "8rem", height: "1.5rem" },
    { width: "5rem", height: "2rem" },
    { width: "7rem", height: "1.25rem" },
    { width: "3.5rem", height: "1.5rem" },
    { width: "9rem", height: "2rem" },
    { width: "6rem", height: "1.5rem" },
    { width: "4.5rem", height: "1.25rem" },
  ];

  return (
    <div className="flex min-h-[280px] w-full flex-wrap items-center justify-center gap-x-4 gap-y-4 rounded-lg border border-slate-100 bg-slate-50/80 p-6" aria-busy="true">
      {items.map((item, idx) => (
        <Skeleton
          key={idx}
          style={item}
        />
      ))}
    </div>
  );
}

export function EvidencePostsLoading() {
  return (
    <div className="flex-1 space-y-3 overflow-hidden" aria-busy="true">
      {Array.from({ length: 4 }).map((_, idx) => (
        <div key={idx} className="rounded-lg border border-slate-200/80 bg-white p-4">
          <div className="mb-3 flex items-start justify-between gap-3">
            <div className="flex min-w-0 items-center gap-2.5">
              <Skeleton className="h-7 w-7 rounded-md" />
              <div>
                <Skeleton className="h-4 w-28" />
                <Skeleton className="mt-2 h-3 w-16" />
              </div>
            </div>
            <Skeleton className="h-5 w-16 rounded-full" />
          </div>
          <Skeleton className="h-4 w-full" />
          <Skeleton className="mt-2 h-4 w-5/6" />
          <Skeleton className="mt-4 h-4 w-32" />
        </div>
      ))}
    </div>
  );
}

function Skeleton({
  className = "",
  style,
}: {
  className?: string;
  style?: CSSProperties;
}) {
  return (
    <span
      className={`block animate-pulse rounded bg-slate-200/80 ${className}`}
      style={style}
    />
  );
}
