type PageLoadingVariant = "overview" | "trends" | "crises";

export function PageLoading({ variant = "overview" }: { variant?: PageLoadingVariant }) {
  if (variant === "trends") {
    return <TrendsLoading />;
  }

  if (variant === "crises") {
    return <CrisesLoading />;
  }

  return <OverviewLoading />;
}

function OverviewLoading() {
  return (
    <div className="space-y-8" aria-busy="true" aria-live="polite">
      <HeaderLoading />
      <div className="grid grid-cols-1 gap-5 md:grid-cols-2 xl:grid-cols-3">
        <KpiLoading tone="indigo" />
        <KpiLoading tone="rose" />
        <KpiLoading tone="emerald" />
      </div>
      <div className="grid grid-cols-1 gap-5 xl:grid-cols-12">
        <PanelLoading className="xl:col-span-4 min-h-[390px]" />
        <TableLoading className="xl:col-span-8" rows={7} />
      </div>
    </div>
  );
}

function TrendsLoading() {
  return (
    <div className="space-y-6" aria-busy="true" aria-live="polite">
      <HeaderLoading />
      <div className="grid grid-cols-1 gap-5 xl:grid-cols-12">
        <aside className="xl:col-span-3 rounded-xl border border-slate-200/80 bg-white shadow-[0_1px_2px_rgba(15,23,42,0.04)]">
          <div className="border-b border-slate-100 p-4">
            <div className="mb-3 flex items-center justify-between">
              <Skeleton className="h-4 w-20" />
              <Skeleton className="h-3 w-14" />
            </div>
            <Skeleton className="h-9 w-full rounded-lg" />
          </div>
          <div className="space-y-2 p-3">
            {Array.from({ length: 9 }).map((_, idx) => (
              <div key={idx} className="rounded-lg px-3 py-2.5">
                <div className="flex items-start justify-between gap-3">
                  <Skeleton className="h-4 w-3/4" />
                  <Skeleton className="h-5 w-8 rounded-md" />
                </div>
                <Skeleton className="mt-2 h-3 w-2/3" />
              </div>
            ))}
          </div>
        </aside>
        <div className="space-y-5 xl:col-span-9">
          <PanelLoading className="min-h-[156px]" />
          <div className="grid grid-cols-1 gap-5 xl:grid-cols-2">
            <PanelLoading className="min-h-[340px]" />
            <PanelLoading className="min-h-[340px]" />
          </div>
          <div className="grid grid-cols-1 gap-5 xl:grid-cols-12">
            <PanelLoading className="min-h-[360px] xl:col-span-5" />
            <TableLoading className="min-h-[480px] xl:col-span-7" rows={4} />
          </div>
        </div>
      </div>
    </div>
  );
}

function CrisesLoading() {
  return (
    <div className="space-y-6" aria-busy="true" aria-live="polite">
      <HeaderLoading />
      <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
        {Array.from({ length: 4 }).map((_, idx) => (
          <KpiLoading key={idx} compact tone={idx === 1 ? "rose" : "slate"} />
        ))}
      </div>
      <PanelLoading className="min-h-[350px]" />
      <div className="grid grid-cols-1 gap-5 md:grid-cols-2 xl:grid-cols-3">
        {Array.from({ length: 3 }).map((_, idx) => (
          <PanelLoading key={idx} className="min-h-[260px]" />
        ))}
      </div>
    </div>
  );
}

function HeaderLoading() {
  return (
    <div className="flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
      <div>
        <Skeleton className="h-3 w-24" />
        <Skeleton className="mt-3 h-9 w-56" />
        <Skeleton className="mt-3 h-4 w-full max-w-[520px]" />
      </div>
      <div className="flex items-center gap-2">
        <Skeleton className="h-2 w-2 rounded-full" />
        <Skeleton className="h-4 w-28" />
      </div>
    </div>
  );
}

function KpiLoading({
  compact = false,
  tone,
}: {
  compact?: boolean;
  tone: "slate" | "indigo" | "rose" | "emerald";
}) {
  const toneClass = {
    slate: "bg-slate-100",
    indigo: "bg-indigo-100",
    rose: "bg-rose-100",
    emerald: "bg-emerald-100",
  }[tone];

  return (
    <div className="rounded-xl border border-slate-200/80 bg-white p-6 shadow-[0_1px_2px_rgba(15,23,42,0.04)]">
      <div className="flex items-start justify-between">
        <div>
          <Skeleton className="h-3 w-28" />
          <Skeleton className={`${compact ? "mt-3 h-7 w-16" : "mt-4 h-10 w-28"}`} />
          <Skeleton className="mt-3 h-5 w-32 rounded-full" />
        </div>
        <Skeleton className={`h-10 w-10 rounded-lg ${toneClass}`} />
      </div>
    </div>
  );
}

function PanelLoading({ className = "" }: { className?: string }) {
  return (
    <div className={`rounded-xl border border-slate-200/80 bg-white p-6 shadow-[0_1px_2px_rgba(15,23,42,0.04)] ${className}`}>
      <div className="flex items-start justify-between">
        <div className="flex items-center gap-2.5">
          <Skeleton className="h-7 w-7 rounded-md" />
          <div>
            <Skeleton className="h-4 w-36" />
            <Skeleton className="mt-2 h-3 w-44" />
          </div>
        </div>
        <Skeleton className="h-6 w-16" />
      </div>
      <div className="mt-8 space-y-4">
        <Skeleton className="h-36 w-full rounded-lg" />
        <div className="grid grid-cols-4 gap-3">
          <Skeleton className="h-3" />
          <Skeleton className="h-3" />
          <Skeleton className="h-3" />
          <Skeleton className="h-3" />
        </div>
      </div>
    </div>
  );
}

function TableLoading({ className = "", rows }: { className?: string; rows: number }) {
  return (
    <div className={`rounded-xl border border-slate-200/80 bg-white shadow-[0_1px_2px_rgba(15,23,42,0.04)] ${className}`}>
      <div className="flex items-center justify-between border-b border-slate-100 px-6 pb-4 pt-5">
        <div className="flex items-center gap-2.5">
          <Skeleton className="h-7 w-7 rounded-md" />
          <div>
            <Skeleton className="h-4 w-40" />
            <Skeleton className="mt-2 h-3 w-52" />
          </div>
        </div>
        <Skeleton className="h-4 w-16" />
      </div>
      <div className="space-y-3 p-5">
        {Array.from({ length: rows }).map((_, idx) => (
          <div key={idx} className="grid grid-cols-12 items-center gap-3">
            <Skeleton className="col-span-1 h-6 w-6 rounded-md" />
            <Skeleton className="col-span-5 h-4" />
            <Skeleton className="col-span-4 h-2 rounded-full" />
            <Skeleton className="col-span-1 h-4" />
            <Skeleton className="col-span-1 h-4" />
          </div>
        ))}
      </div>
    </div>
  );
}

function Skeleton({ className = "" }: { className?: string }) {
  return (
    <span
      className={`block animate-pulse rounded bg-slate-200/80 ${className}`}
    />
  );
}
