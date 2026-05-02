"use client";

const keywords = [
  { word: "chip M4", weight: 95 },
  { word: "hiệu năng", weight: 80 },
  { word: "Apple Intelligence", weight: 88 },
  { word: "iPad Pro", weight: 75 },
  { word: "giá bán", weight: 65 },
  { word: "tản nhiệt", weight: 55 },
  { word: "OLED", weight: 70 },
  { word: "mỏng nhẹ", weight: 60 },
  { word: "AI", weight: 85 },
  { word: "nâng cấp", weight: 50 },
  { word: "pin", weight: 45 },
  { word: "MacBook", weight: 40 },
];

export function WordCloud() {
  return (
    <div className="relative flex flex-wrap items-center justify-center gap-x-4 gap-y-3 min-h-[280px] w-full p-6 rounded-lg bg-gradient-to-br from-slate-50/80 via-white to-violet-50/30 border border-slate-100">
      {keywords.map((kw, i) => {
        const size = Math.max(13, (kw.weight / 100) * 36);
        const opacity = Math.max(0.55, kw.weight / 100);
        let colorClass = "text-slate-500";
        if (kw.weight >= 85) colorClass = "text-indigo-600";
        else if (kw.weight >= 70) colorClass = "text-slate-800";
        else if (kw.weight >= 55) colorClass = "text-slate-700";

        return (
          <span
            key={i}
            className={`${colorClass} font-semibold tracking-tight hover:text-indigo-600 transition-all cursor-default leading-none`}
            style={{
              fontSize: `${size}px`,
              opacity,
            }}
          >
            {kw.word}
          </span>
        );
      })}
    </div>
  );
}
