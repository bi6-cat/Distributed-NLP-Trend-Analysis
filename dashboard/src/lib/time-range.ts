export const TIME_RANGE_PRESETS = [
  { value: "latest7d", label: "Latest data 7d" },
  { value: "latest30d", label: "Latest data 30d" },
  { value: "last24h", label: "Last 24h" },
  { value: "last7d", label: "Last 7d" },
  { value: "last30d", label: "Last 30d" },
  { value: "custom", label: "Custom" },
] as const;

export type TimeRangeMode = (typeof TIME_RANGE_PRESETS)[number]["value"];

export type TimeRangeSearchParams = {
  range?: string | string[];
  from?: string | string[];
  to?: string | string[];
};

export type ResolvedTimeRange = {
  mode: TimeRangeMode;
  start: string;
  end: string;
  label: string;
  fromDate?: string;
  toDate?: string;
};

export const DEFAULT_TIME_RANGE_MODE = "latest7d" satisfies TimeRangeMode;

export function firstParam(value: string | string[] | undefined) {
  return Array.isArray(value) ? value[0] : value;
}

export function isTimeRangeMode(value: string | undefined): value is TimeRangeMode {
  return TIME_RANGE_PRESETS.some((preset) => preset.value === value);
}

export function normalizeTimeRangeMode(value: string | undefined): TimeRangeMode {
  return isTimeRangeMode(value) ? value : DEFAULT_TIME_RANGE_MODE;
}

export function isDateOnly(value: string | undefined): value is string {
  return Boolean(value && /^\d{4}-\d{2}-\d{2}$/.test(value));
}

export function getTimeRangePresetLabel(mode: TimeRangeMode) {
  return TIME_RANGE_PRESETS.find((preset) => preset.value === mode)?.label ?? TIME_RANGE_PRESETS[0].label;
}
