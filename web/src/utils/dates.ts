const DAY_MS = 86_400_000;

function isoCalendarDay(value: string): number | null {
  const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(value);
  if (!match) return null;
  const year = Number(match[1]);
  const month = Number(match[2]);
  const day = Number(match[3]);
  const ms = Date.UTC(year, month - 1, day);
  const parsed = new Date(ms);
  if (
    parsed.getUTCFullYear() !== year ||
    parsed.getUTCMonth() !== month - 1 ||
    parsed.getUTCDate() !== day
  ) {
    return null;
  }
  return Math.floor(ms / DAY_MS);
}

/** True only for a real Gregorian calendar date in YYYY-MM-DD form. */
export function isIsoCalendarDate(value: string): boolean {
  return isoCalendarDay(value) !== null;
}

function localTodayCalendarDay(now = new Date()): number {
  return Math.floor(
    Date.UTC(now.getFullYear(), now.getMonth(), now.getDate()) / DAY_MS,
  );
}

/** YYYY-MM-DD in the viewer's local calendar, without UTC date rollover. */
export function localIsoDate(now = new Date()): string {
  const year = now.getFullYear();
  const month = String(now.getMonth() + 1).padStart(2, "0");
  const day = String(now.getDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
}

export function dateAdd(isoDate: string, days: number): string {
  const calendarDay = isoCalendarDay(isoDate);
  if (calendarDay === null || !Number.isFinite(days)) return "—";
  return new Date((calendarDay + days) * DAY_MS).toISOString().slice(0, 10);
}

/** Calendar-day lag, independent of browser timezone and daylight saving. */
export function calendarDayLag(
  isoDate: string,
  now = new Date(),
): number | null {
  const calendarDay = isoCalendarDay(isoDate);
  if (calendarDay === null) return null;
  return localTodayCalendarDay(now) - calendarDay;
}

export interface FreshnessInfo {
  cls: "fresh" | "stale" | "expired";
  label: string;
  msg: string;
}

/**
 * Format a UTC ISO timestamp (e.g. risk_map's latest_data_time) as a short
 * local date-time in the viewer's zone. FIRMS acq times are UTC; the browser
 * converts to local (Asia/Bangkok for a Thai user) so "06:48Z" shows "13:48".
 * Returns null on an unparseable/missing value.
 */
export function formatDataTime(iso: string | null | undefined, lang: "en" | "th"): string | null {
  if (!iso) return null;
  const d = new Date(iso);
  if (isNaN(d.getTime())) return null;
  return d.toLocaleString(lang === "th" ? "th-TH" : "en-GB", {
    day: "numeric", month: "short", hour: "2-digit", minute: "2-digit",
  });
}

export function computeFreshness(baseDateIso: string | null): FreshnessInfo | null {
  if (!baseDateIso || baseDateIso === "N/A") {
    return { cls: "expired", label: "no data", msg: "" };
  }
  const lagDays = calendarDayLag(baseDateIso);
  if (lagDays === null) {
    return { cls: "expired", label: "invalid date", msg: "Prediction date is invalid." };
  }

  if (lagDays < 0) {
    return {
      cls: "expired",
      label: "future date",
      msg: "Prediction date is ahead of the current calendar date.",
    };
  }
  if (lagDays === 0)
    return {
      cls: "fresh",
      label: "live",
      msg: "Data is current — the model reports one risk window, not an exact fire date.",
    };
  if (lagDays === 1)
    return {
      cls: "fresh",
      label: "1 day old",
      msg: "Yesterday's snapshot — its outcome window is already in progress.",
    };
  if (lagDays <= 3)
    return {
      cls: "stale",
      label: `${lagDays} d behind`,
      msg: `The snapshot is ${lagDays} days old. Refresh prediction data before operational use.`,
    };
  return {
    cls: "expired",
    label: `${lagDays} d behind`,
    msg: `Data is ${lagDays} days stale — do not treat this snapshot as current risk.`,
  };
}
