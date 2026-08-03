export type TimestampPrecision = "minute" | "date";

export interface ParsedSourceTime {
  ms: number;
  iso: string;
  precision: TimestampPrecision;
}

function validTimestamp(ms: number): boolean {
  return Number.isFinite(ms) && !Number.isNaN(new Date(ms).getTime());
}

/**
 * GISTDA ArcGIS responses have appeared in three forms over time:
 * Unix milliseconds, Unix seconds, and YYYYMMDD plus a separate HHMM field.
 * Return null instead of making an invalid/missing timestamp look current.
 */
export function parseGistdaTime(
  date: number | string | null | undefined,
  time: number | string | null | undefined,
): ParsedSourceTime | null {
  if (date == null || date === "") return null;
  const rawDate = String(date).trim();

  if (/^\d{13}$/.test(rawDate) || /^\d{10}$/.test(rawDate)) {
    const numeric = Number(rawDate);
    const ms = rawDate.length === 13 ? numeric : numeric * 1000;
    if (!validTimestamp(ms)) return null;
    return {
      ms,
      iso: new Date(ms).toISOString(),
      precision: "minute",
    };
  }

  if (!/^\d{8}$/.test(rawDate)) return null;
  const year = Number(rawDate.slice(0, 4));
  const month = Number(rawDate.slice(4, 6));
  const day = Number(rawDate.slice(6, 8));
  // Date.parse normalises impossible dates (for example 20260231 becomes
  // March 3). Reject them explicitly so a malformed source date never appears
  // as a plausible, fresh hotspot.
  const calendarCheck = new Date(Date.UTC(year, month - 1, day));
  if (
    calendarCheck.getUTCFullYear() !== year ||
    calendarCheck.getUTCMonth() !== month - 1 ||
    calendarCheck.getUTCDate() !== day
  ) {
    return null;
  }
  const yyyyMmDd =
    `${rawDate.slice(0, 4)}-${rawDate.slice(4, 6)}-${rawDate.slice(6, 8)}`;
  const rawTime = time == null ? "" : String(time).trim().padStart(4, "0");
  const hasMinute = /^\d{4}$/.test(rawTime) &&
    Number(rawTime.slice(0, 2)) <= 23 &&
    Number(rawTime.slice(2, 4)) <= 59;
  const hhmm = hasMinute
    ? `${rawTime.slice(0, 2)}:${rawTime.slice(2, 4)}`
    : "00:00";
  const iso = `${yyyyMmDd}T${hhmm}:00+07:00`;
  const ms = Date.parse(iso);
  if (!validTimestamp(ms)) return null;
  return {
    ms,
    iso,
    precision: hasMinute ? "minute" : "date",
  };
}

export function ageDays(
  timestampMs: number | null,
  nowMs = Date.now(),
): number | null {
  if (timestampMs == null || !validTimestamp(timestampMs)) return null;
  const age = (nowMs - timestampMs) / 86_400_000;
  // A future source timestamp is not a recent detection.
  return age < 0 ? null : age;
}
