// Pure, testable logic extracted from ComparePage.tsx — the "Predicted vs
// Actual" honesty audit. Kept separate from the Leaflet/React rendering code
// so the recall-availability rule can be unit tested without a DOM.

export type MatchStatus = "hit" | "alarm" | "future";

export const MIN_COMPARE_RATE_SAMPLE = 20;

const ISO_DATE_RE = /^\d{4}-\d{2}-\d{2}$/;

function parseIsoDate(date: string): Date | null {
  if (!ISO_DATE_RE.test(date)) return null;
  const parsed = new Date(`${date}T00:00:00Z`);
  return Number.isNaN(parsed.getTime()) ? null : parsed;
}

function addUtcDays(date: Date, days: number): string {
  const shifted = new Date(date.getTime() + days * 86_400_000);
  return shifted.toISOString().slice(0, 10);
}

export interface DateWindow {
  start: string;
  end: string;
}

/**
 * The binary target is "a hotspot on base_date+1 .. base_date+horizon".
 * Keeping this calculation in one place prevents the visual context controls
 * from accidentally changing the evaluation population.
 */
export function riskWindowForBaseDate(
  baseDate: string,
  horizonDays = 3,
): DateWindow | null {
  const base = parseIsoDate(baseDate);
  if (!base || !Number.isInteger(horizonDays) || horizonDays < 1) return null;
  return {
    start: addUtcDays(base, 1),
    end: addUtcDays(base, horizonDays),
  };
}

export function contextWindowForBaseDate(
  baseDate: string,
  toleranceDays: number,
  horizonDays = 3,
): DateWindow | null {
  const base = parseIsoDate(baseDate);
  if (
    !base ||
    !Number.isInteger(toleranceDays) ||
    toleranceDays < 0 ||
    !Number.isInteger(horizonDays) ||
    horizonDays < 1
  ) {
    return null;
  }
  return {
    start: addUtcDays(base, 1 - toleranceDays),
    end: addUtcDays(base, horizonDays + toleranceDays),
  };
}

export function dateFallsWithin(
  date: string | null | undefined,
  window: DateWindow | null,
): boolean {
  if (!window || !date) return false;
  const dateOnly = date.slice(0, 10);
  return ISO_DATE_RE.test(dateOnly) &&
    dateOnly >= window.start &&
    dateOnly <= window.end;
}

/**
 * A list containing only today's FIRMS points cannot evaluate an archived
 * prediction window. Require the observed artifact's date range to span the
 * full target window before enabling recall/F1.
 */
export function observedDatesCoverWindow(
  dates: Array<string | null | undefined>,
  window: DateWindow | null,
): boolean {
  if (!window) return false;
  const valid = dates
    .map((date) => date?.slice(0, 10) ?? "")
    .filter((date) => ISO_DATE_RE.test(date))
    .sort();
  if (valid.length === 0) return false;
  return valid[0] <= window.start && valid[valid.length - 1] >= window.end;
}

export function hasSufficientRateSample(
  denominator: number,
  minimum = MIN_COMPARE_RATE_SAMPLE,
): boolean {
  return Number.isFinite(denominator) && denominator >= minimum;
}

export interface ReviewDateCandidate {
  date: string;
  actionableDecided: number;
  decided: number;
}

/**
 * Choose a useful default without hiding newer snapshots. Callers keep the
 * full list selectable; this only avoids opening the audit on an unstable tiny
 * operational sample when a newer sufficiently-sized sample exists.
 */
export function selectDefaultReviewDate(
  newestFirst: ReviewDateCandidate[],
  minimum = MIN_COMPARE_RATE_SAMPLE,
): string | null {
  const reviewable = newestFirst.find(
    (item) => hasSufficientRateSample(item.actionableDecided, minimum),
  );
  const operational = newestFirst.find((item) => item.actionableDecided > 0);
  const anyDecided = newestFirst.find((item) => item.decided > 0);
  return reviewable?.date ??
    operational?.date ??
    anyDecided?.date ??
    newestFirst[0]?.date ??
    null;
}

export interface CompareStats {
  hits: number;
  falseAlarms: number;
  misses: number;
  future: number;
  precision: number;
  recall: number;
  f1: number;
}

/**
 * Every prediction for one base_date shares the SAME risk window
 * (base_date+1 .. base_date+horizon), so validation_status is either mature
 * for the whole snapshot or none of it. Recall's denominator (hits + misses)
 * is only meaningful once the window has closed — while any cell is still
 * "future", real fires that will still occur can't yet be counted as misses,
 * so recall would understate the model's performance. `false` (immature)
 * also covers the empty-snapshot case.
 */
export function isWindowMature(statuses: MatchStatus[]): boolean {
  return statuses.length > 0 && statuses.every((s) => s !== "future");
}

/**
 * precision = hits / (hits + false alarms) — meaningful as soon as any
 * prediction is decided, independent of window maturity.
 * recall = hits / (hits + misses) — gated on `windowMature` (see above).
 */
export function computeCompareStats(
  statuses: MatchStatus[],
  missedObservedCount: number,
  windowMature: boolean,
): CompareStats {
  let hits = 0, falseAlarms = 0, future = 0;
  for (const s of statuses) {
    if (s === "hit") hits += 1;
    else if (s === "alarm") falseAlarms += 1;
    else future += 1;
  }
  const misses = missedObservedCount;
  const decided = hits + falseAlarms;
  const recallDenom = hits + misses;
  const precision = decided ? hits / decided : 0;
  const recall = windowMature && recallDenom ? hits / recallDenom : 0;
  const f1 = precision + recall ? (2 * precision * recall) / (precision + recall) : 0;
  return { hits, falseAlarms, misses, future, precision, recall, f1 };
}
