// Pure, testable logic extracted from ComparePage.tsx — the "Predicted vs
// Actual" honesty audit. Kept separate from the Leaflet/React rendering code
// so the supported precision-only audit can be unit tested without a DOM.

import { dateAdd, isIsoCalendarDate } from "./dates";

export type MatchStatus = "hit" | "alarm" | "future" | "unknown";

export const MIN_COMPARE_RATE_SAMPLE = 20;

function parseIsoDate(date: string): Date | null {
  if (!isIsoCalendarDate(date)) return null;
  const parsed = new Date(`${date}T00:00:00Z`);
  return Number.isNaN(parsed.getTime()) ? null : parsed;
}

function addUtcDays(date: Date, days: number): string {
  return dateAdd(date.toISOString().slice(0, 10), days);
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
  return isIsoCalendarDate(dateOnly) &&
    dateOnly >= window.start &&
    dateOnly <= window.end;
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
  misses: null;
  future: number;
  precision: number;
  recall: null;
  f1: null;
}

/**
 * Every prediction for one base_date shares the SAME risk window
 * (base_date+1 .. base_date+horizon), so validation_status is either mature
 * for the whole snapshot or none of it. Persisted alert precision should not
 * be presented while any outcome is still marked `future`. `false` also
 * covers the empty-snapshot case.
 */
export function isWindowMature(statuses: MatchStatus[]): boolean {
  return statuses.length > 0 && statuses.every((s) => s !== "future");
}

/**
 * Precision is supported because every archived prediction cell carries its
 * own persisted validation_status. Recall/F1 are deliberately unavailable:
 * the map GeoJSON retains only a display-oriented subset of observations, so
 * it cannot provide a complete fire-outcome denominator. Returning null makes
 * that constraint structural instead of relying on presentation code to hide
 * a misleading number.
 */
export function computeCompareStats(statuses: MatchStatus[]): CompareStats {
  let hits = 0, falseAlarms = 0, future = 0;
  for (const s of statuses) {
    if (s === "hit") hits += 1;
    else if (s === "alarm") falseAlarms += 1;
    else if (s === "future") future += 1;
    // Missing/unknown validation labels are deliberately omitted. Treating
    // them as false alarms or pending outcomes would fabricate evidence.
  }
  const decided = hits + falseAlarms;
  const precision = decided ? hits / decided : 0;
  return {
    hits,
    falseAlarms,
    misses: null,
    future,
    precision,
    recall: null,
    f1: null,
  };
}
