import type { FireFeature } from "../types";
import { isIsoCalendarDate } from "./dates";

/**
 * Describe the calendar-date coverage of the FIRMS observations currently
 * loaded on the map. Feature dates are the most direct provenance; the
 * artifact freshness timestamp is used only when no valid feature date is
 * available.
 */
export function observedOverlayDateScope(
  observed: FireFeature[],
  latestDataTime?: string | null,
): string | null {
  const dates = [
    ...new Set(
      observed
        .map((feature) => feature.properties.date?.trim() ?? "")
        .filter(isIsoCalendarDate),
    ),
  ].sort();

  if (dates.length === 1) return dates[0];
  if (dates.length > 1) return `${dates[0]} – ${dates[dates.length - 1]}`;

  const timestampDate = latestDataTime?.slice(0, 10) ?? "";
  return isIsoCalendarDate(timestampDate) ? timestampDate : null;
}
