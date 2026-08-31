export type HotspotSource = "GISTDA" | "FIRMS";

export interface RecencyRecord {
  daysOld: number | null;
  timePrecision: "minute" | "date" | null;
}

/**
 * GISTDA's endpoint is Thailand-scoped. FIRMS is global, so an approximate
 * rectangular bbox is not sufficient evidence of Thai membership; require
 * the province annotation produced by the backend's Thailand polygon lookup.
 */
export function isVerifiedThailandHotspot(
  source: HotspotSource,
  province: string | null | undefined,
): boolean {
  if (source === "GISTDA") return true;
  const normalized = province?.trim() ?? "";
  return normalized.length > 0 && normalized !== "—";
}

/** Fixed 1 h / 24 h KPI counts, independent of the list's chosen time filter. */
export function countFixedRecencyWindows(rows: RecencyRecord[]): {
  last1h: number;
  last24h: number;
} {
  let last1h = 0;
  let last24h = 0;
  for (const row of rows) {
    if (
      row.timePrecision !== "minute" ||
      row.daysOld == null ||
      !Number.isFinite(row.daysOld) ||
      row.daysOld < 0
    ) {
      continue;
    }
    if (row.daysOld * 24 <= 1) last1h += 1;
    if (row.daysOld <= 1) last24h += 1;
  }
  return { last1h, last24h };
}
