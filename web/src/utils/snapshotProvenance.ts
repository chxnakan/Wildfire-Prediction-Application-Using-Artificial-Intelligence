import { isIsoCalendarDate } from "./dates";

export type SnapshotProvenance = "current" | "archived" | "unknown";

export interface SnapshotDateIndex {
  allBaseDates: string[];
  latestBaseDate: string;
}

/**
 * Build the snapshot picker without assuming that a current snapshot has rows.
 * A valid metadata date is authoritative: a no-alert run may intentionally
 * emit zero predictions while archived rows remain in the append-only file.
 */
export function resolveSnapshotDateIndex(
  rowDates: Array<string | null | undefined>,
  metadataBaseDate: string | null | undefined,
): SnapshotDateIndex {
  const authoritativeDate = isIsoCalendarDate(metadataBaseDate ?? "")
    ? metadataBaseDate!
    : null;
  const allBaseDates = Array.from(new Set([
    ...rowDates.filter((date): date is string => isIsoCalendarDate(date ?? "")),
    ...(authoritativeDate ? [authoritativeDate] : []),
  ])).sort();
  return {
    allBaseDates,
    latestBaseDate:
      authoritativeDate ?? allBaseDates[allBaseDates.length - 1] ?? "N/A",
  };
}

/**
 * Numeric thresholds and model metrics in GeoJSON metadata describe the
 * artifact's current base_date. Historical feature rows retain their own
 * probability and urgency_level, but do not carry a versioned threshold
 * bundle. Fail closed when either date is missing so current cutoffs are never
 * silently presented as the rules that governed an older snapshot.
 */
export function snapshotProvenance(
  activeBaseDate: string | null | undefined,
  artifactBaseDate: string | null | undefined,
): SnapshotProvenance {
  const active = activeBaseDate?.trim();
  const artifact = artifactBaseDate?.trim();
  if (!active || active === "N/A" || !artifact || artifact === "N/A") {
    return "unknown";
  }
  return active === artifact ? "current" : "archived";
}

export function currentMetadataAppliesToSnapshot(
  activeBaseDate: string | null | undefined,
  artifactBaseDate: string | null | undefined,
): boolean {
  return snapshotProvenance(activeBaseDate, artifactBaseDate) === "current";
}
