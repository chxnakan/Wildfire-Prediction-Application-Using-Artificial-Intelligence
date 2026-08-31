import type {
  FireFeature,
  GeoJsonMetadata,
  PredictionProperties,
  UrgencyLevel,
} from "../types";
import {
  formatProbabilityPercent,
  readProbability,
} from "./probability";

export const ALL_PRIORITY_PROVINCES = "__all__";

type PersistedRankingProperties = PredictionProperties & {
  ranking_tier?: unknown;
};

export type PriorityRankingMode = "persisted" | "probability_fallback";
export type PrioritySnapshotSource = "metadata_base_date" | "latest_feature_date";

export interface PriorityAreaRow {
  feature: FireFeature;
  /** Global rank within the loaded active snapshot, not rank within a province. */
  rank: number;
  probability: number | null;
  probabilityLabel: string;
  priorityTier: string | null;
  absoluteTier: UrgencyLevel;
  percentile: number | null;
  province: string;
  historicalFireDaysPerYear: number | null;
  historicalFireCount30d: number | null;
  nearestCity: string | null;
  nearestCityDistanceKm: number | null;
  latitude: number;
  longitude: number;
}

export interface PrioritySnapshot {
  baseDate: string | null;
  snapshotSource: PrioritySnapshotSource;
  metadataMatchesSnapshot: boolean;
  rankingMode: PriorityRankingMode;
  rows: PriorityAreaRow[];
  provinces: string[];
  loadedCandidateCount: number;
  retainedMapCandidateCount: number | null;
  totalScoredCells: number | null;
  latestDataTime: string | null;
  generatedAt: string | null;
}

export interface PriorityFreshness {
  timestamp: string | null;
  timestampKind: "source_data" | "snapshot_generated" | "unavailable";
  ageHours: number | null;
  isAheadOfBrowserClock: boolean;
}

const ISO_DAY = /^\d{4}-\d{2}-\d{2}$/;
const RANKING_TIERS = new Set([
  "RELATIVE_TOP_10",
  "RELATIVE_TOP_20",
  "UNRANKED",
  // Legacy artifacts used alert-like names for relative bands. They remain
  // readable, but the component labels them explicitly as legacy-relative.
  "HIGH",
  "WATCH",
  "EXTENDED_WATCH",
  "LOW",
]);

function finiteNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function nonnegativeNumber(value: unknown): number | null {
  const number = finiteNumber(value);
  return number !== null && number >= 0 ? number : null;
}

function positiveInteger(value: unknown): number | null {
  return typeof value === "number" && Number.isInteger(value) && value > 0
    ? value
    : null;
}

function nonnegativeInteger(value: unknown): number | null {
  return typeof value === "number" && Number.isInteger(value) && value >= 0
    ? value
    : null;
}

function validPercentile(value: unknown): number | null {
  const number = finiteNumber(value);
  return number !== null && number >= 0 && number <= 1 ? number : null;
}

function persistedRankingTier(value: unknown): string | null {
  return typeof value === "string" && RANKING_TIERS.has(value)
    ? value
    : null;
}

function provinceName(value: unknown): string {
  return typeof value === "string" && value.trim() ? value.trim() : "—";
}

function nullableText(value: unknown): string | null {
  return typeof value === "string" && value.trim() ? value.trim() : null;
}

function validSnapshotDate(value: unknown): value is string {
  if (typeof value !== "string" || !ISO_DAY.test(value)) return false;
  const parsed = new Date(`${value}T00:00:00Z`);
  return !Number.isNaN(parsed.getTime()) && parsed.toISOString().startsWith(value);
}

function coordinateOrder(a: FireFeature, b: FireFeature): number {
  const [aLon, aLat] = a.geometry.coordinates;
  const [bLon, bLat] = b.geometry.coordinates;
  return aLat - bLat || aLon - bLon;
}

function hasValidCoordinates(feature: FireFeature): boolean {
  const coordinates = feature.geometry?.coordinates;
  if (!Array.isArray(coordinates) || coordinates.length < 2) return false;
  const [longitude, latitude] = coordinates;
  return Number.isFinite(longitude) && Number.isFinite(latitude) &&
    longitude >= -180 && longitude <= 180 && latitude >= -90 && latitude <= 90;
}

function displayAbsoluteTier(value: PredictionProperties["urgency_level"]): UrgencyLevel {
  return value === "CRITICAL" || value === "HIGH" || value === "MEDIUM" ||
      value === "LOW" || value === "NONE"
    ? value
    : "NONE";
}

function persistedRanksAreComplete(features: FireFeature[]): boolean {
  if (features.length === 0) return false;
  const ranks = features.map((feature) =>
    positiveInteger((feature.properties as PersistedRankingProperties).rank)
  );
  // Competition ranking deliberately repeats a rank for tied scores. Requiring
  // uniqueness rejected valid backend ranks and recreated a fake coordinate
  // order in the browser.
  return ranks.every((rank): rank is number => rank !== null);
}

function toPriorityRow(
  feature: FireFeature,
  rank: number,
  rankingMode: PriorityRankingMode,
): PriorityAreaRow {
  const props = feature.properties as PersistedRankingProperties;
  const probability = readProbability(props);
  const [longitude, latitude] = feature.geometry.coordinates;
  return {
    feature,
    rank,
    probability,
    probabilityLabel: formatProbabilityPercent(probability),
    priorityTier: rankingMode === "persisted"
      ? persistedRankingTier(props.ranking_tier)
      : null,
    absoluteTier: displayAbsoluteTier(props.urgency_level),
    percentile: rankingMode === "persisted"
      ? validPercentile(props.percentile)
      : null,
    province: provinceName(props.province),
    historicalFireDaysPerYear: nonnegativeNumber(props.fire_days_per_year),
    historicalFireCount30d: nonnegativeInteger(props.historical_fire_count_30d),
    nearestCity: nullableText(props.nearest_urban_area),
    nearestCityDistanceKm: nonnegativeNumber(props.nearest_urban_distance_km),
    latitude,
    longitude,
  };
}

/**
 * Select and rank one active prediction snapshot.
 *
 * `metadata.base_date` controls when it has matching predicted rows. If it is
 * absent or does not match the loaded features, the latest valid feature date
 * is used and the mismatch is surfaced to the caller. Observations and older
 * snapshots never enter the ranking.
 *
 * A persisted rank is used when every active-snapshot row has a positive
 * integer competition rank; duplicates are valid ties. Otherwise the whole
 * snapshot falls back to probability competition ranks. Coordinates stabilise
 * row display within a tie but never create a different rank or priority band.
 */
export function buildPrioritySnapshot(
  features: FireFeature[],
  metadata?: GeoJsonMetadata | null,
): PrioritySnapshot {
  const predictedByDate = new Map<string, FireFeature[]>();
  for (const feature of features) {
    if (feature.properties.source !== "predicted") continue;
    if (!hasValidCoordinates(feature)) continue;
    const baseDate = feature.properties.base_date;
    if (!validSnapshotDate(baseDate)) continue;
    const current = predictedByDate.get(baseDate) ?? [];
    current.push(feature);
    predictedByDate.set(baseDate, current);
  }

  const metadataDate = validSnapshotDate(metadata?.base_date)
    ? metadata!.base_date!
    : null;
  const sortedFeatureDates = Array.from(predictedByDate.keys()).sort();
  const latestFeatureDate = sortedFeatureDates.length
    ? sortedFeatureDates[sortedFeatureDates.length - 1]
    : null;
  // A valid metadata date is authoritative even with zero emitted rows. The
  // backend may intentionally omit all Background context on a no-alert day;
  // falling back to the latest row date would show an older map as current.
  const baseDate = metadataDate ?? latestFeatureDate;
  const snapshotSource: PrioritySnapshotSource = metadataDate !== null
    ? "metadata_base_date"
    : "latest_feature_date";
  // Fail closed: timestamps and selection counts belong to this snapshot only
  // when metadata carries the same explicit, valid base date. Missing metadata
  // provenance is not treated as an implicit match.
  const metadataMatchesSnapshot = metadataDate !== null && metadataDate === baseDate;
  const snapshotFeatures = baseDate ? predictedByDate.get(baseDate) ?? [] : [];
  const rankingMode: PriorityRankingMode = persistedRanksAreComplete(snapshotFeatures)
    ? "persisted"
    : "probability_fallback";

  const ordered = [...snapshotFeatures];
  if (rankingMode === "persisted") {
    ordered.sort((a, b) => {
      const aRank = positiveInteger((a.properties as PersistedRankingProperties).rank)!;
      const bRank = positiveInteger((b.properties as PersistedRankingProperties).rank)!;
      return aRank - bRank || coordinateOrder(a, b);
    });
  } else {
    ordered.sort((a, b) => {
      const aProbability = readProbability(a.properties);
      const bProbability = readProbability(b.properties);
      if (aProbability === null && bProbability !== null) return 1;
      if (aProbability !== null && bProbability === null) return -1;
      if (aProbability !== null && bProbability !== null && aProbability !== bProbability) {
        return bProbability - aProbability;
      }
      return coordinateOrder(a, b);
    });
  }

  let previousFallbackProbability: number | null | undefined;
  let previousFallbackRank = 0;
  const rows = ordered.map((feature, index) => {
    let rank: number;
    if (rankingMode === "persisted") {
      rank = positiveInteger(
        (feature.properties as PersistedRankingProperties).rank,
      )!;
    } else {
      const probability = readProbability(feature.properties);
      const isTie = index > 0 && probability === previousFallbackProbability;
      rank = isTie ? previousFallbackRank : index + 1;
      previousFallbackProbability = probability;
      previousFallbackRank = rank;
    }
    return toPriorityRow(feature, rank, rankingMode);
  });

  const selectionApplies = baseDate !== null && metadataDate === baseDate;
  const retainedMapCandidateCount = selectionApplies
    ? nonnegativeInteger(metadata?.selection_policy?.retained_map_candidate_cells)
    : null;
  const totalScoredCells = selectionApplies
    ? nonnegativeInteger(metadata?.selection_policy?.total_scored_cells)
    : null;

  return {
    baseDate,
    snapshotSource,
    metadataMatchesSnapshot,
    rankingMode,
    rows,
    provinces: Array.from(new Set(rows.map((row) => row.province))).sort((a, b) =>
      a.localeCompare(b, "en")
    ),
    loadedCandidateCount: rows.length,
    retainedMapCandidateCount,
    totalScoredCells,
    latestDataTime: metadataMatchesSnapshot && metadata?.latest_data_time
      ? metadata.latest_data_time
      : null,
    generatedAt: metadataMatchesSnapshot && metadata?.generated_at
      ? metadata.generated_at
      : null,
  };
}

export function filterPriorityRows(
  rows: PriorityAreaRow[],
  province: string = ALL_PRIORITY_PROVINCES,
  limit: number = 10,
): PriorityAreaRow[] {
  const safeLimit = Number.isInteger(limit) && limit > 0 ? limit : 10;
  const filtered = province === ALL_PRIORITY_PROVINCES
    ? rows
    : rows.filter((row) => row.province === province);
  const limited = filtered.slice(0, safeLimit);
  const boundaryRank = limited[limited.length - 1]?.rank;
  if (boundaryRank === undefined) return limited;
  // A row-count limit must not split an equal-score competition-rank group and
  // imply that coordinates made one tied cell more important than another.
  return filtered.filter((row) => row.rank <= boundaryRank);
}

export function getPriorityFreshness(
  metadata: Pick<PrioritySnapshot, "latestDataTime" | "generatedAt">,
  nowMs: number = Date.now(),
): PriorityFreshness {
  const timestamp = metadata.latestDataTime ?? metadata.generatedAt;
  const timestampKind = metadata.latestDataTime
    ? "source_data"
    : metadata.generatedAt
      ? "snapshot_generated"
      : "unavailable";
  if (!timestamp) {
    return { timestamp: null, timestampKind, ageHours: null, isAheadOfBrowserClock: false };
  }
  const parsed = Date.parse(timestamp);
  if (!Number.isFinite(parsed) || !Number.isFinite(nowMs)) {
    return { timestamp, timestampKind, ageHours: null, isAheadOfBrowserClock: false };
  }
  const ageHours = (nowMs - parsed) / 3_600_000;
  return {
    timestamp,
    timestampKind,
    ageHours,
    isAheadOfBrowserClock: ageHours < -1,
  };
}

export function isPrioritySnapshotCurrent(
  baseDate: string | null,
  nowMs: number = Date.now(),
): boolean {
  if (!validSnapshotDate(baseDate) || !Number.isFinite(nowMs)) return false;
  const parts = new Intl.DateTimeFormat("en", {
    timeZone: "Asia/Bangkok",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  }).formatToParts(new Date(nowMs));
  const part = (type: "year" | "month" | "day") =>
    parts.find((item) => item.type === type)?.value ?? "";
  return baseDate === `${part("year")}-${part("month")}-${part("day")}`;
}

function spreadsheetSafeText(value: string | null): string {
  if (!value) return "";
  return /^[\t\r ]*[=+\-@]/.test(value) ? `'${value}` : value;
}

export function buildPriorityCsvRows(
  rows: PriorityAreaRow[],
  rankingMode: PriorityRankingMode,
  baseDate: string | null,
): (string | number | null)[][] {
  return [
    [
      "global_snapshot_rank",
      "ranking_method",
      "priority_tier",
      "absolute_alert_tier",
      "risk_probability",
      "risk_probability_display",
      "base_date",
      "province",
      "lat",
      "lon",
      "fire_days_per_year",
      "historical_fire_count_30d",
      "nearest_city",
      "nearest_city_distance_km",
      "source_scope",
    ],
    ...rows.map((row) => [
      row.rank,
      rankingMode,
      row.priorityTier ?? "",
      row.absoluteTier,
      row.probability,
      row.probabilityLabel,
      baseDate,
      spreadsheetSafeText(row.province),
      Number(row.latitude.toFixed(4)),
      Number(row.longitude.toFixed(4)),
      row.historicalFireDaysPerYear,
      row.historicalFireCount30d,
      spreadsheetSafeText(row.nearestCity),
      row.nearestCityDistanceKm,
      "retained_map_candidates",
    ]),
  ];
}
