import { describe, expect, it } from "vitest";
import type { FireFeature, GeoJsonMetadata } from "../types";
import {
  ALL_PRIORITY_PROVINCES,
  buildPriorityCsvRows,
  buildPrioritySnapshot,
  filterPriorityRows,
  getPriorityFreshness,
  isPrioritySnapshotCurrent,
} from "./priorityAreas";

function feature({
  date = "2026-08-05",
  source = "predicted",
  probability = 0.1,
  rank,
  province = "Chiang Mai",
  lat = 18.8,
  lon = 98.9,
}: {
  date?: string;
  source?: "predicted" | "observed";
  probability?: number;
  rank?: number;
  province?: string;
  lat?: number;
  lon?: number;
} = {}): FireFeature {
  return {
    type: "Feature",
    geometry: { type: "Point", coordinates: [lon, lat] },
    properties: {
      source,
      base_date: date,
      probability,
      urgency_level: "LOW",
      province,
      fire_days_per_year: 12.5,
      nearest_urban_area: "Chiang Mai",
      rank,
      ranking_tier: "RELATIVE_TOP_10",
      percentile: 0.1,
    },
  };
}

describe("priority snapshot selection", () => {
  it("uses matching metadata base date and excludes observations and older rows", () => {
    const metadata: GeoJsonMetadata = { base_date: "2026-08-05" };
    const snapshot = buildPrioritySnapshot([
      feature({ date: "2026-08-04", rank: 1 }),
      feature({ date: "2026-08-05", rank: 2, probability: 0.9 }),
      feature({ date: "2026-08-05", rank: 1, probability: 0.2, lon: 99 }),
      feature({ date: "2026-08-05", source: "observed", rank: 3 }),
    ], metadata);

    expect(snapshot.baseDate).toBe("2026-08-05");
    expect(snapshot.snapshotSource).toBe("metadata_base_date");
    expect(snapshot.rankingMode).toBe("persisted");
    expect(snapshot.rows.map((row) => row.rank)).toEqual([1, 2]);
    expect(snapshot.loadedCandidateCount).toBe(2);
  });

  it("uses the latest valid feature snapshot and withholds invalid metadata scope", () => {
    const metadata: GeoJsonMetadata = {
      base_date: "2026-02-30",
      latest_data_time: "2026-08-06T12:00:00Z",
      selection_policy: {
        retained_map_candidate_cells: 100,
        total_scored_cells: 1_000,
      },
    };
    const snapshot = buildPrioritySnapshot([
      feature({ date: "2026-08-04", rank: 1 }),
      feature({ date: "2026-08-05", rank: 1 }),
    ], metadata);

    expect(snapshot.baseDate).toBe("2026-08-05");
    expect(snapshot.snapshotSource).toBe("latest_feature_date");
    expect(snapshot.metadataMatchesSnapshot).toBe(false);
    expect(snapshot.retainedMapCandidateCount).toBeNull();
    expect(snapshot.totalScoredCells).toBeNull();
    expect(snapshot.latestDataTime).toBeNull();
  });

  it("keeps an authoritative metadata snapshot current when it emits zero rows", () => {
    const snapshot = buildPrioritySnapshot([
      feature({ date: "2026-08-05", rank: 1 }),
    ], {
      base_date: "2026-08-06",
      latest_data_time: "2026-08-06T12:00:00Z",
      selection_policy: {
        retained_map_candidate_cells: 0,
        total_scored_cells: 1_000,
      },
    });

    expect(snapshot.baseDate).toBe("2026-08-06");
    expect(snapshot.snapshotSource).toBe("metadata_base_date");
    expect(snapshot.metadataMatchesSnapshot).toBe(true);
    expect(snapshot.rows).toEqual([]);
    expect(snapshot.loadedCandidateCount).toBe(0);
    expect(snapshot.retainedMapCandidateCount).toBe(0);
    expect(snapshot.latestDataTime).toBe("2026-08-06T12:00:00Z");
  });

  it("does not attach unversioned metadata timestamps to a feature snapshot", () => {
    const snapshot = buildPrioritySnapshot([feature({ rank: 1 })], {
      latest_data_time: "2026-08-05T12:00:00Z",
      generated_at: "2026-08-05T12:30:00Z",
    });

    expect(snapshot.metadataMatchesSnapshot).toBe(false);
    expect(snapshot.latestDataTime).toBeNull();
    expect(snapshot.generatedAt).toBeNull();
  });

  it("accepts duplicate persisted competition ranks", () => {
    const snapshot = buildPrioritySnapshot([
      feature({ probability: 0.8, rank: 1, lat: 15, lon: 101 }),
      feature({ probability: 0.8, rank: 1, lat: 17, lon: 100 }),
      feature({ probability: 0.2, rank: 3, lat: 14, lon: 102 }),
    ]);

    expect(snapshot.rankingMode).toBe("persisted");
    expect(snapshot.rows.map((row) => row.rank)).toEqual([1, 1, 3]);
  });

  it("falls back for incomplete ranks without breaking probability ties by coordinates", () => {
    const snapshot = buildPrioritySnapshot([
      feature({ probability: 0.2, rank: 2, lat: 15, lon: 101 }),
      feature({ probability: 0.8, rank: 1, lat: 17, lon: 100 }),
      feature({ probability: 0.2, lat: 14, lon: 102 }),
    ]);

    expect(snapshot.rankingMode).toBe("probability_fallback");
    expect(snapshot.rows.map((row) => [row.rank, row.probability, row.latitude])).toEqual([
      [1, 0.8, 17],
      [2, 0.2, 14],
      [2, 0.2, 15],
    ]);
    expect(snapshot.rows.every((row) => row.priorityTier === null)).toBe(true);
  });

  it("uses selection counts only for the metadata-controlled snapshot", () => {
    const snapshot = buildPrioritySnapshot([
      feature({ rank: 1 }),
      feature({ rank: 2, lon: 99 }),
    ], {
      base_date: "2026-08-05",
      selection_policy: {
        retained_map_candidate_cells: 66,
        total_scored_cells: 11_835,
      },
    });

    expect(snapshot.retainedMapCandidateCount).toBe(66);
    expect(snapshot.totalScoredCells).toBe(11_835);
    expect(snapshot.loadedCandidateCount).toBe(2);
  });
});

describe("priority filters and export", () => {
  it("filters by province without renumbering the global snapshot rank", () => {
    const snapshot = buildPrioritySnapshot([
      feature({ rank: 1, province: "Bangkok" }),
      feature({ rank: 2, province: "Chiang Mai", lon: 99 }),
      feature({ rank: 3, province: "Chiang Mai", lon: 100 }),
    ]);

    expect(filterPriorityRows(snapshot.rows, "Chiang Mai", 10).map((row) => row.rank))
      .toEqual([2, 3]);
    expect(filterPriorityRows(snapshot.rows, ALL_PRIORITY_PROVINCES, 1)).toHaveLength(1);
  });

  it("does not split a competition-rank tie at the row limit", () => {
    const rows = buildPrioritySnapshot([
      ...Array.from({ length: 9 }, (_, index) => feature({
        rank: index + 1,
        probability: 1 - index / 100,
        lat: 10 + index / 10,
        lon: 99,
      })),
      feature({ rank: 10, probability: 0.5, lat: 15, lon: 100 }),
      feature({ rank: 10, probability: 0.5, lat: 16, lon: 100 }),
    ]).rows;

    expect(filterPriorityRows(rows, ALL_PRIORITY_PROVINCES, 10)).toHaveLength(11);
    expect(filterPriorityRows(rows, ALL_PRIORITY_PROVINCES, 10).slice(-2)
      .map((row) => row.rank)).toEqual([10, 10]);
  });

  it("exports raw probability, display probability, ranking method, and retained scope", () => {
    const snapshot = buildPrioritySnapshot([
      feature({ rank: 1, probability: 0.000013, province: "=FORMULA" }),
    ]);
    const csvRows = buildPriorityCsvRows(
      snapshot.rows,
      snapshot.rankingMode,
      snapshot.baseDate,
    );

    expect(csvRows[0]).toContain("risk_probability");
    expect(csvRows[1]).toContain(0.000013);
    expect(csvRows[1]).toContain("<0.01%");
    expect(csvRows[1]).toContain("persisted");
    expect(csvRows[1]).toContain("'=FORMULA");
    expect(csvRows[1]).toContain("retained_map_candidates");
  });
});

describe("priority freshness", () => {
  it("prefers latest source-input time over snapshot generation time", () => {
    expect(getPriorityFreshness({
      latestDataTime: "2026-08-05T10:00:00Z",
      generatedAt: "2026-08-05T11:00:00Z",
    }, Date.parse("2026-08-05T13:00:00Z"))).toEqual({
      timestamp: "2026-08-05T10:00:00Z",
      timestampKind: "source_data",
      ageHours: 3,
      isAheadOfBrowserClock: false,
    });
  });

  it("compares the base date against the calendar date in Thailand", () => {
    const justBeforeMidnightUtc = Date.parse("2026-08-05T18:30:00Z");
    expect(isPrioritySnapshotCurrent("2026-08-06", justBeforeMidnightUtc)).toBe(true);
    expect(isPrioritySnapshotCurrent("2026-08-05", justBeforeMidnightUtc)).toBe(false);
  });
});
