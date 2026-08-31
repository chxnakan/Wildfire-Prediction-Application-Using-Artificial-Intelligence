import { describe, it, expect } from "vitest";
import {
  DEFAULT_TIER_FILTER,
  TIER_TO_URGENCY,
  urgencyToTier,
  isActionableUrgency,
  matchesTierFilter,
  readProbability,
  resolveTierMode,
} from "./risk";

describe("risk-tier semantics", () => {
  it("maps tiers onto persisted urgency levels and back (round-trip)", () => {
    for (const tier of ["High", "Watch", "Background"] as const) {
      expect(urgencyToTier(TIER_TO_URGENCY[tier])).toBe(tier);
    }
  });

  it("treats non-predicted urgency values as no tier", () => {
    expect(urgencyToTier("OBSERVED")).toBeNull();
    expect(urgencyToTier("NONE")).toBeNull();
    expect(urgencyToTier(undefined)).toBeNull();
  });

  it("filters by tier, and 'all' matches every recognised risk tier", () => {
    expect(matchesTierFilter("CRITICAL", "High")).toBe(true);
    expect(matchesTierFilter("HIGH", "High")).toBe(true);
    expect(matchesTierFilter("MEDIUM", "High")).toBe(false);
    expect(matchesTierFilter("MEDIUM", "Watch")).toBe(true);
    expect(matchesTierFilter("LOW", "Background")).toBe(true);
    expect(matchesTierFilter("LOW", "all")).toBe(true);
    expect(matchesTierFilter(undefined, "all")).toBe(false);
  });

  it("shows the complete retained tier set by default", () => {
    expect(DEFAULT_TIER_FILTER).toBe("all");
    expect(matchesTierFilter("CRITICAL", DEFAULT_TIER_FILTER)).toBe(true);
    expect(matchesTierFilter("HIGH", DEFAULT_TIER_FILTER)).toBe(true);
    expect(matchesTierFilter("MEDIUM", DEFAULT_TIER_FILTER)).toBe(true);
    expect(matchesTierFilter("LOW", DEFAULT_TIER_FILTER)).toBe(true);

    const plateau = Array.from({ length: 100 }, () => "LOW" as const);
    expect(plateau.filter((tier) => matchesTierFilter(tier, DEFAULT_TIER_FILTER)))
      .toHaveLength(100);
    expect(plateau.filter((tier) => matchesTierFilter(tier, "Background")))
      .toHaveLength(100);
  });

  it("counts every alert tier as actionable", () => {
    expect(isActionableUrgency("HIGH")).toBe(true);
    expect(isActionableUrgency("MEDIUM")).toBe(true);
    expect(isActionableUrgency("CRITICAL")).toBe(true);
    expect(isActionableUrgency("LOW")).toBe(false);
  });
});

describe("probability semantics", () => {
  it("prefers a valid calibrated probability without masking invalid units", () => {
    expect(readProbability({ probability: 0.42 })).toBeCloseTo(0.42);
    expect(readProbability({ probability: 1.5 })).toBeNull();
    expect(readProbability({ probability: -0.2 })).toBeNull();
  });

  it("uses the canonical piecewise inverse for legacy pseudo-days", () => {
    expect(readProbability({ raw_prediction: 0.5 })).toBeCloseTo(0.7);
    expect(readProbability({ raw_prediction: 1 })).toBeCloseTo(0.6);
    expect(readProbability({ raw_prediction: 7 })).toBeCloseTo(0);
  });

  it("falls back to legacy data when a persisted probability is malformed", () => {
    expect(readProbability({ probability: 42, raw_prediction: 1.5 }))
      .toBeCloseTo(0.5);
  });

  it("returns null when neither field is present (unscored ≠ zero)", () => {
    expect(readProbability({})).toBeNull();
  });
});

describe("threshold semantics", () => {
  const probabilityThresholds = {
    CRITICAL: 1.01,
    HIGH: 0.5236,
    MEDIUM: 0.4376,
    LOW: 0,
  };

  it("treats ordered 0..1 cutoffs as absolute when source metadata is absent", () => {
    expect(resolveTierMode(probabilityThresholds, undefined)).toBe("absolute");
  });

  it("honours explicit priority-ranking metadata", () => {
    expect(resolveTierMode(probabilityThresholds, "PRIORITY RANKING (legacy)"))
      .toBe("relative");
  });

  it("keeps legacy pseudo-day thresholds in days mode", () => {
    expect(resolveTierMode(
      { CRITICAL: 0, HIGH: 2, MEDIUM: 4, LOW: 7 },
      undefined,
    )).toBe("days");
  });
});
