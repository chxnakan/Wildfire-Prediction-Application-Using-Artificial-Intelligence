import { describe, expect, it } from "vitest";
import { URGENCY_COLORS } from "../constants";
import type { UrgencyThresholds } from "../types";
import {
  heatmapColor,
  heatmapProbability,
  heatmapRiskIntensity,
  probabilityCutoffs,
} from "./heatmapScale";

const CURRENT_THRESHOLDS: UrgencyThresholds = {
  CRITICAL: 1.01,
  HIGH: 0.5236,
  MEDIUM: 0.4376,
  LOW: 0,
};

describe("heatmap probability scale", () => {
  it("uses the persisted probability instead of pseudo-days", () => {
    expect(heatmapProbability({
      probability: 0.0000013,
      raw_prediction: 6.9999,
    })).toBeCloseTo(0.0000013);
  });

  it("keeps probability thresholds on the probability scale", () => {
    expect(probabilityCutoffs(CURRENT_THRESHOLDS)).toEqual({
      critical: 1.01,
      high: 0.5236,
      medium: 0.4376,
    });
  });

  it("preserves an unused CRITICAL sentinel instead of inventing that tier", () => {
    expect(heatmapColor(1, CURRENT_THRESHOLDS)).toBe(URGENCY_COLORS.HIGH);
  });

  it("keeps sub-WATCH cells in the LOW color tier", () => {
    expect(heatmapColor(0.4312, CURRENT_THRESHOLDS)).toBe(
      URGENCY_COLORS.LOW,
    );
    expect(heatmapColor(0.44, CURRENT_THRESHOLDS)).toBe(
      URGENCY_COLORS.MEDIUM,
    );
  });

  it("keeps background cells quiet and reserves strong halos for alerts", () => {
    expect(heatmapRiskIntensity(0.0000013, CURRENT_THRESHOLDS)).toBeLessThan(
      0.000001,
    );
    expect(heatmapRiskIntensity(0.4312, CURRENT_THRESHOLDS)).toBeLessThan(
      0.36,
    );
    expect(heatmapRiskIntensity(0.44, CURRENT_THRESHOLDS)).toBeGreaterThan(
      0.45,
    );
  });
});
