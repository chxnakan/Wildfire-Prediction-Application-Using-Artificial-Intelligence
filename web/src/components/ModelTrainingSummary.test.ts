import { describe, expect, it } from "vitest";
import type { ValidationMetrics } from "../types";
import { computeStability } from "./ModelTrainingSummary";

describe("ModelTrainingSummary stability provenance", () => {
  it("does not fabricate missing aggregate statistics as zeros", () => {
    const metrics = {
      stability_auc_mean: 0.812,
      stability_valid_months: 7,
      stability_months: 12,
    } as ValidationMetrics;

    expect(computeStability([], metrics)).toEqual({
      n: 7,
      aucMean: 0.812,
      aucStd: null,
      aucMin: null,
      aucMax: null,
      posRateMean: null,
      posRateStd: null,
      posRateMin: null,
      posRateMax: null,
    });
  });

  it("returns no summary when no persisted mean or monthly rows exist", () => {
    expect(computeStability([], {} as ValidationMetrics)).toBeNull();
  });
});
