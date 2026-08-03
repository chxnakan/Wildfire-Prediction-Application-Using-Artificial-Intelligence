import { describe, expect, it } from "vitest";
import {
  formatProbabilityPercent,
  isVeryLowProbability,
  probabilityPercent,
} from "./probability";

describe("probability percentage display", () => {
  it("keeps two decimal places for low non-zero probabilities", () => {
    expect(probabilityPercent(0.0004)).toBe(0.04);
    expect(formatProbabilityPercent(0.0004)).toBe("0.04%");
  });

  it("retains trailing zeroes in the display string", () => {
    expect(probabilityPercent(0.001)).toBe(0.1);
    expect(formatProbabilityPercent(0.001)).toBe("0.10%");
    expect(formatProbabilityPercent(0.4312666)).toBe("43.13%");
  });

  it("does not present a positive sub-resolution value as exact zero", () => {
    expect(probabilityPercent(0.0000013)).toBe(0);
    expect(formatProbabilityPercent(0.0000013)).toBe("<0.01%");
    expect(formatProbabilityPercent(0)).toBe("0.00%");
  });

  it("uses an exact, testable cutoff for the optional map filter", () => {
    expect(isVeryLowProbability(0.000099)).toBe(true);
    expect(isVeryLowProbability(0.0001)).toBe(false);
    expect(isVeryLowProbability(null)).toBe(false);
  });

  it("renders missing values without treating them as zero", () => {
    expect(probabilityPercent(null)).toBeNull();
    expect(formatProbabilityPercent(null)).toBe("—");
  });

  it("does not format out-of-domain values as plausible probabilities", () => {
    expect(probabilityPercent(-0.1)).toBeNull();
    expect(probabilityPercent(1.1)).toBeNull();
    expect(formatProbabilityPercent(1.1)).toBe("—");
  });
});
