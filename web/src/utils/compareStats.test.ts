import { describe, it, expect } from "vitest";
import {
  MIN_COMPARE_RATE_SAMPLE,
  computeCompareStats,
  contextWindowForBaseDate,
  dateFallsWithin,
  hasSufficientRateSample,
  isWindowMature,
  observedDatesCoverWindow,
  riskWindowForBaseDate,
  selectDefaultReviewDate,
} from "./compareStats";

describe("comparison date windows", () => {
  it("uses the model target base_date+1 through base_date+3", () => {
    expect(riskWindowForBaseDate("2026-07-16")).toEqual({
      start: "2026-07-17",
      end: "2026-07-19",
    });
  });

  it("widens only the visual context window", () => {
    expect(contextWindowForBaseDate("2026-07-16", 1)).toEqual({
      start: "2026-07-16",
      end: "2026-07-20",
    });
  });

  it("does not treat an unrelated later FIRMS date as an evaluated outcome", () => {
    const window = riskWindowForBaseDate("2026-07-16");
    expect(dateFallsWithin("2026-07-24", window)).toBe(false);
    expect(observedDatesCoverWindow(["2026-07-24"], window)).toBe(false);
  });

  it("recognizes an observed artifact spanning the complete target window", () => {
    const window = riskWindowForBaseDate("2026-07-16");
    expect(observedDatesCoverWindow(
      ["2026-07-17", "2026-07-18", "2026-07-19"],
      window,
    )).toBe(true);
  });
});

describe("hasSufficientRateSample", () => {
  it("withholds unstable rates below the documented denominator", () => {
    expect(hasSufficientRateSample(MIN_COMPARE_RATE_SAMPLE - 1)).toBe(false);
    expect(hasSufficientRateSample(MIN_COMPARE_RATE_SAMPLE)).toBe(true);
  });
});

describe("selectDefaultReviewDate", () => {
  it("prefers a meaningful operational sample over a newer tiny sample", () => {
    expect(selectDefaultReviewDate([
      { date: "2026-07-16", actionableDecided: 5, decided: 181 },
      { date: "2026-05-26", actionableDecided: 84, decided: 200 },
    ])).toBe("2026-05-26");
  });

  it("keeps a small operational sample available as the fallback", () => {
    expect(selectDefaultReviewDate([
      { date: "2026-07-16", actionableDecided: 5, decided: 181 },
      { date: "2026-07-15", actionableDecided: 0, decided: 159 },
    ])).toBe("2026-07-16");
  });
});

describe("isWindowMature", () => {
  it("is false for an empty snapshot", () => {
    expect(isWindowMature([])).toBe(false);
  });

  it("is false while any cell in the snapshot is still future", () => {
    expect(isWindowMature(["hit", "alarm", "future"])).toBe(false);
  });

  it("is true once every cell has a decided (non-future) status", () => {
    expect(isWindowMature(["hit", "alarm", "hit"])).toBe(true);
  });
});

describe("computeCompareStats", () => {
  it("computes precision independent of window maturity", () => {
    // 2 hits, 2 false alarms, window immature (recall must stay 0)
    const s = computeCompareStats(["hit", "hit", "alarm", "alarm"], 5, false);
    expect(s.precision).toBeCloseTo(0.5);
    expect(s.recall).toBe(0); // gated off — this is the bug this fix prevents
  });

  it("computes recall once the window is mature", () => {
    // 3 hits, 1 alarm, 2 missed observed fires → recall = 3 / (3+2) = 0.6
    const s = computeCompareStats(["hit", "hit", "hit", "alarm"], 2, true);
    expect(s.recall).toBeCloseTo(0.6);
    expect(s.precision).toBeCloseTo(0.75);
  });

  it("never divides by zero", () => {
    const s = computeCompareStats([], 0, true);
    expect(s.precision).toBe(0);
    expect(s.recall).toBe(0);
    expect(s.f1).toBe(0);
  });

  it("f1 is the harmonic mean of precision and recall when both are available", () => {
    const s = computeCompareStats(["hit", "hit", "alarm"], 2, true);
    // precision = 2/3, recall = 2/4 = 0.5
    const expectedF1 = (2 * (2 / 3) * 0.5) / ((2 / 3) + 0.5);
    expect(s.f1).toBeCloseTo(expectedF1);
  });

  it("counts future separately from hits/alarms", () => {
    const s = computeCompareStats(["hit", "future", "future"], 0, false);
    expect(s.future).toBe(2);
    expect(s.hits).toBe(1);
  });
});
