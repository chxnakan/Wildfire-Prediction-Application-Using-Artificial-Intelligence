import { describe, it, expect } from "vitest";
import {
  MIN_COMPARE_RATE_SAMPLE,
  computeCompareStats,
  contextWindowForBaseDate,
  dateFallsWithin,
  hasSufficientRateSample,
  isWindowMature,
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

  it("honours a persisted non-default model horizon", () => {
    expect(riskWindowForBaseDate("2026-07-16", 5)).toEqual({
      start: "2026-07-17",
      end: "2026-07-21",
    });
    expect(contextWindowForBaseDate("2026-07-16", 1, 5)).toEqual({
      start: "2026-07-16",
      end: "2026-07-22",
    });
  });

  it("does not treat an unrelated later FIRMS date as an evaluated outcome", () => {
    const window = riskWindowForBaseDate("2026-07-16");
    expect(dateFallsWithin("2026-07-24", window)).toBe(false);
  });

  it("rejects impossible calendar dates instead of normalising them", () => {
    expect(riskWindowForBaseDate("2026-02-31")).toBeNull();
    expect(dateFallsWithin("2026-02-31", {
      start: "2026-02-01",
      end: "2026-03-05",
    })).toBe(false);
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
  it("does not turn an unknown validation label into a false alarm", () => {
    expect(computeCompareStats(["unknown"])).toMatchObject({
      hits: 0,
      falseAlarms: 0,
      future: 0,
      precision: 0,
    });
  });
  it("computes precision from persisted cell outcomes", () => {
    const s = computeCompareStats(["hit", "hit", "alarm", "alarm"]);
    expect(s.precision).toBeCloseTo(0.5);
    expect(s.hits).toBe(2);
    expect(s.falseAlarms).toBe(2);
  });

  it("structurally withholds recall, misses, and F1", () => {
    const s = computeCompareStats(["hit", "hit", "hit", "alarm"]);
    expect(s.precision).toBeCloseTo(0.75);
    expect(s.misses).toBeNull();
    expect(s.recall).toBeNull();
    expect(s.f1).toBeNull();
  });

  it("never divides by zero", () => {
    const s = computeCompareStats([]);
    expect(s.precision).toBe(0);
  });

  it("counts future separately from hits/alarms", () => {
    const s = computeCompareStats(["hit", "future", "future"]);
    expect(s.future).toBe(2);
    expect(s.hits).toBe(1);
  });
});
