import { describe, expect, it } from "vitest";
import type { ScientificStats } from "../types";
import { scientificSampleSplits } from "./scientificStats";

function split(n: number) {
  return {
    n,
    positives: Math.floor(n / 10),
    positive_rate: 0.1,
    date_range: ["2026-01-01", "2026-01-31"] as [
      string | null,
      string | null,
    ],
  };
}

describe("scientificSampleSplits", () => {
  it("keeps all four splits from the current persisted report schema", () => {
    const samples = {
      total_densified: 100,
      train: split(40),
      validation: split(20),
      calibration: split(20),
      test: split(20),
    } satisfies ScientificStats["samples"];

    expect(scientificSampleSplits(samples).map((row) => row.key)).toEqual([
      "train",
      "validation",
      "calibration",
      "test",
    ]);
  });

  it("retains compatibility with legacy reports that used val", () => {
    const samples = {
      total_densified: 100,
      train: split(60),
      val: split(20),
      test: split(20),
    } satisfies ScientificStats["samples"];

    expect(scientificSampleSplits(samples).map((row) => row.key)).toEqual([
      "train",
      "validation",
      "test",
    ]);
  });
});
