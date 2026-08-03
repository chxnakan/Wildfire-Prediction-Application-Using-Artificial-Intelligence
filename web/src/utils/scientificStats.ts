import type { SampleSplit, ScientificStats } from "../types";

export interface NamedSampleSplit {
  key: "train" | "validation" | "calibration" | "test";
  label: string;
  data: SampleSplit;
}

/**
 * Normalise the persisted scientific-report split schema.
 *
 * Current reports publish four chronological splits named `validation` and
 * `calibration`. Early dashboard builds called the validation split `val` and
 * omitted calibration. Supporting both keeps old artifacts readable while
 * ensuring a newly published four-way report cannot crash the Reports route.
 */
export function scientificSampleSplits(
  samples: ScientificStats["samples"],
): NamedSampleSplit[] {
  const validation = samples.validation ?? samples.val;
  return [
    { key: "train", label: "Training", data: samples.train },
    ...(validation
      ? [{ key: "validation" as const, label: "Validation", data: validation }]
      : []),
    ...(samples.calibration
      ? [{
          key: "calibration" as const,
          label: "Calibration",
          data: samples.calibration,
        }]
      : []),
    { key: "test", label: "Test", data: samples.test },
  ];
}
