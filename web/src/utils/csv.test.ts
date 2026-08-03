import { describe, expect, it } from "vitest";
import type { FireFeature } from "../types";
import { buildCellsCsv } from "./csv";

const FEATURE: FireFeature = {
  type: "Feature",
  geometry: { type: "Point", coordinates: [100.12345, 15.98765] },
  properties: {
    source: "predicted",
    base_date: "2026-07-24",
    predicted_fire_date: "2099-01-01",
    days_until_fire: 6,
    probability: 0.000013,
    urgency_level: "LOW",
    historical_fire_count_30d: 3,
    fire_days_per_year: 36.3,
    province: "Nakhon Ratchasima",
  },
};

describe("buildCellsCsv", () => {
  it("exports calibrated risk and its 3-day window without pseudo-date fields", () => {
    const csv = buildCellsCsv([FEATURE]);
    const [header, row] = csv.split("\r\n");

    expect(header).toContain("risk_probability_pct");
    expect(header).toContain("risk_window_start");
    expect(header).toContain("risk_window_end");
    expect(header).not.toContain("predicted_fire_date");
    expect(header).not.toContain("days_until_fire");
    expect(row).toContain("0.00,BACKGROUND,2026-07-24,2026-07-25,2026-07-27");
    expect(csv).not.toContain("2099-01-01");
  });

  it("preserves two decimal places for non-zero displayed percentages", () => {
    const feature = {
      ...FEATURE,
      properties: { ...FEATURE.properties, probability: 0.001 },
    };

    expect(buildCellsCsv([feature])).toContain(",0.10,BACKGROUND,");
  });
});
