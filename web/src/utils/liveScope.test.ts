import { describe, expect, it } from "vitest";
import {
  countFixedRecencyWindows,
  isVerifiedThailandHotspot,
} from "./liveScope";

describe("Thailand hotspot scope", () => {
  it("trusts the Thailand-scoped GISTDA feed", () => {
    expect(isVerifiedThailandHotspot("GISTDA", "")).toBe(true);
  });

  it("requires a polygon-derived province for global FIRMS rows", () => {
    expect(isVerifiedThailandHotspot("FIRMS", "Chiang Mai")).toBe(true);
    expect(isVerifiedThailandHotspot("FIRMS", "")).toBe(false);
    expect(isVerifiedThailandHotspot("FIRMS", "—")).toBe(false);
  });
});

describe("fixed live KPI windows", () => {
  it("counts 1 h and 24 h independently of any list time-window filter", () => {
    expect(countFixedRecencyWindows([
      { daysOld: 0.02, timePrecision: "minute" },
      { daysOld: 0.5, timePrecision: "minute" },
      { daysOld: 2, timePrecision: "minute" },
      { daysOld: 0, timePrecision: "date" },
      { daysOld: -0.1, timePrecision: "minute" },
    ])).toEqual({ last1h: 1, last24h: 2 });
  });
});
