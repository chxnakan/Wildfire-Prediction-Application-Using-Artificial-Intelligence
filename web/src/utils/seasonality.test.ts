import { describe, expect, it } from "vitest";
import { thaiFireSeasonForDate } from "./seasonality";

describe("Thai fire-season context", () => {
  it.each(["2026-11-01", "2026-12-15", "2027-01-20", "2027-04-30"])(
    "classifies %s as in-season",
    (date) => {
      expect(thaiFireSeasonForDate(date)).toBe("in-season");
    },
  );

  it.each(["2026-05-01", "2026-07-24", "2026-10-31"])(
    "classifies %s as off-season",
    (date) => {
      expect(thaiFireSeasonForDate(date)).toBe("off-season");
    },
  );

  it("declines to infer a season from an invalid or missing date", () => {
    expect(thaiFireSeasonForDate(undefined)).toBeNull();
    expect(thaiFireSeasonForDate("2026-13-01")).toBeNull();
    expect(thaiFireSeasonForDate("July 2026")).toBeNull();
  });
});
