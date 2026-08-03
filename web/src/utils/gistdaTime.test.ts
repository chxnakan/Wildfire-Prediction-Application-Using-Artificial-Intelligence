import { describe, expect, it } from "vitest";
import { ageDays, parseGistdaTime } from "./gistdaTime";

describe("parseGistdaTime", () => {
  it("parses ArcGIS Unix milliseconds", () => {
    const parsed = parseGistdaTime(1_721_820_000_000, null);
    expect(parsed?.ms).toBe(1_721_820_000_000);
    expect(parsed?.precision).toBe("minute");
  });

  it("parses YYYYMMDD and HHMM in Thailand time", () => {
    const parsed = parseGistdaTime(20260724, "1430");
    expect(parsed?.iso).toBe("2026-07-24T14:30:00+07:00");
    expect(parsed?.precision).toBe("minute");
  });

  it("marks a date without a usable time as date precision", () => {
    const parsed = parseGistdaTime("20260724", null);
    expect(parsed?.precision).toBe("date");
  });

  it("rejects missing and malformed dates", () => {
    expect(parseGistdaTime(null, null)).toBeNull();
    expect(parseGistdaTime("not-a-date", "1200")).toBeNull();
    expect(parseGistdaTime("20261340", "1200")).toBeNull();
    expect(parseGistdaTime("20260231", "1200")).toBeNull();
  });
});

describe("ageDays", () => {
  it("does not classify a missing or future timestamp as fresh", () => {
    const now = Date.parse("2026-07-24T12:00:00Z");
    expect(ageDays(null, now)).toBeNull();
    expect(ageDays(now + 1000, now)).toBeNull();
  });
});
