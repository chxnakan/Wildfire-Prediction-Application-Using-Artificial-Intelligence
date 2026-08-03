import { describe, expect, it } from "vitest";
import { formatGistdaAcquisitionTime } from "./fireAlerts";

describe("GISTDA acquisition time", () => {
  it("formats the current backend Unix-millisecond shape", () => {
    const value = formatGistdaAcquisitionTime(
      Date.parse("2026-07-24T07:01:00Z"),
    );
    expect(value).toContain("14:01");
  });

  it("prefers the explicit Thailand timestamp when present", () => {
    const value = formatGistdaAcquisitionTime(
      undefined,
      undefined,
      "2026-07-24T14:01:00+07:00",
    );
    expect(value).toContain("14:01");
  });

  it("retains support for legacy YYYYMMDD payloads", () => {
    expect(formatGistdaAcquisitionTime(20260724, "14:01"))
      .toBe("2026-07-24T14:01");
  });

  it("rejects impossible legacy calendar dates", () => {
    expect(formatGistdaAcquisitionTime(20260231, "14:01")).toBe("");
  });
});
