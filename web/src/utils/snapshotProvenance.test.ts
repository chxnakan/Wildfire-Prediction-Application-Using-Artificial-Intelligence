import { describe, expect, it } from "vitest";
import {
  currentMetadataAppliesToSnapshot,
  resolveSnapshotDateIndex,
  snapshotProvenance,
} from "./snapshotProvenance";

describe("snapshot provenance", () => {
  it("allows current metadata only for the artifact base date", () => {
    expect(snapshotProvenance("2026-08-05", "2026-08-05")).toBe("current");
    expect(currentMetadataAppliesToSnapshot("2026-08-05", "2026-08-05"))
      .toBe(true);
  });

  it("marks a retained historical snapshot as archived", () => {
    expect(snapshotProvenance("2026-07-24", "2026-08-05"))
      .toBe("archived");
    expect(currentMetadataAppliesToSnapshot("2026-07-24", "2026-08-05"))
      .toBe(false);
  });

  it("fails closed when the artifact date is missing", () => {
    expect(snapshotProvenance("2026-08-05", undefined)).toBe("unknown");
    expect(currentMetadataAppliesToSnapshot("2026-08-05", undefined))
      .toBe(false);
  });
});

describe("snapshot date index", () => {
  it("keeps an authoritative zero-row current snapshot ahead of archived rows", () => {
    expect(resolveSnapshotDateIndex(
      ["2026-08-17", "2026-08-11"],
      "2026-08-25",
    )).toEqual({
      allBaseDates: ["2026-08-11", "2026-08-17", "2026-08-25"],
      latestBaseDate: "2026-08-25",
    });
  });

  it("falls back to the latest valid row date only when metadata is invalid", () => {
    expect(resolveSnapshotDateIndex(
      ["bad", "2026-08-17", "2026-08-11"],
      "2026-02-30",
    )).toEqual({
      allBaseDates: ["2026-08-11", "2026-08-17"],
      latestBaseDate: "2026-08-17",
    });
  });
});
