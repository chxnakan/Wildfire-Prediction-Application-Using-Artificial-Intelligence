import { describe, expect, it } from "vitest";
import type { FireFeature } from "../types";
import { observedOverlayDateScope } from "./observedOverlay";

function observed(date?: string): FireFeature {
  return {
    type: "Feature",
    geometry: { type: "Point", coordinates: [100.5, 13.7] },
    properties: { source: "observed", date },
  };
}

describe("observedOverlayDateScope", () => {
  it("uses the dates carried by the currently loaded FIRMS overlay", () => {
    expect(
      observedOverlayDateScope([
        observed("2026-08-05"),
        observed("2026-08-03"),
        observed("2026-08-05"),
      ]),
    ).toBe("2026-08-03 – 2026-08-05");
  });

  it("falls back to the artifact freshness date when rows have no valid date", () => {
    expect(
      observedOverlayDateScope(
        [observed("not-a-date")],
        "2026-08-05T07:54:00+00:00",
      ),
    ).toBe("2026-08-05");
  });

  it("does not invent a date when neither source records one", () => {
    expect(observedOverlayDateScope([observed()], "invalid")).toBeNull();
  });
});
