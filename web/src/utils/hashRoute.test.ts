import { afterEach, describe, expect, it, vi } from "vitest";
import { getRoute } from "./hashRoute";

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("hash routing", () => {
  it.each([
    "dashboard",
    "notify",
    "live",
    "reports",
    "compare",
    "analytics",
  ] as const)("recognizes the %s tab route", (route) => {
    vi.stubGlobal("window", { location: { hash: `#${route}` } });
    expect(getRoute()).toBe(route);
  });

  it("restores Analytics directly from the URL hash", () => {
    vi.stubGlobal("window", { location: { hash: "#analytics" } });
    expect(getRoute()).toBe("analytics");
  });

  it("accepts the slash-prefixed Analytics hash used by shared links", () => {
    vi.stubGlobal("window", { location: { hash: "#/analytics" } });
    expect(getRoute()).toBe("analytics");
  });

  it("falls back to the dashboard for an unknown route", () => {
    vi.stubGlobal("window", { location: { hash: "#unknown" } });
    expect(getRoute()).toBe("dashboard");
  });
});
