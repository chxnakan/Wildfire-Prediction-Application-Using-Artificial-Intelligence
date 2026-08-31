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
  ] as const)("recognizes the %s tab route", (route) => {
    vi.stubGlobal("window", { location: { hash: `#${route}` } });
    expect(getRoute()).toBe(route);
  });

  it.each(["#operations", "#analytics", "#/analytics"])(
    "falls back to the dashboard for removed route %s",
    (hash) => {
      vi.stubGlobal("window", { location: { hash } });
      expect(getRoute()).toBe("dashboard");
    },
  );

  it("falls back to the dashboard for an unknown route", () => {
    vi.stubGlobal("window", { location: { hash: "#unknown" } });
    expect(getRoute()).toBe("dashboard");
  });
});
