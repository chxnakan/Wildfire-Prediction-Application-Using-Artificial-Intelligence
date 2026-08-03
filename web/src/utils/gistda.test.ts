import { afterEach, describe, expect, it, vi } from "vitest";
import { fetchLiveFires, normaliseRefreshSeconds } from "./gistda";

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("live-fire refresh configuration", () => {
  it("uses a safe default for invalid or overly aggressive values", () => {
    expect(normaliseRefreshSeconds(undefined)).toBe(60);
    expect(normaliseRefreshSeconds("abc")).toBe(60);
    expect(normaliseRefreshSeconds(0)).toBe(60);
    expect(normaliseRefreshSeconds(5)).toBe(60);
    expect(normaliseRefreshSeconds("30")).toBe(30);
  });
});

describe("fetchLiveFires", () => {
  it("rejects an invalid payload so the app can retain its last good cache", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ features: null }),
    } as Response));

    await expect(fetchLiveFires()).rejects.toThrow(
      "Invalid live-hotspot response",
    );
  });

  it("rejects malformed feature records before the live page can crash", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ features: [{}] }),
    } as Response));

    await expect(fetchLiveFires()).rejects.toThrow(
      "Invalid live-hotspot response",
    );
  });
});
