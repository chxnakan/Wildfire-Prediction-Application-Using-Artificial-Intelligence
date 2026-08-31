import { afterEach, describe, expect, it, vi } from "vitest";
import {
  fetchCellWeather,
  fetchGeoJson,
  fetchHealth,
  fetchHotspotAnalytics,
  fetchNotifyCapabilities,
  fetchNotifyLog,
  fetchRollingEval,
} from "./api";

function jsonResponse(body: unknown, status = 200): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
  } as Response;
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("fetchCellWeather", () => {
  it("uses backend ERA5 data when the local API is available", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({
      available: true,
      source: "ECMWF ERA5",
      date: "2026-07-15",
      temp_min_c: 25.2,
      temp_max_c: 30.7,
    }));
    vi.stubGlobal("fetch", fetchMock);

    const result = await fetchCellWeather(14.6, 101.1, "2026-07-24");

    expect(result.available).toBe(true);
    expect(result.source).toBe("ECMWF ERA5");
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("falls back to real Open-Meteo data when the backend is offline", async () => {
    const fetchMock = vi.fn()
      .mockRejectedValueOnce(new TypeError("Failed to fetch"))
      .mockResolvedValueOnce(jsonResponse({
        latitude: 14.587,
        longitude: 101.114,
        daily: {
          time: [
            "2026-07-18",
            "2026-07-19",
            "2026-07-20",
            "2026-07-21",
            "2026-07-22",
            "2026-07-23",
            "2026-07-24",
          ],
          temperature_2m_max: [32.5, 32.8, 32.3, 31.8, 31.3, 31.6, 31.7],
          temperature_2m_min: [24.9, 25.1, 25.2, 25.9, 24.9, 25.2, 24.2],
          precipitation_sum: [0.1, 0, 0.3, 1.2, 5.3, 11.4, 10.6],
          wind_speed_10m_max: [16.3, 14.1, 14.8, 15.3, 14.6, 21.1, 19.1],
          et0_fao_evapotranspiration: [5.45, 6.04, 6.07, 4.76, 3.8, 3.4, 4.35],
        },
      }));
    vi.stubGlobal("fetch", fetchMock);

    const result = await fetchCellWeather(14.6, 101.1, "2026-07-24");

    expect(result.available).toBe(true);
    expect(result.source).toContain("Open-Meteo");
    expect(result.temp_min_c).toBe(24.2);
    expect(result.temp_max_c).toBe(31.7);
    expect(result.precip_7d_mm).toBeCloseTo(28.9);
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(String(fetchMock.mock.calls[1][0])).toContain("open-meteo.com");
  });

  it("returns an honest unavailable state when both sources fail", async () => {
    const fetchMock = vi.fn()
      .mockRejectedValueOnce(new TypeError("Failed to fetch"))
      .mockRejectedValueOnce(new TypeError("Network unavailable"));
    vi.stubGlobal("fetch", fetchMock);

    const result = await fetchCellWeather(14.6, 101.1, "2026-07-24");

    expect(result.available).toBe(false);
    expect(result.reason).toContain("Open-Meteo fallback unavailable");
  });

  it("does not infer dry conditions from an incomplete rainfall window", async () => {
    const fetchMock = vi.fn()
      .mockRejectedValueOnce(new TypeError("Failed to fetch"))
      .mockResolvedValueOnce(jsonResponse({
        daily: {
          time: [
            "2026-07-18",
            "2026-07-19",
            "2026-07-20",
            "2026-07-21",
            "2026-07-22",
            "2026-07-23",
            "2026-07-24",
          ],
          temperature_2m_max: [36, 36, 36, 36, 36, 36, 36],
          temperature_2m_min: [25, 25, 25, 25, 25, 25, 25],
          precipitation_sum: [0, 0, null, 0, 0, 0, 0],
          wind_speed_10m_max: [10, 10, 10, 10, 10, 10, 10],
        },
      }));
    vi.stubGlobal("fetch", fetchMock);

    const result = await fetchCellWeather(14.6, 101.1, "2026-07-24");

    expect(result.available).toBe(true);
    expect(result.precip_7d_mm).toBeNull();
    expect(result.fire_weather_index?.level_key).toBe("unknown");
  });

  it("does not silently replace an impossible requested date with today", async () => {
    const fetchMock = vi.fn()
      .mockRejectedValueOnce(new TypeError("Backend unavailable"));
    vi.stubGlobal("fetch", fetchMock);

    const result = await fetchCellWeather(14.6, 101.1, "2026-02-31");

    expect(result.available).toBe(false);
    expect(result.reason).toContain("invalid weather date");
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });
});

describe("fetchHealth", () => {
  it("uses the current backend readiness and geojson-age fields", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse({
      status: "ok",
      ready: false,
      geojson_age_days: 0.1,
    })));
    expect(await fetchHealth()).toBe("stale");
  });

  it("marks an old current-schema snapshot stale", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse({
      status: "ok",
      ready: true,
      geojson_age_days: 6,
    })));
    expect(await fetchHealth()).toBe("stale");
  });

  it("does not call stale source data live just because GeoJSON was rewritten", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse({
      status: "ok",
      ready: true,
      geojson_age_days: 0.01,
      data_stale_days: 12,
    })));
    expect(await fetchHealth()).toBe("stale");
  });

  it("treats a materially future-dated source artifact as degraded", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse({
      status: "ok",
      ready: true,
      geojson_age_days: 0.01,
      data_stale_days: -3,
    })));
    expect(await fetchHealth()).toBe("stale");
  });

  it("reports live only for a ready API with a fresh snapshot", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse({
      status: "ok",
      ready: true,
      geojson_age_days: 0.1,
    })));
    expect(await fetchHealth()).toBe("live");
  });
});

describe("fetchGeoJson", () => {
  it("rejects a successful HTML/catch-all response instead of crashing later", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse({
      type: "not-geojson",
    })));
    await expect(fetchGeoJson()).rejects.toThrow("invalid feature collection");
  });

  it("rejects malformed rows before they can white-screen the map", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse({
      type: "FeatureCollection",
      features: [{ type: "Feature", geometry: null, properties: null }],
    })));

    await expect(fetchGeoJson()).rejects.toThrow("invalid feature record");
  });
});

describe("fetchHotspotAnalytics", () => {
  const validPayload = {
    as_of: "2026-07-24",
    transboundary: [{ country: "Thailand", count: 3 }],
    transboundary_metadata: {
      available: true,
      window_hours: 24,
    },
    historical_burned_mrai: [
      { year: "2567", year_ce: 2024, burned_mrai: 1.25 },
    ],
    historical_burned_mrai_metadata: {
      available: true,
      source: "versioned test fixture",
    },
  };

  it("loads the analytics endpoint through the shared API client", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(validPayload));
    vi.stubGlobal("fetch", fetchMock);

    await expect(fetchHotspotAnalytics()).resolves.toEqual(validPayload);
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/analytics/hotspots",
      expect.objectContaining({ cache: "no-store" }),
    );
  });

  it("rejects a catch-all or malformed payload instead of showing false zeros", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse({
      as_of: "2026-07-24",
      transboundary: "not-an-array",
      historical_burned_mrai: [],
      historical_burned_mrai_metadata: { available: false },
    })));

    await expect(fetchHotspotAnalytics()).rejects.toThrow(
      "invalid response schema",
    );
  });
});

describe("secondary API response validation", () => {
  it("loads the non-secret notification capability contract", async () => {
    const payload = {
      mode: "stub",
      delivery_available: false,
      browser_queue_available: false,
      browser_log_available: false,
      persistence: "process_memory",
    };
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(payload));
    vi.stubGlobal("fetch", fetchMock);

    await expect(fetchNotifyCapabilities()).resolves.toEqual(payload);
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/notify/capabilities",
      expect.objectContaining({ cache: "no-store" }),
    );
  });

  it("fails closed when the notification capability response is malformed", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse({
      mode: "provider",
      delivery_available: true,
      browser_queue_available: true,
    })));

    await expect(fetchNotifyCapabilities()).rejects.toThrow(
      "invalid response schema",
    );
  });

  it("rejects malformed rolling-evaluation rows before a chart can crash", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse({
      summary: null,
      months: "not-an-array",
    })));

    await expect(fetchRollingEval()).rejects.toThrow("invalid response schema");
  });

  it("rejects a malformed notification log instead of treating it as empty", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse({
      records: "not-an-array",
    })));

    await expect(fetchNotifyLog()).rejects.toThrow("invalid response schema");
  });

  it("rejects malformed notification records before rendering the log", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse({
      records: [{ id: "missing-required-fields" }],
    })));

    await expect(fetchNotifyLog()).rejects.toThrow("invalid record");
  });
});
