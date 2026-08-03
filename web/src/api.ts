import type {
  FireGeoJson,
  NotifyLogRecord,
  NotifyRequest,
  NotifyResponse,
  RollingMonthPoint,
  TrainingSummary,
} from "./types";

// In dev the Vite proxy forwards /geojson → http://localhost:8000.
// In prod FastAPI serves the SPA from the same origin, so /geojson is
// already on the right host. VITE_API_BASE lets a deployer override
// (e.g. point the static SPA at a separate API host) without rebuilding.
const API_BASE = (import.meta.env.VITE_API_BASE ?? "").replace(/\/$/, "");

export async function fetchGeoJson(): Promise<FireGeoJson> {
  const res = await fetch(`${API_BASE}/geojson`, {
    cache: "no-store",
    signal: AbortSignal.timeout(15_000),
  });
  if (!res.ok) throw new Error(`GeoJSON fetch failed: HTTP ${res.status}`);
  const payload = await res.json() as FireGeoJson;
  if (
    payload?.type !== "FeatureCollection" ||
    !Array.isArray(payload.features)
  ) {
    throw new Error("GeoJSON fetch failed: invalid feature collection");
  }
  const hasInvalidFeature = payload.features.some((feature) => {
    if (
      typeof feature !== "object" ||
      feature === null ||
      typeof feature.properties !== "object" ||
      feature.properties === null ||
      feature.geometry?.type !== "Point" ||
      !Array.isArray(feature.geometry.coordinates) ||
      feature.geometry.coordinates.length < 2
    ) {
      return true;
    }
    const [lon, lat] = feature.geometry.coordinates;
    return (
      !Number.isFinite(Number(lat)) ||
      !Number.isFinite(Number(lon)) ||
      (
        feature.properties.source !== "predicted" &&
        feature.properties.source !== "observed"
      )
    );
  });
  if (hasInvalidFeature) {
    throw new Error("GeoJSON fetch failed: invalid feature record");
  }
  return payload;
}

export interface RollingEvalResponse {
  summary: { auc_mean?: number; auc_std?: number; auc_min?: number; auc_max?: number; valid_months?: number } | null;
  months: RollingMonthPoint[];
}

export async function fetchRollingEval(): Promise<RollingEvalResponse> {
  const res = await fetch(`${API_BASE}/api/rolling-eval`, {
    cache: "no-store",
    signal: AbortSignal.timeout(10_000),
  });
  if (!res.ok) throw new Error(`Rolling-eval fetch failed: HTTP ${res.status}`);
  const payload: unknown = await res.json();
  if (
    typeof payload !== "object" ||
    payload === null ||
    !Array.isArray((payload as Partial<RollingEvalResponse>).months)
  ) {
    throw new Error("Rolling-eval fetch failed: invalid response schema");
  }
  const result = payload as RollingEvalResponse;
  if (result.months.some(
    (row) =>
      typeof row?.month !== "string" ||
      typeof row.auc !== "number" ||
      !Number.isFinite(row.auc) ||
      typeof row.positive_rate !== "number" ||
      !Number.isFinite(row.positive_rate) ||
      typeof row.n !== "number" ||
      !Number.isFinite(row.n),
  )) {
    throw new Error("Rolling-eval fetch failed: invalid month record");
  }
  return result;
}

export async function fetchTrainingSummary(): Promise<TrainingSummary> {
  const res = await fetch(`${API_BASE}/api/training-summary`, {
    cache: "no-store",
    signal: AbortSignal.timeout(10_000),
  });
  if (!res.ok) throw new Error(`Training summary fetch failed: HTTP ${res.status}`);
  const payload: unknown = await res.json();
  if (typeof payload !== "object" || payload === null || Array.isArray(payload)) {
    throw new Error("Training summary fetch failed: invalid response schema");
  }
  return payload as TrainingSummary;
}

export interface HotspotAnalyticsData {
  as_of: string;
  generated_at_utc?: string;
  transboundary: { country: string; count: number }[];
  transboundary_metadata?: {
    available: boolean;
    reason?: string | null;
    window_hours?: number;
    window_start_utc?: string;
    window_end_utc?: string;
    source_rows_in_window?: number;
    assigned_rows?: number;
    unassigned_rows?: number;
    counts_are_mutually_exclusive?: boolean;
    method?: string;
  };
  historical_burned_mrai: {
    year: string;
    year_ce: number;
    burned_mrai: number;
  }[];
}

function isHotspotAnalyticsData(value: unknown): value is HotspotAnalyticsData {
  if (typeof value !== "object" || value === null) return false;
  const data = value as Partial<HotspotAnalyticsData>;
  if (
    typeof data.as_of !== "string" ||
    !Array.isArray(data.transboundary) ||
    !Array.isArray(data.historical_burned_mrai)
  ) {
    return false;
  }
  const validTransboundary = data.transboundary.every(
    (row) =>
      typeof row === "object" &&
      row !== null &&
      typeof row.country === "string" &&
      typeof row.count === "number" &&
      Number.isFinite(row.count),
  );
  const validHistory = data.historical_burned_mrai.every(
    (row) =>
      typeof row === "object" &&
      row !== null &&
      typeof row.year === "string" &&
      typeof row.year_ce === "number" &&
      Number.isFinite(row.year_ce) &&
      typeof row.burned_mrai === "number" &&
      Number.isFinite(row.burned_mrai),
  );
  return validTransboundary && validHistory;
}

/**
 * Fetch the analytics sidecar through the same configurable API origin as
 * every other request. The caller's signal handles route unmounts; the local
 * controller additionally prevents a stalled sidecar from spinning forever.
 */
export async function fetchHotspotAnalytics(
  signal?: AbortSignal,
  timeoutMs = 15_000,
): Promise<HotspotAnalyticsData> {
  const controller = new AbortController();
  const abortFromCaller = () => controller.abort(signal?.reason);
  if (signal?.aborted) abortFromCaller();
  else signal?.addEventListener("abort", abortFromCaller, { once: true });
  const timer = globalThis.setTimeout(
    () => controller.abort(new DOMException("Analytics request timed out", "TimeoutError")),
    timeoutMs,
  );
  try {
    const res = await fetch(`${API_BASE}/api/analytics/hotspots`, {
      cache: "no-store",
      signal: controller.signal,
    });
    if (!res.ok) {
      throw new Error(`Analytics request failed: HTTP ${res.status}`);
    }
    const payload: unknown = await res.json();
    if (!isHotspotAnalyticsData(payload)) {
      throw new Error("Analytics request failed: invalid response schema");
    }
    return payload;
  } finally {
    globalThis.clearTimeout(timer);
    signal?.removeEventListener("abort", abortFromCaller);
  }
}

// ── Live status (used by the header pulse badge) ──
export type LiveStatus = "live" | "stale" | "offline";

export async function fetchHealth(timeoutMs = 4000): Promise<LiveStatus> {
  const ctrl = new AbortController();
  const timer = globalThis.setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const res = await fetch(`${API_BASE}/health`, { signal: ctrl.signal });
    if (!res.ok) return "offline";
    const body = await res.json().catch(() => ({}));
    // Current FastAPI returns `geojson_age_days` + `ready`; retain
    // `data_stale_days` for compatibility with older deployments. The
    // previous frontend checked only the obsolete field and could show LIVE
    // while the current backend explicitly reported ready=false.
    const ageDays =
      typeof body?.geojson_age_days === "number"
        ? body.geojson_age_days
        : body?.data_stale_days;
    if (
      body?.ready === false ||
      body?.status !== "ok" ||
      (typeof ageDays === "number" && ageDays > 5)
    ) {
      return "stale";
    }
    return "live";
  } catch {
    return "offline";
  } finally {
    globalThis.clearTimeout(timer);
  }
}

// ── Notify dispatch + log ──
export async function postNotify(req: NotifyRequest): Promise<NotifyResponse> {
  const res = await fetch(`${API_BASE}/api/notify`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
    signal: AbortSignal.timeout(15_000),
  });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`Notify POST failed: HTTP ${res.status} ${body}`);
  }
  return res.json();
}

export async function fetchNotifyLog(limit = 100): Promise<NotifyLogRecord[]> {
  const safeLimit = Math.max(1, Math.min(100, Math.trunc(limit) || 100));
  const res = await fetch(`${API_BASE}/api/notify/log?limit=${safeLimit}`, {
    cache: "no-store",
    signal: AbortSignal.timeout(10_000),
  });
  if (!res.ok) throw new Error(`Notify log fetch failed: HTTP ${res.status}`);
  const body: unknown = await res.json();
  if (
    typeof body !== "object" ||
    body === null ||
    !Array.isArray((body as { records?: unknown }).records)
  ) {
    throw new Error("Notify log fetch failed: invalid response schema");
  }
  const records = (body as { records: unknown[] }).records;
  if (records.some(
    (record) =>
      typeof record !== "object" ||
      record === null ||
      typeof (record as Partial<NotifyLogRecord>).id !== "string" ||
      typeof (record as Partial<NotifyLogRecord>).timestamp !== "string" ||
      typeof (record as Partial<NotifyLogRecord>).status !== "string" ||
      typeof (record as Partial<NotifyLogRecord>).recipients_count !== "number" ||
      typeof (record as Partial<NotifyLogRecord>).zone_ids_count !== "number",
  )) {
    throw new Error("Notify log fetch failed: invalid record");
  }
  return records as NotifyLogRecord[];
}

// ── Per-cell weather (ERA5) + Fire Weather Index ──
export interface CellWeather {
  available: boolean;
  reason?: string;
  source?: string;
  lat_grid?: number;
  lon_grid?: number;
  date?: string;
  temp_min_c?: number | null;
  temp_max_c?: number | null;
  precip_sum_mm?: number | null;
  wind_max_kmh?: number | null;
  et0_mm?: number | null;
  precip_7d_mm?: number | null;
  fire_weather_index?: {
    level: string;
    level_key?: "very_high" | "high" | "moderate" | "low" | "unknown";
    color: string;
    emoji: string;
    method?: string;
    is_official_fwi?: boolean;
  };
}

interface OpenMeteoDailyPayload {
  latitude?: number;
  longitude?: number;
  daily?: {
    time?: string[];
    temperature_2m_max?: Array<number | null>;
    temperature_2m_min?: Array<number | null>;
    precipitation_sum?: Array<number | null>;
    wind_speed_10m_max?: Array<number | null>;
    et0_fao_evapotranspiration?: Array<number | null>;
  };
}

const DAY_MS = 86_400_000;

function isoDate(date: Date): string {
  return date.toISOString().slice(0, 10);
}

function validIsoDate(value: string | undefined): value is string {
  const match = value
    ? /^(\d{4})-(\d{2})-(\d{2})$/.exec(value)
    : null;
  if (!match) return false;
  const year = Number(match[1]);
  const month = Number(match[2]);
  const day = Number(match[3]);
  const parsed = new Date(Date.UTC(year, month - 1, day));
  return (
    parsed.getUTCFullYear() === year &&
    parsed.getUTCMonth() === month - 1 &&
    parsed.getUTCDate() === day
  );
}

function finiteOrNull(value: number | null | undefined): number | null {
  return typeof value === "number" && isFinite(value) ? value : null;
}

function fallbackFireIndex(
  tempMax: number | null,
  precip7d: number | null,
): NonNullable<CellWeather["fire_weather_index"]> {
  // The rule needs both inputs. Treating unknown rainfall as 0 mm previously
  // manufactured a "high" dryness signal from incomplete weather data.
  if (tempMax === null || precip7d === null) {
    return {
      level: "Unknown",
      level_key: "unknown",
      color: "#6c707a",
      emoji: "❔",
    };
  }
  if (tempMax > 35 && precip7d < 5) {
    return {
      level: "Very high",
      level_key: "very_high",
      color: "#ef4444",
      emoji: "🔴",
    };
  }
  if (tempMax > 32 && precip7d < 15) {
    return {
      level: "High",
      level_key: "high",
      color: "#f97316",
      emoji: "🟠",
    };
  }
  if (tempMax > 28) {
    return {
      level: "Moderate",
      level_key: "moderate",
      color: "#eab308",
      emoji: "🟡",
    };
  }
  return {
    level: "Low",
    level_key: "low",
    color: "#22c55e",
    emoji: "🟢",
  };
}

async function fetchOpenMeteoFallback(
  lat: number,
  lon: number,
  requestedDate?: string,
): Promise<CellWeather> {
  if (requestedDate !== undefined && !validIsoDate(requestedDate)) {
    return { available: false, reason: "invalid weather date" };
  }
  const target = validIsoDate(requestedDate)
    ? new Date(`${requestedDate}T00:00:00Z`)
    : new Date();
  if (!Number.isFinite(target.getTime())) {
    return { available: false, reason: "invalid weather date" };
  }

  const targetDate = isoDate(target);
  const startDate = isoDate(new Date(target.getTime() - 6 * DAY_MS));
  const today = new Date(`${isoDate(new Date())}T00:00:00Z`);
  const ageDays = Math.round((today.getTime() - target.getTime()) / DAY_MS);
  const isForecastWindow = ageDays >= -16 && ageDays <= 92;
  const endpoint = isForecastWindow
    ? "https://api.open-meteo.com/v1/forecast"
    : "https://archive-api.open-meteo.com/v1/archive";
  const params = new URLSearchParams({
    latitude: String(lat),
    longitude: String(lon),
    start_date: startDate,
    end_date: targetDate,
    daily: [
      "temperature_2m_max",
      "temperature_2m_min",
      "precipitation_sum",
      "wind_speed_10m_max",
      "et0_fao_evapotranspiration",
    ].join(","),
    timezone: "Asia/Bangkok",
  });

  const res = await fetch(`${endpoint}?${params}`, {
    signal: AbortSignal.timeout(10_000),
  });
  if (!res.ok) {
    return {
      available: false,
      reason: `Open-Meteo HTTP ${res.status}`,
    };
  }

  const payload = await res.json() as OpenMeteoDailyPayload;
  const daily = payload.daily;
  const times = daily?.time ?? [];
  const targetIndex = times.lastIndexOf(targetDate);
  if (!daily || targetIndex < 0) {
    return {
      available: false,
      reason: "Open-Meteo returned no data for this date",
    };
  }

  const tempMin = finiteOrNull(daily.temperature_2m_min?.[targetIndex]);
  const tempMax = finiteOrNull(daily.temperature_2m_max?.[targetIndex]);
  const precip = finiteOrNull(daily.precipitation_sum?.[targetIndex]);
  const wind = finiteOrNull(daily.wind_speed_10m_max?.[targetIndex]);
  const et0 = finiteOrNull(daily.et0_fao_evapotranspiration?.[targetIndex]);
  const precipitationWindow = (daily.precipitation_sum ?? [])
    .slice(Math.max(0, targetIndex - 6), targetIndex + 1);
  const hasCompleteRainWindow =
    precipitationWindow.length === 7 &&
    precipitationWindow.every(
      (value) => typeof value === "number" && isFinite(value),
    );
  const precip7d = hasCompleteRainWindow
    ? (precipitationWindow as number[])
        .reduce((sum, value) => sum + value, 0)
    : null;

  return {
    available: tempMin !== null || tempMax !== null || precip !== null,
    reason:
      tempMin === null && tempMax === null && precip === null
        ? "Open-Meteo returned empty weather values"
        : undefined,
    source: isForecastWindow
      ? "Open-Meteo live forecast fallback"
      : "ECMWF ERA5 via Open-Meteo fallback",
    lat_grid: finiteOrNull(payload.latitude) ?? undefined,
    lon_grid: finiteOrNull(payload.longitude) ?? undefined,
    date: targetDate,
    temp_min_c: tempMin,
    temp_max_c: tempMax,
    precip_sum_mm: precip,
    wind_max_kmh: wind,
    et0_mm: et0,
    precip_7d_mm: precip7d,
    fire_weather_index: fallbackFireIndex(tempMax, precip7d),
  };
}

export async function fetchCellWeather(
  lat: number, lon: number, date?: string,
): Promise<CellWeather> {
  const params = new URLSearchParams({ lat: String(lat), lon: String(lon) });
  if (date) params.set("date", date);
  let backendReason = "backend unavailable";
  try {
    const res = await fetch(`${API_BASE}/api/cell_weather?${params}`, {
      signal: AbortSignal.timeout(5_000),
    });
    if (res.ok) {
      const weather = await res.json() as CellWeather;
      if (weather.available) return weather;
      backendReason = weather.reason ?? "backend weather unavailable";
    } else {
      backendReason = `backend HTTP ${res.status}`;
    }
  } catch (error) {
    backendReason =
      error instanceof Error ? error.message : "backend request failed";
  }

  try {
    const fallback = await fetchOpenMeteoFallback(lat, lon, date);
    if (fallback.available) return fallback;
    return {
      available: false,
      reason: `${backendReason}; ${fallback.reason ?? "fallback unavailable"}`,
    };
  } catch {
    return {
      available: false,
      reason: `${backendReason}; Open-Meteo fallback unavailable`,
    };
  }
}
