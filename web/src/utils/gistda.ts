import type { GistdaFeature } from "../types";

// Live GISTDA hotspots — served by OUR FastAPI backend, never by GISTDA
// directly. The backend polls GISTDA's ArcGIS layers every ~5 minutes
// (server-side, advisory-lock guarded, persisted to PostgreSQL) and this
// module only reads that cache. Requesting more often than the backend
// polls cannot produce fresher data — the satellite source itself only
// updates per overpass.
const API_BASE = (import.meta.env.VITE_API_BASE ?? "").replace(/\/$/, "");

export interface LiveFireMetadata {
  count: number;
  isStale: boolean;
  lastSuccessAt: string | null;
  pollingIntervalSeconds: number | null;
  lastError: string | null;
}

export interface LiveFiresResult {
  features: GistdaFeature[];
  metadata: LiveFireMetadata;
}

export async function fetchLiveFires(signal?: AbortSignal): Promise<LiveFiresResult> {
  const controller = new AbortController();
  const abortFromCaller = () => controller.abort(signal?.reason);
  if (signal?.aborted) abortFromCaller();
  else signal?.addEventListener("abort", abortFromCaller, { once: true });
  const timer = globalThis.setTimeout(
    () => controller.abort(new DOMException("Live feed request timed out", "TimeoutError")),
    20_000,
  );
  try {
    const res = await fetch(`${API_BASE}/live-hotspots?limit=3000`, {
      signal: controller.signal,
      cache: "no-store",
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const payload = await res.json();
    if (
      !Array.isArray(payload?.features) ||
      payload.features.some(
        (feature: unknown) =>
          typeof feature !== "object" ||
          feature === null ||
          typeof (feature as { attributes?: unknown }).attributes !== "object" ||
          (feature as { attributes?: unknown }).attributes === null,
      )
    ) {
      throw new Error("Invalid live-hotspot response");
    }
    const meta = payload.metadata ?? {};
    return {
      // Backend already emits the legacy GistdaFeature shape under `features`.
      features: payload.features as GistdaFeature[],
      metadata: {
        count: Number(meta.count ?? 0),
        isStale: Boolean(meta.is_stale),
        lastSuccessAt: meta.last_success_at ?? null,
        pollingIntervalSeconds: meta.polling_interval_seconds ?? null,
        lastError: meta.last_error ?? null,
      },
    };
  } finally {
    globalThis.clearTimeout(timer);
    signal?.removeEventListener("abort", abortFromCaller);
  }
}

export const LIVE_FIRE_COLOR = "#06b6d4"; // cyan — distinct from urgency + observed
// Browser refresh of the backend CACHE. The backend polls GISTDA every
// GISTDA_POLL_SECONDS (default 300 s); refreshing the cache every 60 s keeps
// the UI snappy without implying the satellite updates that often.
export function normaliseRefreshSeconds(value: unknown): number {
  const seconds = Number(value);
  // A typo such as VITE_CLIENT_REFRESH_SECONDS=abc previously became a 0 ms
  // timeout and hammered /live-hotspots in a tight loop.
  return Number.isFinite(seconds) && seconds >= 15 ? seconds : 60;
}

export const LIVE_REFRESH_MS =
  normaliseRefreshSeconds(import.meta.env.VITE_CLIENT_REFRESH_SECONDS) * 1000;
