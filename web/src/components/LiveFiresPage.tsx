import { useEffect, useMemo, useState } from "react";
import type { FireFeature, GistdaFeature, LiveFireMeta } from "../types";
import {
  ageDays,
  parseGistdaTime,
  type TimestampPrecision,
} from "../utils/gistdaTime";
import { fmtTr, useLang } from "../utils/i18n";

interface Props {
  liveFires: GistdaFeature[];               // GISTDA NRT — live <30 min latency
  observed: FireFeature[];                  // NASA FIRMS observed (last days)
  observedAvailable: boolean;
  liveFireMeta: LiveFireMeta;
  onRefresh?: () => void;
  onNavigateToMap: (lat: number, lon: number) => void;
}

// ─────────────────────────────────────────────────────────────
// Live Fires Feed page — news-dashboard-style chronological list
// of REAL satellite fire detections.
//
// Two real-data sources with official links for separate verification:
//   1. GISTDA NRT VIIRS  — Thai government, < 30 min latency
//      verify: https://fire.gistda.or.th/
//   2. NASA FIRMS NRT    — NASA Earth Data, satellite thermal anomalies
//      verify: https://firms.modaps.eosdis.nasa.gov/map/
//
// What this page is NOT:
//   - Predictions (those live on /#dashboard)
//   - Citizen reports (no input mechanism yet)
//
// Each incident card links to the source's official map for verification.
// ─────────────────────────────────────────────────────────────

type IncidentSource = "GISTDA" | "FIRMS";

interface Incident {
  id: string;
  source: IncidentSource;
  detectedAt: number;         // unix ms
  detectedAtStr: string;      // human-readable
  lat: number;
  lon: number;
  province: string;
  district: string;
  satellite: string;
  confidence: string;
  detectionCount: number | null;
  landUse: string;
  verifyUrl: string;
  daysOld: number | null;       // null means source time is not usable
  timePrecision: TimestampPrecision | null;
}

function validCoordinate(lat: number, lon: number): boolean {
  return Number.isFinite(lat) &&
    Number.isFinite(lon) &&
    lat >= -90 &&
    lat <= 90 &&
    lon >= -180 &&
    lon <= 180;
}

function _gistdaToIncident(f: GistdaFeature): Incident | null {
  const a = f.attributes;
  const parsed = parseGistdaTime(
    a.date as number | string | null | undefined,
    a.time as number | string | null | undefined,
  );
  const lat = Number(a.latitude);
  const lon = Number(a.longitude);
  if (!validCoordinate(lat, lon)) return null;
  return {
    id: `gistda-${lat.toFixed(4)}-${lon.toFixed(4)}-${a.date ?? ""}-${a.time ?? ""}`,
    source: "GISTDA",
    detectedAt: parsed?.ms ?? 0,
    detectedAtStr: parsed?.iso ?? "—",
    lat,
    lon,
    province: (a.pv_tn ?? "").toString() || "—",
    district: (a.ap_tn ?? "").toString() || "",
    satellite: (a.satellite ?? "VIIRS").toString(),
    confidence: (a.confident ?? "").toString(),
    detectionCount: null,
    landUse: (a.lu_name ?? "").toString() || "",
    verifyUrl: "https://fire.gistda.or.th/",
    daysOld: ageDays(parsed?.ms ?? null),
    timePrecision: parsed?.precision ?? null,
  };
}

// Approximate Thailand bbox — used as a fallback for FIRMS observed features
// that don't have a province annotation (risk_map.py only annotates predicted
// cells; observed FIRMS rows in the GeoJSON keep raw lat/lon only).
// Bounds cover Mae Hong Son (NW) → Yala (S) → Ubon Ratchathani (E).
function _isInThailandBbox(lat: number, lon: number): boolean {
  return lat >= 5.5 && lat <= 20.6 && lon >= 97.3 && lon <= 105.7;
}

function _firmsObservedToIncident(f: FireFeature): Incident | null {
  const p = f.properties;
  const [lon, lat] = f.geometry.coordinates;
  if (!validCoordinate(lat, lon)) return null;
  const dateStr = p.date ?? "";
  const dateOnly = /^\d{4}-\d{2}-\d{2}$/.test(dateStr);
  const parsedMs = dateStr
    ? Date.parse(dateOnly ? `${dateStr}T00:00:00+07:00` : dateStr)
    : Number.NaN;
  const ms = Number.isFinite(parsedMs) ? parsedMs : 0;
  const detectionCount =
    typeof p.fire_count === "number" && Number.isFinite(p.fire_count)
      ? p.fire_count
      : null;
  return {
    id: `firms-${lat.toFixed(4)}-${lon.toFixed(4)}-${dateStr}`,
    source: "FIRMS",
    detectedAt: ms,
    detectedAtStr: dateStr,
    lat,
    lon,
    province: p.province ?? "—",
    district: "",
    satellite: "VIIRS-SNPP/NOAA20",
    confidence: "",
    detectionCount,
    landUse: "",
    verifyUrl: `https://firms.modaps.eosdis.nasa.gov/map/#l:viirs;@${lon.toFixed(3)},${lat.toFixed(3)},9z`,
    daysOld: ageDays(ms || null),
    timePrecision: ms ? (dateOnly ? "date" : "minute") : null,
  };
}

function _timeAgo(
  incident: Incident,
  t: (k: string, fb?: string) => string,
): string {
  if (!incident.detectedAt || incident.daysOld == null) {
    return t("live.time.unavailable");
  }
  if (incident.timePrecision === "date") {
    return t("live.time.date_only");
  }
  const diff = Math.max(0, (Date.now() - incident.detectedAt) / 1000);
  if (diff < 60)    return fmtTr(t("live.ago.seconds"), { n: Math.round(diff) });
  if (diff < 3600)  return fmtTr(t("live.ago.minutes"), { n: Math.round(diff / 60) });
  if (diff < 86400) return fmtTr(t("live.ago.hours"),   { n: Math.round(diff / 3600) });
  return                   fmtTr(t("live.ago.days"),    { n: Math.round(diff / 86400) });
}

const SOURCE_COLOR: Record<IncidentSource, string> = {
  GISTDA: "#06b6d4",   // cyan — matches live fires layer
  FIRMS:  "#f97316",   // orange — matches observed layer
};

const SOURCE_INFO: Record<IncidentSource, { full: string; org: string }> = {
  GISTDA: { full: "GISTDA NRT VIIRS", org: "สำนักงานพัฒนาเทคโนโลยีอวกาศและภูมิสารสนเทศ" },
  FIRMS:  { full: "NASA FIRMS",        org: "NASA Earth Science Data" },
};

export default function LiveFiresPage({
  liveFires,
  observed,
  observedAvailable,
  liveFireMeta,
  onRefresh,
  onNavigateToMap,
}: Props) {
  const { t } = useLang();
  const [sourceFilter, setSourceFilter] = useState<IncidentSource | "ALL">("ALL");
  const [provinceFilter, setProvinceFilter] = useState<string>("ALL");
  const [timeWindow, setTimeWindow] = useState<number>(3); // days
  const [search, setSearch] = useState<string>("");
  // Thailand-only filter — defaults ON because the FIRMS layer's BBOX
  // (96–107°E, 4–22°N) spills into Cambodia/Vietnam/Myanmar/Laos.
  // Without this filter the feed is mostly cross-border fires which isn't
  // what a Thai operator wants to see.
  const [thailandOnly, setThailandOnly] = useState<boolean>(true);
  const [showHelp, setShowHelp] = useState<boolean>(false);
  // Pagination — start with 50 visible, "load more" reveals 50 at a time.
  const [pageSize, setPageSize] = useState<number>(50);

  const incidents = useMemo<Incident[]>(() => {
    const out: Incident[] = [];
    for (const f of liveFires) {
      const incident = _gistdaToIncident(f);
      if (incident) out.push(incident);
    }
    for (const f of observed) {
      const incident = _firmsObservedToIncident(f);
      if (incident) out.push(incident);
    }
    out.sort((a, b) => b.detectedAt - a.detectedAt);
    return out;
  }, [liveFires, observed]);

  const provinces = useMemo(() => {
    const s = new Set<string>();
    for (const i of incidents) if (i.province && i.province !== "—") s.add(i.province);
    return Array.from(s).sort();
  }, [incidents]);

  const filtered = useMemo(() => {
    return incidents.filter((i) => {
      if (sourceFilter !== "ALL" && i.source !== sourceFilter) return false;
      if (provinceFilter !== "ALL" && i.province !== provinceFilter) return false;
      if (i.daysOld == null || i.daysOld > timeWindow) return false;
      // A date-only FIRMS row cannot honestly be placed inside an exact
      // sub-day window. Keep it in 3/7/30-day views, where day precision is
      // sufficient, instead of presenting it as a precise "past 24 h" hit.
      if (timeWindow <= 1 && i.timePrecision !== "minute") return false;
      // Thailand only — accept if province is annotated OR coordinates fall
      // inside the Thai bbox (fallback for FIRMS observed rows, which the
      // backend doesn't yet annotate with province).
      if (thailandOnly) {
        const hasProvince = i.province && i.province !== "—";
        const inBbox = _isInThailandBbox(i.lat, i.lon);
        if (!hasProvince && !inBbox) return false;
      }
      if (search.trim()) {
        const q = search.toLowerCase();
        if (
          !i.province.toLowerCase().includes(q) &&
          !i.district.toLowerCase().includes(q) &&
          !`${i.lat},${i.lon}`.includes(q)
        ) return false;
      }
      return true;
    });
  }, [incidents, sourceFilter, provinceFilter, timeWindow, search, thailandOnly]);

  // Source-overlap cue: this is supporting context, not independent proof.
  // GISTDA and FIRMS may distribute observations from the same VIIRS sensor.
  const sourceOverlap = useMemo(() => {
    const map = new Map<string, IncidentSource[]>();
    const others = incidents;
    for (const i of incidents) {
      const matches: IncidentSource[] = [];
      for (const o of others) {
        if (o.id === i.id || o.source === i.source) continue;
        const dLat = (i.lat - o.lat) * 111;
        const dLon = (i.lon - o.lon) * 111 * Math.cos((i.lat * Math.PI) / 180);
        const km = Math.sqrt(dLat * dLat + dLon * dLon);
        if (!i.detectedAt || !o.detectedAt) continue;
        const dt = Math.abs(i.detectedAt - o.detectedAt);
        if (km <= 12 && dt <= 24 * 3600 * 1000) matches.push(o.source);
      }
      if (matches.length) map.set(i.id, Array.from(new Set(matches)));
    }
    return map;
  }, [incidents]);

  const sourceCounts = useMemo(() => {
    const c = { GISTDA: 0, FIRMS: 0, total: 0, last1h: 0, last24h: 0 };
    for (const i of incidents) {
      if (i.source === "GISTDA") c.GISTDA++;
      if (i.source === "FIRMS") c.FIRMS++;
      c.total++;
      if (
        i.timePrecision === "minute" &&
        i.daysOld != null &&
        i.daysOld * 24 <= 1
      ) {
        c.last1h++;
      }
      if (
        i.timePrecision === "minute" &&
        i.daysOld != null &&
        i.daysOld <= 1
      ) {
        c.last24h++;
      }
    }
    return c;
  }, [incidents]);

  const liveStatus = liveFireMeta.status;
  const hasCachedGistda = liveFires.length > 0;
  const waitingForFirstLiveResult =
    (liveStatus === "idle" || liveStatus === "loading") &&
    !hasCachedGistda;
  const allSourcesUnavailable =
    liveStatus === "error" &&
    !hasCachedGistda &&
    !observedAvailable;
  const effectiveLiveStatus =
    liveStatus === "error" && hasCachedGistda
      ? "cached"
      : liveFireMeta.isStale && hasCachedGistda
        ? "cached"
        : liveStatus;
  const liveStatusLabel = fmtTr(
    t(`live.status.${effectiveLiveStatus}`),
    { count: liveFires.length },
  );

  // Reset pagination whenever filters change so user starts from the top
  // again — otherwise switching province while at page 200 looks empty.
  useEffect(() => {
    setPageSize(50);
  }, [sourceFilter, provinceFilter, timeWindow, search, thailandOnly]);

  return (
    <div className="notify-page">
      <header className="notify-page-header" style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 16, flexWrap: "wrap" }}>
        <div>
          <h1>{t("live.title")}</h1>
          <p className="notify-page-subtitle">
            {t("live.subtitle")}
            <button
              type="button"
              onClick={() => setShowHelp(!showHelp)}
              aria-expanded={showHelp}
              aria-controls="live-feed-help"
              style={{ marginLeft: 8, background: "transparent", border: "none", color: "var(--accent)", cursor: "pointer", fontSize: 12, fontWeight: 600 }}
            >
              {showHelp ? t("live.help.hide") : t("live.help.show")}
            </button>
          </p>
        </div>
        <div style={{ display: "flex", gap: 6, alignItems: "center", flexDirection: "column" }}>
          <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
            <span
              className={`status-badge ${
                effectiveLiveStatus === "ok"
                  ? "great"
                  : effectiveLiveStatus === "error"
                    ? "bad"
                    : "ok"
              }`}
              role="status"
            >
              ⚡ GISTDA {liveStatusLabel}
            </span>
            {onRefresh && (
              <button
                type="button"
                className="action-btn"
                onClick={onRefresh}
                title={t("live.error.retry")}
              >
                ↻ {t("live.error.retry")}
              </button>
            )}
          </div>
          {liveFireMeta.lastFetch && (
            <div style={{ fontSize: 10, color: "var(--text-3)", textAlign: "right" }}>
              {fmtTr(t("live.updated"), { time: liveFireMeta.lastFetch.toLocaleTimeString() })}
            </div>
          )}
        </div>
      </header>

      {showHelp && (
        <div id="live-feed-help" className="help-card" style={{
          background: "var(--surface)",
          border: "1px solid var(--border)",
          borderLeft: "4px solid var(--accent)",
          borderRadius: 8,
          padding: "16px 20px",
          marginBottom: 16,
        }}>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(2, 1fr)", gap: 18, marginBottom: 16 }}>
            {/* GISTDA */}
            <div style={{ borderLeft: `3px solid ${SOURCE_COLOR.GISTDA}`, paddingLeft: 14 }}>
              <h4 style={{ fontSize: 13, fontWeight: 700, marginBottom: 6, color: SOURCE_COLOR.GISTDA }}>
                {t("live.help.gistda")}
              </h4>
              <p style={{ fontSize: 12, color: "var(--text-2)", lineHeight: 1.55, marginBottom: 6 }}>
                {t("live.help.gistda.sub")}
              </p>
              <ul style={{ fontSize: 11, color: "var(--text-3)", paddingLeft: 18, lineHeight: 1.6 }}>
                <li>{t("live.help.gistda.lat")}</li>
                <li>{t("live.help.gistda.cov")}</li>
                <li>{t("live.help.gistda.data")}</li>
                <li>{t("live.help.gistda.verify")}: <a href="https://fire.gistda.or.th/" target="_blank" rel="noopener noreferrer" style={{ color: SOURCE_COLOR.GISTDA }}>fire.gistda.or.th</a></li>
              </ul>
            </div>
            {/* FIRMS */}
            <div style={{ borderLeft: `3px solid ${SOURCE_COLOR.FIRMS}`, paddingLeft: 14 }}>
              <h4 style={{ fontSize: 13, fontWeight: 700, marginBottom: 6, color: SOURCE_COLOR.FIRMS }}>
                {t("live.help.firms")}
              </h4>
              <p style={{ fontSize: 12, color: "var(--text-2)", lineHeight: 1.55, marginBottom: 6 }}>
                {t("live.help.firms.sub")}
              </p>
              <ul style={{ fontSize: 11, color: "var(--text-3)", paddingLeft: 18, lineHeight: 1.6 }}>
                <li>{t("live.help.firms.lat")}</li>
                <li>{t("live.help.firms.cov")}</li>
                <li>{t("live.help.firms.data")}</li>
                <li>{t("live.help.firms.verify")}: <a href="https://firms.modaps.eosdis.nasa.gov/map/" target="_blank" rel="noopener noreferrer" style={{ color: SOURCE_COLOR.FIRMS }}>firms.modaps.eosdis.nasa.gov</a></li>
              </ul>
            </div>
          </div>

          <div style={{ padding: "12px 14px", background: "var(--surface-2)", borderRadius: 6, marginBottom: 12 }}>
            <h4 style={{ fontSize: 12, fontWeight: 700, marginBottom: 6, color: "var(--text)" }}>
              {t("live.help.howreal")}
            </h4>
            <ol style={{ fontSize: 12, color: "var(--text-2)", paddingLeft: 18, lineHeight: 1.7 }}>
              <li>{t("live.help.bullet1")}</li>
              <li>{t("live.help.bullet2")}</li>
              <li>{t("live.help.bullet3")}</li>
            </ol>
          </div>

          <div style={{ padding: "10px 12px", background: "rgba(234, 179, 8, 0.10)", border: "1px solid rgba(234, 179, 8, 0.3)", borderRadius: 6, fontSize: 11, color: "var(--text-2)", lineHeight: 1.55 }}>
            {t("live.help.province")}
          </div>
        </div>
      )}

      {liveStatus === "error" && !hasCachedGistda && (
        <div className="analytics-error" role="alert">
          <div>
            <strong>{t("live.error.title")}</strong>
            <span>
              {liveFireMeta.error || t("live.error.hint")}
            </span>
          </div>
          {onRefresh && (
            <button type="button" onClick={onRefresh}>
              {t("live.error.retry")}
            </button>
          )}
        </div>
      )}

      {/* Top counts strip */}
      <div className="alert-counts" style={{ gridTemplateColumns: "repeat(4, 1fr)" }}>
        <div className="alert-count-card" style={{ borderLeftColor: "#ef4444" }}>
          <div className="alert-count-label" style={{ color: "#ef4444" }}>{t("live.count.h1")}</div>
          <div className="alert-count-value">
            {waitingForFirstLiveResult && !observedAvailable
              ? "—"
              : sourceCounts.last1h}
          </div>
        </div>
        <div className="alert-count-card" style={{ borderLeftColor: "#f97316" }}>
          <div className="alert-count-label" style={{ color: "#f97316" }}>{t("live.count.h24")}</div>
          <div className="alert-count-value">
            {waitingForFirstLiveResult && !observedAvailable
              ? "—"
              : sourceCounts.last24h}
          </div>
        </div>
        <button
          type="button"
          className={`alert-count-card ${sourceFilter === "GISTDA" ? "active" : ""}`}
          onClick={() => setSourceFilter(sourceFilter === "GISTDA" ? "ALL" : "GISTDA")}
          aria-pressed={sourceFilter === "GISTDA"}
          style={{ borderLeftColor: SOURCE_COLOR.GISTDA }}
        >
          <div className="alert-count-label" style={{ color: SOURCE_COLOR.GISTDA }}>GISTDA</div>
          <div className="alert-count-value">
            {liveStatus === "ok" || hasCachedGistda
              ? sourceCounts.GISTDA
              : "—"}
          </div>
        </button>
        <button
          type="button"
          className={`alert-count-card ${sourceFilter === "FIRMS" ? "active" : ""}`}
          onClick={() => setSourceFilter(sourceFilter === "FIRMS" ? "ALL" : "FIRMS")}
          aria-pressed={sourceFilter === "FIRMS"}
          style={{ borderLeftColor: SOURCE_COLOR.FIRMS }}
        >
          <div className="alert-count-label" style={{ color: SOURCE_COLOR.FIRMS }}>FIRMS</div>
          <div className="alert-count-value">
            {observedAvailable ? sourceCounts.FIRMS : "—"}
          </div>
        </button>
      </div>

      {/* Filters */}
      <div className="alert-filters">
        <label style={{
          display: "inline-flex", alignItems: "center", gap: 6,
          padding: "7px 10px",
          background: thailandOnly ? "var(--accent-soft)" : "var(--surface)",
          border: `1px solid ${thailandOnly ? "var(--accent)" : "var(--border)"}`,
          borderRadius: 5,
          fontSize: 12,
          color: thailandOnly ? "var(--accent)" : "var(--text-2)",
          fontWeight: 600,
          cursor: "pointer",
        }}>
          <input
            type="checkbox"
            checked={thailandOnly}
            onChange={(e) => setThailandOnly(e.target.checked)}
            style={{ margin: 0 }}
          />
          {t("live.filter.thailand")}
        </label>
        <select
          value={sourceFilter}
          onChange={(e) => setSourceFilter(e.target.value as IncidentSource | "ALL")}
          aria-label={t("live.filter.source.label")}
        >
          <option value="ALL">{t("live.filter.all_source")}</option>
          <option value="GISTDA">GISTDA only</option>
          <option value="FIRMS">FIRMS only</option>
        </select>
        <select
          value={provinceFilter}
          onChange={(e) => setProvinceFilter(e.target.value)}
          aria-label={t("live.filter.province.label")}
        >
          <option value="ALL">{t("live.filter.all_province")}</option>
          {provinces.map((p) => <option key={p} value={p}>{p}</option>)}
        </select>
        <select
          value={timeWindow}
          onChange={(e) => setTimeWindow(Number(e.target.value))}
          aria-label={t("live.filter.time.label")}
        >
          <option value={0.25}>{t("live.filter.window.6h")}</option>
          <option value={1}>{t("live.filter.window.24h")}</option>
          <option value={3}>{t("live.filter.window.3d")}</option>
          <option value={7}>{t("live.filter.window.7d")}</option>
          <option value={30}>{t("live.filter.window.30d")}</option>
        </select>
        <input
          type="search"
          placeholder={t("live.search.placeholder")}
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          aria-label={t("live.filter.search.label")}
        />
        <div className="alert-filter-summary">
          {fmtTr(t("live.paginator.summary"), {
            n: filtered.length,
            total: incidents.length,
          })}
        </div>
      </div>

      {/* Feed */}
      {filtered.length === 0 ? (
        <div className="empty-state">
          <div className="empty-icon">🌲</div>
          <div className="empty-title">
            {waitingForFirstLiveResult && incidents.length === 0
              ? t("analytics.feed.status.loading")
              : allSourcesUnavailable
              ? t("live.empty.unavailable.title")
              : incidents.length === 0 &&
                  liveStatus === "ok" &&
                  observedAvailable
                ? t("live.empty.confirmed.title")
                : t("live.empty.title")}
          </div>
          <div className="empty-hint">
            {waitingForFirstLiveResult && incidents.length === 0
              ? t("analytics.feed.loading")
              : allSourcesUnavailable
              ? t("live.empty.unavailable.hint")
              : incidents.length === 0 &&
                  liveStatus === "ok" &&
                  observedAvailable
                ? t("live.empty.confirmed.hint")
                : t("live.empty.hint")}
          </div>
        </div>
      ) : (
        <div className="incident-feed">
          {filtered.slice(0, pageSize).map((i) => {
            const overlap = sourceOverlap.get(i.id);
            const now = Date.now();
            const isFresh =
              i.detectedAt > 0 &&
              i.detectedAt <= now &&
              i.timePrecision === "minute" &&
              now - i.detectedAt < 5 * 60 * 1000;
            return (
              <article
                key={i.id}
                className={`incident-card${isFresh ? " incident-card-fresh" : ""}`}
                style={{ borderLeftColor: SOURCE_COLOR[i.source] }}
              >
                <div className="incident-time">
                  <div className="incident-time-rel">{_timeAgo(i, t)}</div>
                  <div className="incident-time-abs">{i.detectedAtStr}</div>
                </div>
                <div className="incident-body">
                  <div className="incident-headline">
                    <span className="incident-source-badge" style={{ background: SOURCE_COLOR[i.source] + "22", color: SOURCE_COLOR[i.source], borderColor: SOURCE_COLOR[i.source] + "55" }}>
                      🛰 {i.source}
                    </span>
                    {overlap && overlap.length > 0 && (
                      <span
                        className="incident-verified"
                        title={fmtTr(t("live.overlap.tooltip"), {
                          sources: overlap.join(", "),
                        })}
                      >
                        ✓ {t("live.overlap.label")}
                      </span>
                    )}
                    {i.confidence && (
                      <span className="incident-conf">
                        {fmtTr(t("live.confidence.code"), {
                          code: i.confidence,
                        })}
                      </span>
                    )}
                    {i.detectionCount != null && (
                      <span className="incident-conf">
                        {fmtTr(t("live.detection_count"), {
                          count: i.detectionCount,
                        })}
                      </span>
                    )}
                  </div>
                  <h3 className="incident-location">
                    {i.province && i.province !== "—" ? (
                      <>
                        {i.province}
                        {i.district ? <span className="incident-district"> · {i.district}</span> : null}
                      </>
                    ) : _isInThailandBbox(i.lat, i.lon) ? (
                      <span style={{ color: "var(--text-2)", fontWeight: 500, fontSize: 14 }}>
                        {t("live.area.thailand")} <span style={{ color: "var(--text-3)", fontSize: 11 }}>{t("live.area.thailand.unk")}</span>
                      </span>
                    ) : (
                      <span style={{ color: "var(--text-3)", fontWeight: 500, fontSize: 13 }}>
                        {t("live.area.foreign")}
                      </span>
                    )}
                  </h3>
                  <div className="incident-meta">
                    <span className="mono">{i.lat.toFixed(4)}°N · {i.lon.toFixed(4)}°E</span>
                    {i.satellite && <span>📡 {i.satellite}</span>}
                    {i.landUse && <span>🌿 {i.landUse}</span>}
                  </div>
                  <div className="incident-actions">
                    <button
                      type="button"
                      className="action-btn"
                      style={{ padding: "4px 10px", fontSize: 11 }}
                      onClick={() => onNavigateToMap(i.lat, i.lon)}
                    >
                      {t("live.btn.view_map")}
                    </button>
                    <a
                      href={i.verifyUrl}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="action-btn"
                      style={{ padding: "4px 10px", fontSize: 11, textDecoration: "none" }}
                      title={`Verify on ${SOURCE_INFO[i.source].full} official map`}
                    >
                      {fmtTr(t("live.btn.verify"), { source: i.source })}
                    </a>
                  </div>
                </div>
              </article>
            );
          })}
          {filtered.length > pageSize ? (
            <div style={{
              display: "flex", flexDirection: "column", gap: 8, alignItems: "center",
              padding: "16px 12px", background: "var(--surface-2)", borderRadius: 8,
              marginTop: 6,
            }}>
              <div style={{ fontSize: 12, color: "var(--text-3)" }}>
                {fmtTr(t("live.paginator.summary"), { n: pageSize, total: filtered.length })}
              </div>
              <div style={{ display: "flex", gap: 8, flexWrap: "wrap", justifyContent: "center" }}>
                <button
                  type="button"
                  className="action-btn primary"
                  onClick={() => setPageSize(pageSize + 50)}
                >
                  {t("live.paginator.more50")}
                </button>
                <button
                  type="button"
                  className="action-btn"
                  onClick={() => setPageSize(pageSize + 200)}
                >
                  {t("live.paginator.more200")}
                </button>
                <button
                  type="button"
                  className="action-btn"
                  onClick={() => setPageSize(filtered.length)}
                >
                  {fmtTr(t("live.paginator.all"), { n: filtered.length })}
                </button>
              </div>
            </div>
          ) : pageSize > 50 ? (
            <div style={{ textAlign: "center", padding: 12 }}>
              <button
                type="button"
                className="action-btn"
                onClick={() => setPageSize(50)}
                style={{ fontSize: 11 }}
              >
                {t("live.paginator.collapse")}
              </button>
            </div>
          ) : null}
        </div>
      )}

      {/* Footer attribution */}
      <footer className="report-footer">
        <p style={{ fontSize: 11, color: "var(--text-3)" }}>
          📡 {t("live.subtitle")}
        </p>
        <div style={{ display: "flex", gap: 16, flexWrap: "wrap", marginTop: 8, justifyContent: "center" }}>
          <a href="https://fire.gistda.or.th/" target="_blank" rel="noopener noreferrer"
             style={{ fontSize: 11, color: SOURCE_COLOR.GISTDA, textDecoration: "none" }}>
            🛰 GISTDA · fire.gistda.or.th
          </a>
          <a href="https://firms.modaps.eosdis.nasa.gov/map/" target="_blank" rel="noopener noreferrer"
             style={{ fontSize: 11, color: SOURCE_COLOR.FIRMS, textDecoration: "none" }}>
            🛰 NASA FIRMS · firms.modaps.eosdis.nasa.gov
          </a>
        </div>
        <p style={{ fontSize: 10, color: "var(--text-3)", marginTop: 8 }}>
          {t("live.help.bullet2")}
        </p>
      </footer>
    </div>
  );
}
