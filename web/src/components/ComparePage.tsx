import { useEffect, useMemo, useRef, useState } from "react";
import L from "leaflet";
import type { FireFeature, ValidationMetrics } from "../types";
import { fmtTr, useLang } from "../utils/i18n";
import { formatProbabilityPercent, readProbability } from "../utils/probability";
import {
  MIN_COMPARE_RATE_SAMPLE,
  computeCompareStats,
  contextWindowForBaseDate,
  dateFallsWithin,
  hasSufficientRateSample,
  isWindowMature,
  riskWindowForBaseDate,
  selectDefaultReviewDate,
  type CompareStats,
} from "../utils/compareStats";
import { isActionableUrgency } from "../utils/risk";

interface Props {
  allFeatures: FireFeature[];
  observedFeatures: FireFeature[];
  metrics: ValidationMetrics | null;
}

// ─────────────────────────────────────────────────────────────
// Compare page — "ทำนายไว้ vs เกิดจริง"
//
// Side-by-side honesty check on the model. Uses validation_status that
// risk_map.py already writes on every cell in the historical GeoJSON:
//
//   hit    = this scored cell recorded a FIRMS hotspot in the model window
//   miss   = this scored cell did not record one in that persisted window
//   future = the configured target window has not closed yet
//
// Overlay map colour code:
//   🟢 green  = ทำนายถูก (hit — predicted AND occurred)
//   🟡 yellow = เตือนเกิน (false alarm — predicted but no fire)
//   🔴 red    = FIRMS observation shown as visual context (not a computed miss)
//
// Precision comes from persisted same-cell validation_status. Recall/F1 stay
// unavailable because the map artifact is not a complete outcome corpus.
//
// Window selector lets the operator pick a single past base_date snapshot to
// audit. The default prefers the newest operational-tier snapshot with enough
// decided alerts to show a rate, rather than presenting a misleading tiny n.
// ─────────────────────────────────────────────────────────────

type Stats = CompareStats;

const PALETTE = {
  hit:   "#22c55e",
  alarm: "#eab308",
  observed: "#ef4444",
  future:"#6c707a",
};

function escapeHtml(value: unknown): string {
  return String(value ?? "—")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

export default function ComparePage({
  allFeatures,
  observedFeatures,
  metrics,
}: Props) {
  const { t } = useLang();
  const configuredHorizon = Number(metrics?.imminent_days);
  const horizonDays =
    Number.isInteger(configuredHorizon) &&
    configuredHorizon >= 1 &&
    configuredHorizon <= 30
      ? configuredHorizon
      : 3;
  const [baseDate, setBaseDate] = useState<string>("__latest__");
  const [view, setView] = useState<"split" | "overlay">("overlay");
  // Hit/miss uses the model's exact same-cell target (validation_status from
  // risk_map.py) — no radius involved. dayWindow only widens which observed
  // FIRMS points are drawn on the map for visual context. It must never alter
  // the persisted target window used to assign validation_status.
  const [dayWindow, setDayWindow] = useState<number>(1);
  // Audit tier filter — default is the deployed operator tiers. Background is
  // a valid absolute-probability tier, but it is not an alert population.
  const [tierFilter, setTierFilter] = useState<"HIGH_MED" | "HIGH" | "ALL">("HIGH_MED");
  const leftMapRef = useRef<HTMLDivElement | null>(null);
  const rightMapRef = useRef<HTMLDivElement | null>(null);
  const overlayMapRef = useRef<HTMLDivElement | null>(null);

  // Available base_dates with closed validation windows (have hit/miss data,
  // not 100% future). Show newest first.
  const availableDates = useMemo(() => {
    type DateCounts = {
      hit: number;
      miss: number;
      future: number;
      actionableHit: number;
      actionableMiss: number;
      actionableFuture: number;
    };
    const dates = new Map<string, DateCounts>();
    for (const f of allFeatures) {
      const bd = f.properties.base_date;
      const st = f.properties.validation_status;
      if (!bd) continue;
      let entry = dates.get(bd);
      if (!entry) {
        entry = {
          hit: 0,
          miss: 0,
          future: 0,
          actionableHit: 0,
          actionableMiss: 0,
          actionableFuture: 0,
        };
        dates.set(bd, entry);
      }
      const actionable = isActionableUrgency(f.properties.urgency_level);
      if (st === "hit") entry.hit += 1;
      else if (st === "miss") entry.miss += 1;
      else if (st === "future") entry.future += 1;
      if (actionable && st === "hit") entry.actionableHit += 1;
      else if (actionable && st === "miss") entry.actionableMiss += 1;
      else if (actionable && st === "future") entry.actionableFuture += 1;
    }
    return Array.from(dates.entries())
      .map(([d, c]) => ({
        date: d,
        ...c,
        decided: c.hit + c.miss,
        actionableDecided: c.actionableHit + c.actionableMiss,
      }))
      .sort((a, b) => b.date.localeCompare(a.date));
  }, [allFeatures]);

  // Prefer the newest operational snapshot with a denominator large enough to
  // report a percentage. Smaller/newer snapshots remain explicitly selectable.
  const resolvedDate = useMemo(() => {
    if (baseDate !== "__latest__") return baseDate;
    return selectDefaultReviewDate(availableDates);
  }, [baseDate, availableDates]);

  // Filter to this snapshot's predicted features, then to the audit-tier
  // filter so the precision/recall numbers reflect the operationally-
  // relevant tier (default HIGH+WATCH). Background can still be selected for
  // diagnostic inspection, but is not presented as an operational alert rate.
  const snapshot = useMemo(() => {
    if (!resolvedDate) return [];
    const base = allFeatures.filter((f) => f.properties.base_date === resolvedDate);
    if (tierFilter === "ALL") return base;
    if (tierFilter === "HIGH") {
      return base.filter((f) => f.properties.urgency_level === "HIGH");
    }
    // HIGH_MED
    return base.filter((f) =>
      f.properties.urgency_level === "HIGH" ||
      f.properties.urgency_level === "MEDIUM"
    );
  }, [allFeatures, resolvedDate, tierFilter]);

  const riskWindow = useMemo(
    () => resolvedDate
      ? riskWindowForBaseDate(resolvedDate, horizonDays)
      : null,
    [resolvedDate, horizonDays],
  );
  const contextWindow = useMemo(
    () => resolvedDate
      ? contextWindowForBaseDate(resolvedDate, dayWindow, horizonDays)
      : null,
    [resolvedDate, dayWindow, horizonDays],
  );

  // Visual-only FIRMS context. Widening this range must not alter metrics.
  const observedInWindow = useMemo(() => {
    if (!resolvedDate) return [];
    return observedFeatures.filter((f) =>
      dateFallsWithin(f.properties.date, contextWindow)
    );
  }, [resolvedDate, observedFeatures, contextWindow]);

  type Match = {
    feature: FireFeature;
    status: "hit" | "alarm" | "future" | "unknown";
  };
  const matched = useMemo<{
    rows: Match[];
    isWindowImmature: boolean;
  }>(() => {
    if (!resolvedDate) {
      return { rows: [], isWindowImmature: true };
    }

    // STRICT MODEL TARGET ONLY. Each prediction's hit/miss comes from the
    // server-computed `validation_status` (same 0.1° cell, base_date+1 ..
    // base_date+horizon inclusive). This supports alert precision. The map
    // GeoJSON is display-oriented and does not retain the complete observed
    // outcome universe needed for recall/F1, so those metrics stay unavailable.
    const rows: Match[] = snapshot.map((pred) => {
      const st = pred.properties.validation_status;
      const status: Match["status"] =
        st === "hit"
          ? "hit"
          : st === "miss"
            ? "alarm"
            : st === "future"
              ? "future"
              : "unknown";
      return { feature: pred, status };
    });

    const isWindowImmature = !isWindowMature(rows.map((r) => r.status));
    return { rows, isWindowImmature };
  }, [snapshot, resolvedDate]);

  const stats: Stats = useMemo(
    () => computeCompareStats(matched.rows.map((m) => m.status)),
    [matched],
  );
  const precisionDenominator = stats.hits + stats.falseAlarms;
  const hasPrecisionOutcomes = precisionDenominator > 0;
  const hasPrecisionSample =
    hasSufficientRateSample(precisionDenominator);

  const heldOutReport =
    metrics?.watch_test_metrics ?? metrics?.high_test_metrics;
  const heldOutRecall =
    heldOutReport?.recall ?? metrics?.deployment_recall;
  const heldOutN = heldOutReport?.n;
  const heldOutTier = metrics?.watch_test_metrics
    ? t("compare.methodology.tier.watch")
    : t("compare.methodology.tier.high");
  const methodology =
    typeof heldOutRecall === "number" && isFinite(heldOutRecall)
      ? fmtTr(t("compare.methodology.with_metric"), {
          tier: heldOutTier,
          pct: `${(heldOutRecall * 100).toFixed(0)}%`,
          n:
            typeof heldOutN === "number" && isFinite(heldOutN)
              ? heldOutN.toLocaleString()
              : t("compare.methodology.unknown_n"),
        })
      : t("compare.methodology");

  // ── Build Leaflet maps ──
  useEffect(() => {
    const ctxMaps: L.Map[] = [];
    const isLight = document.documentElement.getAttribute("data-theme") === "light";
    const tileUrl = isLight
      ? "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
      : "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png";

    function attach(container: HTMLDivElement | null, mode: "predicted" | "actual" | "overlay") {
      if (!container) return null;
      const map = L.map(container, { center: [15.0, 101.0], zoom: 6 });
      L.tileLayer(tileUrl, { attribution: "© OpenStreetMap" }).addTo(map);
      ctxMaps.push(map);

      const statusByFeature = new Map(matched.rows.map((m) => [m.feature, m.status]));

      if (mode === "predicted" || mode === "overlay") {
        for (const f of snapshot) {
          const [lon, lat] = f.geometry.coordinates;
          const u = f.properties.urgency_level;
          const probabilityText = formatProbabilityPercent(readProbability(f.properties));
          const st = statusByFeature.get(f) ?? "unknown";
          let color = PALETTE.future;
          let label = st === "future"
            ? t("compare.label.future")
            : t("compare.label.unknown");
          if (st === "hit") { color = PALETTE.hit; label = t("compare.label.hit"); }
          else if (st === "alarm") { color = PALETTE.alarm; label = t("compare.label.alarm"); }
          const radius = u === "CRITICAL" ? 7 : u === "HIGH" ? 6 : u === "MEDIUM" ? 5 : 4;
          L.circleMarker([lat, lon], {
            radius, fillColor: color, color: "#fff", weight: 1, fillOpacity: 0.85,
          }).bindPopup(
            `<div style="font-size:12px"><b>${escapeHtml(label)}</b><br>` +
            `${escapeHtml(f.properties.province)}<br>` +
            `urgency: ${escapeHtml(u)}, prob: ${escapeHtml(probabilityText)}</div>`
          ).addTo(map);
        }
      }
      if (mode === "actual" || mode === "overlay") {
        for (const f of observedInWindow) {
          const [lon, lat] = f.geometry.coordinates;
          L.circleMarker([lat, lon], {
            radius: mode === "overlay" ? 5 : 4,
            fillColor: PALETTE.observed,
            color: "#fff",
            weight: mode === "overlay" ? 1 : 0.5,
            fillOpacity: 0.9,
          }).bindPopup(
            `<small>${escapeHtml(t("compare.legend.observed"))} · ` +
            `${escapeHtml(f.properties.date)}</small>`,
          ).addTo(map);
        }
      }
      return map;
    }

    if (view === "overlay") {
      attach(overlayMapRef.current, "overlay");
    } else {
      attach(leftMapRef.current, "predicted");
      attach(rightMapRef.current, "actual");
    }

    return () => {
      ctxMaps.forEach((m) => m.remove());
    };
  }, [view, snapshot, observedInWindow, matched, t]);

  const fmt = (v: number, d = 1) => (v * 100).toFixed(d) + "%";

  return (
    <div className="notify-page">
      <header className="notify-page-header" style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 16, flexWrap: "wrap" }}>
        <div>
          <h1>{t("page.compare.title", "🆚 Predicted vs Actual")}</h1>
          <p className="notify-page-subtitle">
            {t("page.compare.subtitle", "Model honesty audit — overlay past predictions onto real FIRMS observations")}
          </p>
        </div>
        <div style={{ display: "flex", gap: 6, alignItems: "center", flexWrap: "wrap" }}>
          <select
            value={baseDate}
            onChange={(e) => setBaseDate(e.target.value)}
            aria-label={t("compare.basedate.label")}
            style={{ padding: "7px 10px", background: "var(--surface-2)", border: "1px solid var(--border)", borderRadius: 5, color: "var(--text)", fontSize: 12, minWidth: 200 }}
          >
            <option value="__latest__">{t("compare.basedate.latest")}</option>
            {availableDates.map((d) => (
              <option key={d.date} value={d.date}>
                {fmtTr(t("compare.basedate.option"), {
                  date: d.date,
                  hits: d.actionableHit,
                  misses: d.actionableMiss,
                  pending: d.actionableFuture,
                })}
              </option>
            ))}
          </select>
          <div style={{ display: "flex", border: "1px solid var(--border)", borderRadius: 5, overflow: "hidden" }}>
            <button type="button" onClick={() => setView("overlay")} aria-pressed={view === "overlay"} className={`action-btn ${view === "overlay" ? "primary" : ""}`} style={{ borderRadius: 0, fontSize: 11 }}>
              {t("compare.view.overlay")}
            </button>
            <button type="button" onClick={() => setView("split")} aria-pressed={view === "split"} className={`action-btn ${view === "split" ? "primary" : ""}`} style={{ borderRadius: 0, fontSize: 11 }}>
              {t("compare.view.split")}
            </button>
          </div>
        </div>
      </header>

      {/* Stats strip */}
      <div className="alert-counts compare-count-grid">
        <div className="alert-count-card" style={{ borderLeftColor: PALETTE.hit }}>
          <div className="alert-count-label" style={{ color: PALETTE.hit }}>{t("compare.label.hit")}</div>
          <div className="alert-count-value">{stats.hits}</div>
        </div>
        <div className="alert-count-card" style={{ borderLeftColor: PALETTE.alarm }}>
          <div className="alert-count-label" style={{ color: PALETTE.alarm }}>{t("compare.label.alarm")}</div>
          <div className="alert-count-value">{stats.falseAlarms}</div>
        </div>
        <div className="alert-count-card" style={{ borderLeftColor: PALETTE.observed }}>
          <div className="alert-count-label" style={{ color: PALETTE.observed }}>{t("compare.label.observed")}</div>
          <div className="alert-count-value">{observedInWindow.length}</div>
        </div>
        <div className="alert-count-card" style={{ borderLeftColor: PALETTE.future }}>
          <div className="alert-count-label" style={{ color: PALETTE.future }}>{t("compare.label.future")}</div>
          <div className="alert-count-value">{stats.future}</div>
        </div>
      </div>

      {/* Methodology callout — clarify why these numbers differ from AccuracyHero */}
      <div style={{
        padding: "10px 14px", marginBottom: 12,
        background: "rgba(59, 130, 246, 0.08)",
        border: "1px solid rgba(59, 130, 246, 0.25)",
        borderRadius: 6, fontSize: 12, color: "var(--text-2)",
        lineHeight: 1.55,
      }}>
        {methodology}
        <div style={{ marginTop: 5, color: "var(--text-3)" }}>
          {fmtTr(t("compare.sample_basis"), {
            date: resolvedDate ?? "—",
            alerts: precisionDenominator,
            fires: "—",
            min: MIN_COMPARE_RATE_SAMPLE,
          })}
        </div>
      </div>

      {/* A future outcome window means even persisted alert precision is not
          yet decided for this snapshot. */}
      {matched.isWindowImmature && (
        <div style={{
          padding: "10px 14px", marginBottom: 12,
          background: "rgba(234, 179, 8, 0.08)",
          border: "1px solid rgba(234, 179, 8, 0.30)",
          borderRadius: 6, fontSize: 12, color: "var(--text-2)",
          lineHeight: 1.55, display: "flex", gap: 8, alignItems: "flex-start",
        }}>
          <span style={{ fontSize: 15, flexShrink: 0 }}>⏳</span>
          <span>
            <b style={{ color: "var(--text)" }}>{t("compare.pending.title")}</b>
            {" — "}
            {fmtTr(t("compare.pending.body"), {
              start: riskWindow?.start ?? "—",
              end: riskWindow?.end ?? "—",
            })}
          </span>
        </div>
      )}

      {/* dayWindow is visual-only. Tier changes the persisted alert population. */}
      <div style={{
        padding: "10px 14px", marginBottom: 12,
        background: "var(--surface-2)",
        border: "1px solid var(--border)",
        borderRadius: 6,
      }}>
        <div style={{ display: "flex", gap: 20, alignItems: "center", flexWrap: "wrap" }}>
          <div style={{ fontSize: 11, fontWeight: 600, textTransform: "uppercase", color: "var(--text-3)", letterSpacing: "0.06em" }}>
            {t("compare.matching.title")}
          </div>
          <label style={{ display: "flex", alignItems: "center", gap: 10, fontSize: 12, flex: "1 1 240px" }}>
            <span style={{ minWidth: 130 }}>
              {t("compare.matching.window")} <b style={{ color: "var(--accent)" }}>{dayWindow}</b>
            </span>
            <input
              type="range" min={0} max={3} step={1}
              value={dayWindow}
              onChange={(e) => setDayWindow(Number(e.target.value))}
              aria-label={t("compare.matching.window")}
              style={{ flex: 1, accentColor: "var(--accent)" }}
            />
            <span style={{ fontSize: 10, color: "var(--text-3)", minWidth: 110 }}>
              {t("compare.matching.window.hint")}
            </span>
          </label>
          <label style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 12, flex: "0 0 auto" }}>
            <span style={{ minWidth: 90, fontWeight: 600 }}>{t("compare.tier.label")}</span>
            <select
              value={tierFilter}
              onChange={(e) => setTierFilter(e.target.value as "HIGH_MED" | "HIGH" | "ALL")}
              style={{ padding: "5px 8px", background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 5, color: "var(--text)", fontSize: 12 }}
            >
              <option value="HIGH_MED">{t("compare.tier.high_med")}</option>
              <option value="HIGH">{t("compare.tier.high")}</option>
              <option value="ALL">{t("compare.tier.all")}</option>
            </select>
          </label>
        </div>
      </div>

      {/* Precision/Recall/F1 */}
      <section className="report-section" style={{ marginBottom: 14 }}>
        <div className="compare-metric-grid">
          <div style={{ padding: "12px 14px", background: "var(--surface-2)", borderRadius: 6 }}>
            <div style={{ fontSize: 10, color: "var(--text-3)", fontWeight: 700, textTransform: "uppercase" }}>{t("compare.precision")}</div>
            <div style={{ fontSize: 24, fontWeight: 700, color: hasPrecisionSample ? (stats.precision === 0 ? PALETTE.observed : "var(--text)") : "var(--text-3)", marginTop: 4 }}>
              {hasPrecisionSample ? fmt(stats.precision) : "—"}
            </div>
            <div style={{ fontSize: 11, color: "var(--text-3)", marginTop: 4 }}>
              {hasPrecisionSample
                ? fmtTr(t("compare.precision.count"), {
                    hits: stats.hits,
                    alerts: precisionDenominator,
                  })
                : hasPrecisionOutcomes
                  ? fmtTr(t("compare.precision.small_sample"), {
                      hits: stats.hits,
                      alerts: precisionDenominator,
                      min: MIN_COMPARE_RATE_SAMPLE,
                    })
                  : t("compare.precision.empty")}
            </div>
          </div>
          <div style={{ padding: "12px 14px", background: "var(--surface-2)", borderRadius: 6, opacity: 0.55 }}>
            <div style={{ fontSize: 10, color: "var(--text-3)", fontWeight: 700, textTransform: "uppercase" }}>
              {t("compare.recall")}
              <span style={{ marginLeft: 6, fontWeight: 400, textTransform: "none", fontSize: 9 }}>
                ({t("compare.state.unavailable")})
              </span>
            </div>
            <div style={{ fontSize: 24, fontWeight: 700, color: "var(--text-3)", marginTop: 4 }}>—</div>
            <div style={{ fontSize: 11, color: "var(--text-3)", marginTop: 4 }}>
              {t("compare.recall.artifact_unavailable")}
            </div>
          </div>
          <div style={{ padding: "12px 14px", background: "var(--surface-2)", borderRadius: 6, opacity: 0.55 }}>
            <div style={{ fontSize: 10, color: "var(--text-3)", fontWeight: 700, textTransform: "uppercase" }}>{t("compare.f1")}</div>
            <div style={{ fontSize: 24, fontWeight: 700, color: "var(--text-3)", marginTop: 4 }}>—</div>
            <div style={{ fontSize: 11, color: "var(--text-3)", marginTop: 4 }}>
              {t("compare.f1.artifact_unavailable")}
            </div>
          </div>
        </div>
      </section>

      {/* Maps */}
      {view === "overlay" ? (
        <div>
          <div style={{ display: "flex", gap: 12, marginBottom: 6, fontSize: 11, flexWrap: "wrap" }}>
            <span><span style={{ display: "inline-block", width: 12, height: 12, borderRadius: 6, background: PALETTE.hit, verticalAlign: "middle", marginRight: 4 }}/>{t("compare.legend.hit")}</span>
            <span><span style={{ display: "inline-block", width: 12, height: 12, borderRadius: 6, background: PALETTE.alarm, verticalAlign: "middle", marginRight: 4 }}/>{t("compare.legend.alarm")}</span>
            <span><span style={{ display: "inline-block", width: 12, height: 12, borderRadius: 6, background: PALETTE.observed, verticalAlign: "middle", marginRight: 4 }}/>{t("compare.legend.observed")}</span>
            <span><span style={{ display: "inline-block", width: 12, height: 12, borderRadius: 6, background: PALETTE.future, verticalAlign: "middle", marginRight: 4 }}/>{t("compare.legend.future")}</span>
            {matched.rows.some((row) => row.status === "unknown") && (
              <span><span style={{ display: "inline-block", width: 12, height: 12, borderRadius: 6, background: PALETTE.future, verticalAlign: "middle", marginRight: 4 }}/>{t("compare.legend.unknown")}</span>
            )}
          </div>
          <div ref={overlayMapRef} style={{ height: 520, borderRadius: 8, overflow: "hidden", border: "1px solid var(--border)" }} />
        </div>
      ) : (
        <div className="compare-map-split">
          <div>
            <h3 style={{ fontSize: 12, fontWeight: 700, marginBottom: 6 }}>{t("compare.map.predicted")}</h3>
            <div ref={leftMapRef} style={{ height: 520, borderRadius: 8, overflow: "hidden", border: "1px solid var(--border)" }} />
          </div>
          <div>
            <h3 style={{ fontSize: 12, fontWeight: 700, marginBottom: 6 }}>{t("compare.map.actual")}</h3>
            <div ref={rightMapRef} style={{ height: 520, borderRadius: 8, overflow: "hidden", border: "1px solid var(--border)" }} />
          </div>
        </div>
      )}

      <p className="report-section-hint" style={{ marginTop: 12, fontSize: 11 }}>
        {t("compare.audit.line")}
      </p>
      {stats.hits + stats.falseAlarms === 0 && stats.future > 0 && (
        <div style={{
          marginTop: 12, padding: "10px 14px",
          background: "rgba(234, 179, 8, 0.10)",
          border: "1px solid rgba(234, 179, 8, 0.35)",
          borderRadius: 6, fontSize: 12, color: "var(--text-2)",
          lineHeight: 1.55,
        }}>
          {fmtTr(t("compare.pending.warn"), { date: resolvedDate ?? "" })}
        </div>
      )}
      {stats.hits + stats.falseAlarms + stats.future === 0 && (
        <div style={{
          marginTop: 12, padding: "10px 14px",
          background: "var(--surface-2)",
          border: "1px solid var(--border)",
          borderRadius: 6, fontSize: 12, color: "var(--text-3)",
          lineHeight: 1.55,
        }}>
          {fmtTr(t("compare.no_status"), { cmd: "cd src && python risk_map.py" })}
        </div>
      )}
    </div>
  );
}
