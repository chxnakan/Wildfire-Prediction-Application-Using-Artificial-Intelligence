import type {
  DaySelection,
  DisplayOptions,
  FireFeature,
  GeoJsonMetadata,
  LiveFireMeta,
  UrgencyLevel,
  UrgencyThresholds,
  ValidationMetrics,
} from "../types";
import { calendarDayLag, computeFreshness, formatDataTime } from "../utils/dates";
import { MIN_COMPARE_RATE_SAMPLE } from "../utils/compareStats";
import { fmtTr, useLang } from "../utils/i18n";
import { resolveTierMode } from "../utils/risk";
import AccuracyHero from "./AccuracyHero";
import ActionToolbar from "./ActionToolbar";
import DataSourcesPanel from "./DataSourcesPanel";
import RetrospectivePanel from "./RetrospectivePanel";

interface SidebarProps {
  // Header / base date
  activeBaseDate: string;
  allBaseDates: string[];
  selectedBaseDate: string; // "latest" or an ISO date
  onBaseDateChange: (v: string) => void;

  // Province
  provinces: string[];
  selectedProvince: string;
  onProvinceChange: (v: string) => void;

  // Day selector
  selectedDay: DaySelection;
  onDayChange: (d: DaySelection) => void;
  predicted: FireFeature[]; // for the snapshot — used by timeline + landcover + urgency
  visibleCount: number;
  hiddenVeryLowCount: number;
  daySelectorMessage: string;

  // Urgency / metrics / hit-rate
  thresholds: UrgencyThresholds | null;
  metrics: ValidationMetrics | null;
  metadata: GeoJsonMetadata | null;

  // ALL predicted features across base_dates — for retrospective panel
  predictedAll: FireFeature[];

  // Display options + export
  options: DisplayOptions;
  onOptionsChange: (o: Partial<DisplayOptions>) => void;
  onExportCsv: () => void;

  // Live-fire status (GISTDA)
  liveFireMeta: LiveFireMeta;

  // Info modal
  onShowInfoModal: () => void;
  onShowAlertSettings: () => void;
}

export default function Sidebar(p: SidebarProps) {
  const { lang, t } = useLang();
  const fresh = computeFreshness(p.activeBaseDate);
  // Tier-cutoff semantics, from risk_map.py's thresholds_source:
  //   absolute — calibrated-probability cutoffs selected on the calib split
  //   relative — percentile ranks of this snapshot (legacy artifact fallback)
  //   days     — legacy fixed-domain days cutoffs
  // Probability-style thresholds are 0..1 (CRITICAL uses a >1 "unused"
  // sentinel); days thresholds are 0..7.
  const tierMode = resolveTierMode(
    p.thresholds,
    p.metadata?.thresholds_source,
  );
  const urgencyCounts: Record<UrgencyLevel, number> = {
    CRITICAL: 0, HIGH: 0, MEDIUM: 0, LOW: 0, NONE: 0,
  };
  let forest = 0, mixed = 0, open = 0, landcoverDataCount = 0;
  for (const f of p.predicted) {
    const u = f.properties.urgency_level;
    if (u && Object.prototype.hasOwnProperty.call(urgencyCounts, u)) {
      urgencyCounts[u]++;
    }
    const c = f.properties.tree_cover_pct_2000;
    if (typeof c === "number" && Number.isFinite(c)) {
      landcoverDataCount++;
      if (c >= 50) forest++;
      else if (c >= 10) mixed++;
      else open++;
    }
  }
  const isLowRiskSnapshot =
    tierMode === "absolute" &&
    urgencyCounts.CRITICAL === 0 &&
    urgencyCounts.HIGH === 0 &&
    urgencyCounts.MEDIUM === 0 &&
    urgencyCounts.LOW > 0 &&
    urgencyCounts.NONE === 0;
  const criticalEnabled =
    urgencyCounts.CRITICAL > 0 ||
    tierMode === "days" ||
    (
      typeof p.thresholds?.CRITICAL === "number" &&
      p.thresholds.CRITICAL <= 1
    );

  const fmtRange = (v: number | undefined) => {
    if (v == null || !Number.isFinite(Number(v))) return "—";
    if (tierMode !== "days") {
      // Probability cutoffs; >1 is the "tier unused" sentinel.
      return Number(v) > 1 ? "not used" : `≥ ${(Number(v) * 100).toFixed(2)}%`;
    }
    return `≤ ${Number(v).toFixed(1)} d`;
  };
  const highLegend = fmtTr(t("tier.high.desc"), {
    threshold: fmtRange(p.thresholds?.HIGH),
  });
  const watchRecall = p.metrics?.watch_test_metrics?.recall;
  const watchLegend = fmtTr(t(
    typeof watchRecall === "number" && isFinite(watchRecall)
      ? "tier.watch.desc.with_recall"
      : "tier.watch.desc"
  ), {
    threshold: fmtRange(p.thresholds?.MEDIUM),
    recall:
      typeof watchRecall === "number" && isFinite(watchRecall)
        ? `${(watchRecall * 100).toFixed(1)}%`
        : "",
  });
  const prevalenceDrift = p.metrics?.prevalence_drift;
  const finiteFixed = (
    value: number | null | undefined,
    digits: number,
    multiplier = 1,
  ) =>
    typeof value === "number" && Number.isFinite(value)
      ? (value * multiplier).toFixed(digits)
      : "—";

  return (
    <div id="sidebar">
      <div className="header">
        <h1>{t("sidebar.brand.title")}</h1>
        <p className="subtitle">{t("sidebar.brand.subtitle")}</p>
      </div>

      {/* Layer toggles — top of sidebar for quick access */}
      <LayerPills
        options={p.options}
        liveFireMeta={p.liveFireMeta}
        hiddenVeryLowCount={p.hiddenVeryLowCount}
        onChange={p.onOptionsChange}
      />

      {/* Quick stats — at-a-glance Critical / High / Total */}
      <QuickStats
        urgencyCounts={urgencyCounts}
        visibleCount={p.visibleCount}
        criticalEnabled={criticalEnabled}
      />
      {isLowRiskSnapshot && (
        <div className="low-risk-notice" role="status">
          <span className="low-risk-notice-icon" aria-hidden="true">✓</span>
          <div>
            <strong>{t("sidebar.lowrisk.title")}</strong>
            <span>
              {fmtTr(t("sidebar.lowrisk.body"), {
                date: p.activeBaseDate,
                count: urgencyCounts.LOW,
              })}
            </span>
          </div>
        </div>
      )}

      {/* Base Date */}
      <div className="info-card">
        <div className="info-label">
          Base Date
          {fresh && (
            <span className={`freshness-badge ${fresh.cls}`}>{fresh.label}</span>
          )}
        </div>
        <div className="info-value">{p.activeBaseDate || "Loading..."}</div>
        {fresh && fresh.msg && <div className="info-sub">{fresh.msg}</div>}
        {(() => {
          const dataTime = formatDataTime(p.metadata?.latest_data_time, lang);
          return dataTime ? (
            <div className="info-sub" title={p.metadata?.latest_data_time}>
              🛰 {t("sidebar.datatime.label")}: <strong>{dataTime}</strong>
            </div>
          ) : null;
        })()}

        <div className="date-picker-row">
          <label className="date-picker-label" htmlFor="baseDatePicker">
            View predictions from:
          </label>
          <select
            id="baseDatePicker"
            className="date-picker"
            value={p.selectedBaseDate}
            onChange={(e) => p.onBaseDateChange(e.target.value)}
          >
            <option value="latest">
              Latest ({p.allBaseDates[p.allBaseDates.length - 1] ?? "—"})
            </option>
            {[...p.allBaseDates].reverse().map((d) => {
              const lag = calendarDayLag(d);
              const isLatest =
                d === p.allBaseDates[p.allBaseDates.length - 1];
              const suffix = isLatest
                ? "newest"
                : lag == null
                  ? "invalid date"
                  : lag < 0
                    ? `${Math.abs(lag)} d ahead of today`
                    : `${lag} d behind today`;
              return (
                <option key={d} value={d}>
                  {d} ({suffix})
                </option>
              );
            })}
          </select>
        </div>

        <div className="date-picker-row">
          <label className="date-picker-label" htmlFor="provinceFilter">
            Filter by province:
          </label>
          <select
            id="provinceFilter"
            className="date-picker"
            value={p.selectedProvince}
            onChange={(e) => p.onProvinceChange(e.target.value)}
          >
            <option value="all">
              All available provinces ({p.provinces.length})
            </option>
            {p.provinces.map((name) => (
              <option key={name} value={name}>
                {name}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Action toolbar — production actions: export, share, threshold, details */}
      <ActionToolbar
        exportCount={
          p.selectedDay === "all"
            ? p.predicted.length
            : p.selectedDay === "High"
              ? urgencyCounts.HIGH
              : p.selectedDay === "Watch"
                ? urgencyCounts.MEDIUM
                : urgencyCounts.LOW
        }
        onExport={p.onExportCsv}
        metrics={p.metrics}
        onShowDetails={p.onShowInfoModal}
        onShowAlertSettings={p.onShowAlertSettings}
      />

      {/* Hero Accuracy — multi-card grid; "model is good" metrics first */}
      <AccuracyHero
        metrics={p.metrics}
        rankingPolicy={p.metadata?.ranking_policy ?? null}
        onShowDetails={p.onShowInfoModal}
      />

      {/* Risk-tier filter — replaces the former day-0..7 pseudo-date buckets.
          The binary model has no per-day resolution: it reports a calibrated
          probability of a hotspot within the next 3 days, mapped to tiers. */}
      <div className="section">
        <h3>🎯 {t("sidebar.tierfilter.label")}</h3>
        <div className="day-selector">
          {(["all", "High", "Watch", "Background"] as const).map((tier) => {
            const count =
              tier === "all" ? p.predicted.length
              : tier === "High" ? urgencyCounts.HIGH
              : tier === "Watch" ? urgencyCounts.MEDIUM
              : urgencyCounts.LOW;
            const hasData = tier === "all" || count > 0;
            const label = tier === "all" ? t("sidebar.daypicker.all") : tier;
            return (
              <button
                key={tier}
                className={`day-btn ${p.selectedDay === tier ? "active" : ""}`}
                onClick={() => hasData && p.onDayChange(tier as DaySelection)}
                disabled={!hasData}
                style={{
                  flexDirection: "column",
                  padding: "6px 8px",
                  gap: 1,
                  opacity: hasData ? 1 : 0.35,
                  cursor: hasData ? "pointer" : "default",
                }}
                title={`${label} · ${count}`}
              >
                <span style={{ fontSize: 11, fontWeight: 700 }}>{label}</span>
                <span style={{ fontSize: 9, opacity: 0.7 }}>
                  {hasData ? count : "—"}
                </span>
              </button>
            );
          })}
        </div>
        <div className="day-selector-info">
          {p.daySelectorMessage}
          <br/>
          <span style={{ fontSize: 10, color: "var(--text-3)" }}>
            {t("sidebar.tierfilter.hint")}
          </span>
        </div>
        {/* WATCH favours recall — it deliberately over-alerts to avoid misses. */}
        <div className="tier-legend" style={{ fontSize: 10, color: "var(--text-3)", marginTop: 6, lineHeight: 1.5 }}>
          <div><strong style={{ color: "#ef4444" }}>● {t("tier.high")}</strong> — {highLegend}</div>
          <div><strong style={{ color: "#f59e0b" }}>● {t("tier.watch")}</strong> — {watchLegend}</div>
        </div>
        {/* Base-rate drift warning, when the model flags it in metadata. */}
        {prevalenceDrift?.drift_flag && (
          <div className="drift-warning" style={{
            marginTop: 8, padding: "6px 8px", borderRadius: 6, fontSize: 10,
            background: "rgba(245,158,11,0.12)", border: "1px solid rgba(245,158,11,0.4)",
            color: "var(--text-2)",
          }}>
            ⚠️ {fmtTr(t("drift.warning"), {
              calib: finiteFixed(prevalenceDrift.calib_prevalence, 1, 100),
              recent: finiteFixed(prevalenceDrift.recent_prevalence, 1, 100),
              ratio: finiteFixed(prevalenceDrift.ratio, 1),
            })}
          </div>
        )}
      </div>

      {/* Urgency */}
      <div className="section">
        <h3>⚡ Urgency Summary</h3>
        <div className="urgency-cards">
          {(["critical", "high", "medium", "low"] as const).map((tier) => {
            const key = tier.toUpperCase() as UrgencyLevel;
            const upper = key as keyof UrgencyThresholds;
            const displayLabel =
              tier === "medium" ? "WATCH"
              : tier === "low" ? "BACKGROUND"
              : tier.toUpperCase();
            const tip = (tierMode !== "days" ? {
              critical: "Not produced by the current model — no calibrated CRITICAL cutoff.",
              high: "Calibrated fire probability at or above the deployment alert threshold. Act first.",
              medium: "Probability above the WATCH recall-target cutoff — recall-oriented monitoring list (may include false positives).",
              low: "Below alert cutoffs — background risk, shown for completeness.",
            } : {
              critical: "Fire predicted within 24 hours of the base date. Triage first.",
              high: "Fire predicted within ~2 days. Watch closely.",
              medium: "Fire predicted within ~4 days. Plan ahead.",
              low: "Fire predicted within the full 7-day horizon. Background risk.",
            })[tier];
            return (
              <div key={tier} className={`urgency-card ${tier}`} title={tip}>
                <div className="urgency-label">{displayLabel}</div>
                <div className="urgency-count">{urgencyCounts[key]}</div>
                <div className="urgency-desc">
                  {fmtRange(p.thresholds?.[upper])}
                </div>
              </div>
            );
          })}
        </div>
        <div className="threshold-note">
          {p.thresholds == null
            ? "Threshold metadata is unavailable. Cells keep their persisted tiers, but numeric cutoffs are not shown."
            : tierMode === "absolute" ? (
              <>
                <strong>Absolute risk tiers:</strong> calibrated-probability
                cutoffs selected on the calibration split at training time —
                HIGH = deployment threshold; WATCH (= MEDIUM alias) = recall-target cutoff.
                Zero HIGH cells on a genuinely low-risk day is expected.
              </>
            ) : tierMode === "relative" ? (
              <>
                <strong>Priority ranking:</strong> percentile ranks of this
                snapshot only (legacy model artifact without calibrated
                cutoffs) — relative ordering, not absolute risk.
              </>
            ) : (
              "Fixed-domain cutoffs (CRITICAL=0d, HIGH≤2d, MEDIUM≤4d, LOW≤7d)."
            )}
        </div>
      </div>

      {/* Land cover + export */}
      <div className="section">
        <h3>🌳 Land Cover (Hansen GFC)</h3>
        <div className="landcover-grid">
          <div className="landcover-row">
            <span className="landcover-bar forest" />
            <span className="landcover-label">Forest (≥50%)</span>
            <span className="landcover-count">
              {landcoverDataCount > 0 ? forest : "—"}
            </span>
          </div>
          <div className="landcover-row">
            <span className="landcover-bar mixed" />
            <span className="landcover-label">Mixed (10–50%)</span>
            <span className="landcover-count">
              {landcoverDataCount > 0 ? mixed : "—"}
            </span>
          </div>
          <div className="landcover-row">
            <span className="landcover-bar open" />
            <span className="landcover-label">Open / Grassland (&lt;10%)</span>
            <span className="landcover-count">
              {landcoverDataCount > 0 ? open : "—"}
            </span>
          </div>
        </div>
        <div className="landcover-note">
          Tree-cover bucket per cell from Hansen GFC v1.11. This is descriptive
          context only; it does not identify a fire's cause or distinguish
          agricultural burning from wildfire.
        </div>
        <button className="export-btn" onClick={p.onExportCsv}>
          ⬇ Export selected cells (CSV)
        </button>
      </div>

      {/* Hit-rate */}
      <HitRate metadata={p.metadata} activeBaseDate={p.activeBaseDate} />

      {/* Past predictions retrospective — strongest trust signal */}
      <RetrospectivePanel allFeatures={p.predictedAll} />

      {/* Data sources / provenance */}
      <DataSourcesPanel />

      <div className="info-footer">
        <small style={{ display: "block", lineHeight: 1.6 }}>
          <b>🔥 FireWatch Thailand · research preview</b>
          <br />
          Open-source · MIT License
          <br />
          <a
            href="https://github.com/0Maruz/Wildfire-Prediction-Project"
            target="_blank"
            rel="noopener noreferrer"
            style={{ color: "var(--accent)" }}
          >
            🐙 GitHub source
          </a>
          {" · "}
          <a
            href="https://github.com/0Maruz/Wildfire-Prediction-Project/blob/main/docs/MODEL_CARD.md"
            target="_blank"
            rel="noopener noreferrer"
            style={{ color: "var(--accent)" }}
          >
            📋 Model Card
          </a>
          {" · "}
          <a
            href="https://github.com/0Maruz/Wildfire-Prediction-Project/blob/main/docs/METHODOLOGY.md"
            target="_blank"
            rel="noopener noreferrer"
            style={{ color: "var(--accent)" }}
          >
            📐 Methodology
          </a>
          <br />
          <span style={{ color: "var(--text-3)", fontSize: 10 }}>
            Data provenance is listed above · no fabricated live observations
            {p.metrics?.calibration_method
              ? ` · Calibration: ${p.metrics.calibration_method}`
              : " · Calibration metadata unavailable"}
            <br />
            ⚠ Research-grade — not for life-safety decisions
          </span>
        </small>
      </div>
    </div>
  );
}

// ───────────────────── HitRate sub-component ─────────────────────

function HitRate({
  metadata,
  activeBaseDate,
}: {
  metadata: GeoJsonMetadata | null;
  activeBaseDate: string;
}) {
  const validation = metadata?.validation_summary;
  const perSnap = validation?.per_snapshot ?? {};
  const bucket = perSnap[activeBaseDate];
  const tiers = (validation?.operational_tiers ?? [])
    .map((tier) => tier === "MEDIUM" ? "WATCH" : tier)
    .join("+");
  const tierLabel = tiers || "operational alert tiers";
  const matchRule = validation?.match_rule?.trim() || null;
  const horizonDays = validation?.horizon_days;
  const methodText =
    matchRule ??
    (typeof horizonDays === "number" && Number.isFinite(horizonDays)
      ? `outcome checked within the ${horizonDays}-day model window`
      : "validation method was not recorded in this snapshot");

  let value = "—";
  let suffix = "no validation record";
  let detail = `${activeBaseDate} · ${tierLabel}`;

  if (!bucket) {
    detail = "This snapshot has no retained validation record.";
  } else {
    const hits = bucket.hits ?? 0;
    const misses = bucket.misses ?? 0;
    const future = bucket.future ?? 0;
    const validatable = hits + misses;
    if (validatable > 0) {
      const pct = (hits / validatable) * 100;
      const enoughForRate = validatable >= MIN_COMPARE_RATE_SAMPLE;
      value = enoughForRate ? `${pct.toFixed(0)}%` : "—";
      suffix = enoughForRate
        ? `${hits} / ${validatable} closed alerts matched`
        : `${hits} / ${validatable} matched · rate withheld (n < ${MIN_COMPARE_RATE_SAMPLE})`;
      detail =
        `${activeBaseDate} · ${tierLabel} · ${methodText}` +
        (future > 0 ? ` · ${future} outcome${future === 1 ? "" : "s"} pending` : "");
    } else if (future > 0) {
      suffix = `${future} pending`;
      const windowLabel =
        typeof horizonDays === "number" && Number.isFinite(horizonDays)
          ? `${horizonDays}-day`
          : "model";
      detail =
        `The ${future === 1 ? "alert's" : "alerts'"} ${windowLabel} ` +
        "outcome window has not closed yet.";
    } else {
      suffix = "no operational alerts";
      detail = `No ${tierLabel} cells were issued for this snapshot.`;
    }
  }

  return (
    <div className="section">
      <h3>🎯 Hit Rate vs FIRMS</h3>
      <div
        className="hitrate-card"
        title={`Of archived ${tierLabel} alerts whose outcome window closed, what fraction matched a real FIRMS detection under the persisted rule: ${methodText}. Background cells are excluded from this alert-level rate.`}
      >
        <div className="hitrate-headline">
          <span id="hitrateValue">{value}</span>
          <span className="hitrate-suffix">{suffix}</span>
        </div>
        <div className="hitrate-detail">{detail}</div>
      </div>
      <div className="metrics-note">
        Archived {tierLabel} alerts are checked against FIRMS using the
        persisted validation rule: {methodText}. This is an alert hit rate,
        not held-out model recall. Background cells remain available in the
        Predicted vs Actual audit.
      </div>
    </div>
  );
}

// ───────────────────── Layer pill toggles ─────────────────────

function LayerPills({
  options,
  liveFireMeta,
  hiddenVeryLowCount,
  onChange,
}: {
  options: DisplayOptions;
  liveFireMeta: LiveFireMeta;
  hiddenVeryLowCount: number;
  onChange: (o: Partial<DisplayOptions>) => void;
}) {
  const { t } = useLang();
  const liveCount =
    liveFireMeta.status === "idle" ? null : liveFireMeta.count;
  const liveError = liveFireMeta.status === "error";
  const lastSuccess = liveFireMeta.lastSuccessAt
    ? new Date(liveFireMeta.lastSuccessAt)
    : null;
  const lastSuccessLabel =
    lastSuccess && Number.isFinite(lastSuccess.getTime())
      ? lastSuccess.toLocaleString()
      : "—";

  return (
    <div className="layer-pills">
      <button
        type="button"
        className={`layer-pill predicted${options.showPredicted ? " active" : ""}`}
        onClick={() => onChange({ showPredicted: !options.showPredicted })}
        aria-pressed={options.showPredicted}
        title={t("sidebar.display.predicted")}
      >
        <span className="layer-pill-dot" style={{ background: "#f97316" }} />
        {t("sidebar.display.predicted")}
      </button>

      <button
        type="button"
        className={`layer-pill observed${options.showObserved ? " active" : ""}`}
        onClick={() => onChange({ showObserved: !options.showObserved })}
        aria-pressed={options.showObserved}
        title={t("sidebar.display.observed")}
      >
        <span className="layer-pill-dot" style={{ background: "#ff6b35" }} />
        {t("sidebar.display.observed")}
      </button>

      <button
        type="button"
        className={`layer-pill live${options.showLiveFires ? " active" : ""}${liveError ? " error-state" : ""}`}
        onClick={() => onChange({ showLiveFires: !options.showLiveFires })}
        aria-pressed={options.showLiveFires}
        aria-busy={liveFireMeta.status === "loading"}
        title={
          liveFireMeta.status === "error"
            ? fmtTr(t("sidebar.live.error"), { msg: liveFireMeta.error ?? "network error" })
            : liveFireMeta.isStale
              ? fmtTr(t("sidebar.live.stale"), {
                  when: lastSuccessLabel,
                })
              : t("sidebar.display.live")
        }
      >
        <span className="layer-pill-dot" style={{ background: "#06b6d4" }} />
        {t("sidebar.display.live")}
        {liveCount !== null && liveCount > 0 && (
          <span className="layer-pill-count">· {liveCount}</span>
        )}
        {liveFireMeta.isStale && liveFireMeta.status !== "error" && (
          <span className="layer-pill-count" title={t("sidebar.live.stale_short")}>⚠</span>
        )}
      </button>

      <button
        type="button"
        className={`layer-pill very-low${options.hideVeryLowRisk ? " active" : ""}`}
        onClick={() => onChange({ hideVeryLowRisk: !options.hideVeryLowRisk })}
        aria-pressed={options.hideVeryLowRisk}
        title={t("sidebar.display.hide_very_low_hint")}
      >
        <span className="layer-pill-dot" style={{ background: "#94a3b8" }} />
        {t("sidebar.display.hide_very_low")}
        {options.hideVeryLowRisk && (
          <span className="layer-pill-count">· {hiddenVeryLowCount}</span>
        )}
      </button>

    </div>
  );
}

// ───────────────────── Quick stats strip ─────────────────────

function QuickStats({
  urgencyCounts,
  visibleCount,
  criticalEnabled,
}: {
  urgencyCounts: Record<UrgencyLevel, number>;
  visibleCount: number;
  criticalEnabled: boolean;
}) {
  return (
    <div className="quick-stats-strip">
      <div
        className="quick-stat critical"
        title={
          criticalEnabled
            ? "CRITICAL tier — highest urgency"
            : "No calibrated CRITICAL cutoff in this model"
        }
      >
        <div className="quick-stat-value">
          {criticalEnabled ? urgencyCounts.CRITICAL : "—"}
        </div>
        <div className="quick-stat-label">Critical</div>
      </div>
      <div className="quick-stat high" title="HIGH tier — calibrated fire probability above the alert threshold">
        <div className="quick-stat-value">{urgencyCounts.HIGH}</div>
        <div className="quick-stat-label">High</div>
      </div>
      <div className="quick-stat medium" title="WATCH tier — probability above the recall-target watch cutoff">
        <div className="quick-stat-value">{urgencyCounts.MEDIUM}</div>
        <div className="quick-stat-label">Watch</div>
      </div>
      <div className="quick-stat total" title="Total visible predictions">
        <div className="quick-stat-value">{visibleCount}</div>
        <div className="quick-stat-label">Visible</div>
      </div>
    </div>
  );
}
