import type { RankingPolicy, ValidationMetrics } from "../types";
import { fmtTr, useLang } from "../utils/i18n";
import MetricCard, { type MetricStatus } from "./MetricCard";

interface Props {
  metrics: ValidationMetrics | null;
  rankingPolicy: RankingPolicy | null;
  onShowDetails: () => void;
}

// Operator-facing performance section — multi-card layout.
//
// The cards show only persisted evidence. Missing monthly diagnostics remain
// visibly unavailable instead of being inferred from another metric.
function auc2Grade(auc?: number): { grade: string; status: MetricStatus } {
  if (typeof auc !== "number" || !isFinite(auc)) return { grade: "—", status: "info" };
  if (auc >= 0.90) return { grade: "A",  status: "great" };
  if (auc >= 0.83) return { grade: "A-", status: "great" };
  if (auc >= 0.77) return { grade: "B+", status: "good" };
  if (auc >= 0.70) return { grade: "B",  status: "ok" };
  if (auc >= 0.65) return { grade: "C",  status: "warn" };
  return { grade: "D", status: "bad" };
}

function eceStatus(ece: number | undefined, t: (k: string, fb?: string) => string): { label: string; status: MetricStatus } {
  if (typeof ece !== "number" || !isFinite(ece)) return { label: "—", status: "info" };
  if (ece < 0.05) return { label: t("hero.card.calib.status.great"), status: "great" };
  if (ece < 0.10) return { label: t("hero.card.calib.status.good"),  status: "good"  };
  if (ece < 0.15) return { label: t("hero.card.calib.status.ok"),    status: "ok"    };
  return { label: t("hero.card.calib.status.bad"), status: "bad" };
}

function stabilityStatus(auc?: number, std?: number): MetricStatus {
  if (typeof auc !== "number" || !isFinite(auc)) return "info";
  if (auc >= 0.85 && (std ?? 1) < 0.10) return "great";
  if (auc >= 0.75) return "good";
  if (auc >= 0.65) return "ok";
  return "warn";
}

export default function AccuracyHero({
  metrics,
  rankingPolicy,
  onShowDetails,
}: Props) {
  const { t } = useLang();
  const isBinary = Boolean(
    metrics?.task === "binary_fire_in_3d" ||
    typeof metrics?.roc_auc === "number" ||
    typeof metrics?.deployment_threshold === "number",
  );
  const hasLegacyRegressionMetrics = Boolean(
    typeof metrics?.accuracy_within_1day === "number" ||
    typeof metrics?.mae_days === "number",
  );

  // Old exports did not carry the task discriminator. Show their date-error
  // metrics only when those values are actually present; an absent metrics
  // payload must not masquerade as a legacy ±1-day model.
  if (!isBinary) {
    if (!hasLegacyRegressionMetrics) {
      return (
        <div className="accuracy-hero">
          <div className="accuracy-hero-top">
            <div className="accuracy-hero-label">
              {t("hero.metrics.unavailable.label")}
            </div>
            <button className="info-btn" type="button" onClick={onShowDetails}>ⓘ</button>
          </div>
          <div className="accuracy-hero-value">—</div>
          <div className="accuracy-hero-sub">
            {t("hero.metrics.unavailable.subtitle")}
          </div>
        </div>
      );
    }
    const acc = metrics?.accuracy_within_1day;
    const mae = metrics?.mae_days;
    return (
      <div className="accuracy-hero">
        <div className="accuracy-hero-top">
          <div className="accuracy-hero-label">{t("hero.legacy.label")}</div>
          <button className="info-btn" type="button" onClick={onShowDetails}>ⓘ</button>
        </div>
        <div className="accuracy-hero-value">
          {typeof acc === "number" ? `${(acc * 100).toFixed(1)}%` : "—"}
        </div>
        <div className="accuracy-hero-sub">
          {fmtTr(t("hero.legacy.subtitle"), {
            mae: typeof mae === "number" ? mae.toFixed(2) : "—",
          })}
        </div>
      </div>
    );
  }

  // ── Compute display values ──
  const auc = metrics?.roc_auc;
  const ece = metrics?.ece;
  const upliftTop20 = metrics?.uplift_at_top_20pct;
  const precisionTop20 = metrics?.precision_at_top_20pct;
  const posRate = metrics?.test_positive_rate;
  const deployRecall = metrics?.deployment_recall;
  const deployThr = metrics?.deployment_threshold;
  const watchRecall = metrics?.watch_test_metrics?.recall;
  const watchThr = metrics?.watch_threshold ?? metrics?.watch_test_metrics?.threshold;
  // Only replace the HIGH headline when WATCH is meaningfully better on the
  // same held-out test distribution. Tiny numerical differences should not
  // silently change the operator-facing operating point.
  const useWatchRecall =
    typeof watchRecall === "number" &&
    isFinite(watchRecall) &&
    (
      typeof deployRecall !== "number" ||
      !isFinite(deployRecall) ||
      watchRecall >= deployRecall + 0.05
    );
  const headlineRecall = useWatchRecall ? watchRecall : deployRecall;
  const headlineThr = useWatchRecall ? watchThr : deployThr;
  const top20Coverage = rankingPolicy?.historical_coverage?.top_20pct;
  const hasTop20Coverage =
    typeof top20Coverage === "number" && isFinite(top20Coverage);
  const driftLimitsWatchRecall =
    useWatchRecall && metrics?.prevalence_drift?.drift_flag === true;

  const stabilityMean = metrics?.stability_auc_mean;
  const stabilityStd = metrics?.stability_auc_std;
  const stabilityMin = metrics?.stability_auc_min;
  const stabilityMax = metrics?.stability_auc_max;
  const stabilityMonths = metrics?.stability_valid_months ?? metrics?.stability_months;
  const hasMonthlyDiagnostic =
    typeof stabilityMean === "number" &&
    isFinite(stabilityMean) &&
    typeof stabilityMonths === "number" &&
    stabilityMonths > 0;
  const hasAuc = typeof auc === "number" && isFinite(auc);
  const hasEce = typeof ece === "number" && isFinite(ece);
  const hasUplift =
    typeof upliftTop20 === "number" && isFinite(upliftTop20);
  const hasHeadlineRecall =
    typeof headlineRecall === "number" && isFinite(headlineRecall);

  // Older artifacts omit calibration_method from the persisted metrics but
  // still record that the threshold was locked on the calib split — which in
  // this pipeline always implies Platt calibration fitted there.
  const calibratedAt = !!(
    metrics?.calibration_method || metrics?.threshold_selected_on === "calib"
  );
  const { grade, status: gradeStatus } = auc2Grade(auc);
  const { label: eceLabel, status: eceStat } = eceStatus(ece, t);
  const stStat = hasMonthlyDiagnostic
    ? stabilityStatus(stabilityMean, stabilityStd)
    : "info";
  const recallStatus: MetricStatus = driftLimitsWatchRecall
    ? "info"
    : !hasHeadlineRecall ? "info"
    : headlineRecall >= 0.80 ? "great"
    : headlineRecall >= 0.60 ? "good"
    : headlineRecall >= 0.40 ? "ok" : "warn";

  const fmtPct = (v?: number, d = 0) =>
    typeof v === "number" && isFinite(v) ? `${(v * 100).toFixed(d)}%` : "—";
  const fmt = (v?: number, d = 2) =>
    typeof v === "number" && isFinite(v) ? v.toFixed(d) : "—";

  return (
    <>
      {/* Header with title + ⓘ */}
      <div style={{ margin: "12px 16px 0", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div style={{ fontSize: 11, fontWeight: 600, letterSpacing: "0.08em", textTransform: "uppercase", color: "var(--text-3)" }}>
          {t("hero.title")}
        </div>
        <button className="info-btn" type="button" onClick={onShowDetails} title={t("hero.info.tooltip")}>
          ⓘ
        </button>
      </div>

      {/* Status badges — quick trust signals */}
      <div className="status-row">
        <span className={`status-badge ${stStat}`}>
          <span className="status-badge-icon">⚡</span>
          {hasMonthlyDiagnostic
            ? fmtTr(t("hero.badge.stable"), { months: stabilityMonths })
            : t("hero.badge.stable.placeholder")}
        </span>
        <span className={`status-badge ${eceStat}`}>
          <span className="status-badge-icon">✓</span>
          {calibratedAt ? t("hero.badge.calibrated") : t("hero.badge.uncalibrated")}
        </span>
        <span className={`status-badge ${gradeStatus}`}>
          <span className="status-badge-icon">⭐</span>
          {fmtTr(t("hero.badge.grade"), { grade })}
        </span>
      </div>

      {/* 2 HERO metrics + collapsible expander for the remaining 3 */}
      <div className="metrics-stack">
        <div className="metrics-grid">
          <MetricCard
            highlight
            label={t(useWatchRecall ? "hero.card.watchRecall.label" : "hero.card.recall.label")}
            value={fmtPct(headlineRecall, 0)}
            status={recallStatus}
            statusLabel={
              driftLimitsWatchRecall
                ? t("hero.card.watchRecall.status.drift")
                : undefined
            }
            subtitle={fmtTr(t(
              useWatchRecall
                ? "hero.card.watchRecall.subtitle"
                : "hero.card.recall.subtitle"
            ), {
              pct: fmtPct(headlineRecall, 0),
              thr: typeof headlineThr === "number" ? headlineThr.toFixed(4) : "—",
              highPct: fmtPct(deployRecall, 0),
            })}
            secondaryStat={
              hasTop20Coverage
                ? fmtTr(t("hero.card.rankingCoverage.top20"), {
                    pct: fmtPct(top20Coverage, 0),
                  })
                : undefined
            }
            range={
              hasHeadlineRecall
                ? {
                    min: 0, max: 1, current: headlineRecall,
                    markers: [
                      { position: 0.4, label: t("hero.card.recall.marker.ok"),    color: "#eab308" },
                      { position: 0.6, label: t("hero.card.recall.marker.good"),  color: "#84cc16" },
                      { position: 0.8, label: t("hero.card.recall.marker.great"), color: "#22c55e" },
                    ],
                  }
                : undefined
            }
            description={t(
              useWatchRecall
                ? "hero.card.watchRecall.description"
                : "hero.card.recall.description"
            )}
            goodRange={t(
              useWatchRecall
                ? "hero.card.watchRecall.range"
                : "hero.card.recall.range"
            )}
          />

          <MetricCard
            highlight
            label={t("hero.card.auc.label")}
            value={fmt(auc, 3)}
            statusLabel={grade}
            status={gradeStatus}
            subtitle={fmtTr(t("hero.card.auc.subtitle"), { grade })}
            range={
              hasAuc
                ? {
                    min: 0.5, max: 1.0, current: auc,
                    markers: [
                      { position: 0.3, label: "C 0.65",  color: "#f59e0b" },
                      { position: 0.4, label: "B 0.70",  color: "#eab308" },
                      { position: 0.66, label: "A- 0.83", color: "#22c55e" },
                      { position: 0.8, label: "A 0.90",  color: "#16a34a" },
                    ],
                  }
                : undefined
            }
            description={t("hero.card.auc.description")}
            goodRange={t("hero.card.auc.range")}
          />
        </div>

        {/* Secondary metrics in collapsible expander */}
        <details className="metrics-expander">
          <summary>
            <span>{t("hero.expander.title")}</span>
            <span className="metrics-expander-hint">{t("hero.expander.hint")}</span>
          </summary>
          <div className="metrics-grid" style={{ marginTop: 10 }}>
            <MetricCard
              label={t("hero.card.stability.label")}
              value={hasMonthlyDiagnostic ? stabilityMean.toFixed(3) : "—"}
              subtitle={
                stabilityMonths != null && typeof stabilityMin === "number" && typeof stabilityMax === "number"
                  ? fmtTr(t("hero.card.stability.subtitle.range"), {
                      months: stabilityMonths,
                      min: stabilityMin.toFixed(2),
                      max: stabilityMax.toFixed(2),
                    })
                  : t("hero.card.stability.subtitle.fallback")
              }
              statusLabel={
                !hasMonthlyDiagnostic ? t("hero.card.stability.status.unavailable")
                : stStat === "great" ? t("hero.card.stability.status.great")
                : stStat === "good" ? t("hero.card.stability.status.good")
                : t("hero.card.stability.status.ok")
              }
              status={stStat}
              range={
                hasMonthlyDiagnostic
                  ? {
                      min: 0.5, max: 1.0, current: stabilityMean,
                      markers: [
                        { position: 0.3, label: "0.65", color: "#eab308" },
                        { position: 0.5, label: "0.75", color: "#84cc16" },
                        { position: 0.7, label: "0.85", color: "#22c55e" },
                      ],
                    }
                  : undefined
              }
              description={t("hero.card.stability.description")}
              goodRange={t("hero.card.stability.range")}
            />

            <MetricCard
              label={t("hero.card.calib.label")}
              value={fmt(ece, 4)}
              statusLabel={eceLabel}
              status={eceStat}
              subtitle={t("hero.card.calib.subtitle")}
              range={
                hasEce
                  ? {
                      min: 0, max: 0.3, current: ece,
                      markers: [
                        { position: 0.05 / 0.3, label: "<0.05", color: "#22c55e" },
                        { position: 0.10 / 0.3, label: "<0.10",  color: "#84cc16" },
                        { position: 0.15 / 0.3, label: "<0.15",    color: "#eab308" },
                      ],
                    }
                  : undefined
              }
              description={t("hero.card.calib.description")}
              goodRange={t("hero.card.calib.range")}
            />

            <MetricCard
              label={t("hero.card.uplift.label")}
              value={typeof upliftTop20 === "number" ? `${upliftTop20.toFixed(1)}×` : "—"}
              status={!hasUplift ? "info" : upliftTop20 >= 3 ? "great" : upliftTop20 >= 2 ? "good" : upliftTop20 >= 1.5 ? "ok" : "warn"}
              subtitle={
                typeof precisionTop20 === "number" && typeof posRate === "number"
                  ? fmtTr(t("hero.card.uplift.subtitle"), {
                      hit: Math.round(precisionTop20 * 1000),
                      random: Math.round(posRate * 1000),
                    })
                  : undefined
              }
              range={
                hasUplift
                  ? {
                      min: 1, max: 6, current: upliftTop20,
                      markers: [
                        { position: 0.2,  label: "2×", color: "#eab308" },
                        { position: 0.4,  label: "3×", color: "#84cc16" },
                        { position: 0.6,  label: "4×", color: "#22c55e" },
                      ],
                    }
                  : undefined
              }
              description={t("hero.card.uplift.description")}
              goodRange={t("hero.card.uplift.range")}
            />
          </div>
          <p style={{ fontSize: 11, color: "var(--text-3)", marginTop: 8, lineHeight: 1.5 }}>
            {t("hero.expander.more")}{" "}
            <a href="#reports" style={{ color: "var(--accent)" }}>{t("hero.expander.reports_link")}</a>
          </p>
        </details>
      </div>
    </>
  );
}
