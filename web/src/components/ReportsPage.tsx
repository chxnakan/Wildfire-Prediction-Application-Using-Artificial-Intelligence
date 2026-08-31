import { useEffect, useMemo, useRef, useState } from "react";
import { fetchRollingEval } from "../api";
import {
  Bar, BarChart, CartesianGrid, Cell, LabelList, Legend, Line, LineChart,
  PolarAngleAxis, PolarGrid, PolarRadiusAxis, Radar, RadarChart,
  ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import type { FireFeature, RollingMonthPoint, ValidationMetrics } from "../types";
import { chartFilename, downloadChartPng } from "../utils/chartExport";
import { downloadCsv, downloadMultiSectionCsv } from "../utils/csvExport";
import { fmtTr, useLang } from "../utils/i18n";
import { scientificSampleSplits } from "../utils/scientificStats";
import ModelTrainingSummary from "./ModelTrainingSummary";
import StatisticsSection from "./StatisticsSection";

type ChartShape = "bar" | "line";

function ChartToolbar({
  containerRef, fileStem, getCsvRows, shape, onShapeChange,
}: {
  containerRef: React.RefObject<HTMLDivElement>;
  fileStem: string;
  getCsvRows?: () => (string | number | null | undefined)[][];
  shape?: ChartShape;
  onShapeChange?: (s: ChartShape) => void;
}) {
  return (
    <div className="chart-toolbar">
      {shape && onShapeChange && (
        <div className="chart-shape-toggle">
          <button
            type="button"
            className={shape === "bar" ? "active" : ""}
            onClick={() => onShapeChange("bar")}
            title="Bar chart"
          >📊 Bar</button>
          <button
            type="button"
            className={shape === "line" ? "active" : ""}
            onClick={() => onShapeChange("line")}
            title="Line chart"
          >📈 Line</button>
        </div>
      )}
      <button
        type="button"
        className="action-btn"
        onClick={() => {
          downloadChartPng(containerRef.current, chartFilename(fileStem, "png"))
            .catch((e) => console.error("PNG export failed:", e));
        }}
        title="Download chart as PNG"
        style={{ padding: "4px 10px", fontSize: 11 }}
      >🖼️ PNG</button>
      {getCsvRows && (
        <button
          type="button"
          className="action-btn"
          onClick={() => downloadCsv(chartFilename(fileStem, "csv"), getCsvRows())}
          title="Download underlying data as CSV"
          style={{ padding: "4px 10px", fontSize: 11 }}
        >📥 CSV</button>
      )}
    </div>
  );
}

function SmallExportBtn({ filename, getRows }: {
  filename: string;
  getRows: () => (string | number | null | undefined)[][];
}) {
  return (
    <button
      type="button"
      className="action-btn"
      onClick={() => downloadCsv(filename, getRows())}
      style={{ padding: "4px 10px", fontSize: 11 }}
      title={`Export ${filename}`}
    >
      📥 CSV
    </button>
  );
}

interface Props {
  metrics: ValidationMetrics | null;
  predictedAll: FireFeature[];
}

// ─────────────────────────────────────────────────────────────
// Reports page — model performance + dataset breakdown
//
// Performance, threshold, risk-tier, monthly diagnostic, feature-importance,
// and province sections. Most data comes from persisted GeoJSON metadata; the
// monthly descriptive diagnostic has an explicit API fallback and error state.
// ─────────────────────────────────────────────────────────────

const TIER_COLORS: Record<string, string> = {
  CRITICAL: "#ef4444",
  HIGH:     "#f97316",
  MEDIUM:   "#eab308",
  LOW:      "#22c55e",
};

export default function ReportsPage({ metrics, predictedAll }: Props) {
  const { t } = useLang();
  // ─── Rolling AUC chart data ───
  // Prefer the per-month series baked into geojson metadata. If absent
  // (rolling_eval.py wasn't run before risk_map.py), fall back to fetching
  // /api/rolling-eval directly so the dashboard reflects the latest file
  // on disk without requiring a re-run of risk_map.
  const inlineMonthly = metrics?.rolling_by_month ?? [];
  const [fetchedMonthly, setFetchedMonthly] = useState<RollingMonthPoint[]>([]);
  const [rollingLoadState, setRollingLoadState] = useState<
    "idle" | "loading" | "loaded" | "error"
  >(inlineMonthly.length > 0 ? "loaded" : "idle");
  const [rollingRetryKey, setRollingRetryKey] = useState(0);
  useEffect(() => {
    if (inlineMonthly.length > 0) {
      setRollingLoadState("loaded");
      return;
    }
    let cancelled = false;
    setRollingLoadState("loading");
    fetchRollingEval()
      .then((r) => {
        if (cancelled) return;
        setFetchedMonthly(r.months ?? []);
        setRollingLoadState("loaded");
      })
      .catch(() => {
        if (!cancelled) setRollingLoadState("error");
      });
    return () => { cancelled = true; };
  }, [inlineMonthly.length, rollingRetryKey]);
  const monthly = inlineMonthly.length > 0 ? inlineMonthly : fetchedMonthly;

  // ─── Top feature importance ───
  // Read only the persisted importance values; absence stays visible.
  const featureImportance: { feature: string; importance: number }[] = useMemo(() => {
    const arr = metrics?.feature_importance_top;
    if (Array.isArray(arr) && arr.length) return arr.slice(0, 20);
    return [];
  }, [metrics]);

  // ─── Provincial breakdown ───
  // Use the LATEST base_date snapshot in predictedAll so the chart matches
  // the dashboard map. Count cells per province per urgency.
  const provinces = useMemo(() => {
    const byDate = new Map<string, FireFeature[]>();
    for (const f of predictedAll) {
      const bd = f.properties.base_date;
      if (!bd) continue;
      if (!byDate.has(bd)) byDate.set(bd, []);
      byDate.get(bd)!.push(f);
    }
    const dates = Array.from(byDate.keys()).sort();
    const latest = dates[dates.length - 1];
    if (!latest) return [];

    const map = new Map<string, { CRITICAL: number; HIGH: number; MEDIUM: number; LOW: number; total: number }>();
    for (const f of byDate.get(latest)!) {
      const p = f.properties;
      const prov = p.province ?? "—";
      const u = p.urgency_level ?? "NONE";
      let entry = map.get(prov);
      if (!entry) {
        entry = { CRITICAL: 0, HIGH: 0, MEDIUM: 0, LOW: 0, total: 0 };
        map.set(prov, entry);
      }
      if (u === "CRITICAL" || u === "HIGH" || u === "MEDIUM" || u === "LOW") {
        entry[u] += 1;
        entry.total += 1;
      }
    }
    return Array.from(map.entries())
      .map(([prov, c]) => ({ prov, ...c }))
      .sort((a, b) => {
        const operationalA = a.CRITICAL + a.HIGH + a.MEDIUM;
        const operationalB = b.CRITICAL + b.HIGH + b.MEDIUM;
        return operationalB - operationalA || b.total - a.total;
      })
      .slice(0, 15);
  }, [predictedAll]);

  // Chart toolbar refs / shape state — local to ReportsPage; one per chart.
  const daysChartRef = useRef<HTMLDivElement>(null);
  const radarChartRef = useRef<HTMLDivElement>(null);
  const thresholdChartRef = useRef<HTMLDivElement>(null);
  const [daysShape, setDaysShape] = useState<ChartShape>("bar");

  // ─── Distribution by persisted absolute risk tier ───
  const tierDistribution = useMemo(() => {
    const byDate = new Map<string, FireFeature[]>();
    for (const f of predictedAll) {
      const bd = f.properties.base_date;
      if (!bd) continue;
      if (!byDate.has(bd)) byDate.set(bd, []);
      byDate.get(bd)!.push(f);
    }
    const dates = Array.from(byDate.keys()).sort();
    const latest = dates[dates.length - 1];
    if (!latest) return { latest: null, rows: [] as { tier: string; count: number; color: string }[] };
    // Absolute risk-tier distribution (High / Watch / Background) — the binary
    // model has no per-day resolution, so this replaces the former pseudo-day
    // histogram. Counts map from the persisted urgency_level.
    const counts: Record<string, number> = {
      CRITICAL: 0,
      HIGH: 0,
      MEDIUM: 0,
      LOW: 0,
    };
    for (const f of byDate.get(latest)!) {
      const u = f.properties.urgency_level;
      if (u && u in counts) counts[u]++;
    }
    const rows = [
      { tier: "Critical", count: counts.CRITICAL, color: TIER_COLORS.CRITICAL },
      { tier: "High", count: counts.HIGH, color: TIER_COLORS.HIGH },
      { tier: "Watch", count: counts.MEDIUM, color: TIER_COLORS.MEDIUM },
      { tier: "Background", count: counts.LOW, color: TIER_COLORS.LOW },
    ];
    return { latest, rows };
  }, [predictedAll]);

  const sciStats = metrics?.scientific_stats;
  const recallLimitedByDrift = metrics?.prevalence_drift?.drift_flag === true;
  const deploymentThreshold =
    typeof metrics?.deployment_threshold === "number" &&
    Number.isFinite(metrics.deployment_threshold)
      ? metrics.deployment_threshold
      : null;
  const separateFileCount =
    (sciStats ? 6 : 0) +
    (monthly.length > 0 ? 1 : 0) +
    (featureImportance.length > 0 ? 1 : 0) +
    (provinces.length > 0 ? 1 : 0);

  // ── Build the section list (used by both combined + separate downloads) ──
  // Defined as a function so it's only evaluated when the user clicks an
  // export button (and so monthly / featureImportance / provinces are fresh).
  const buildSections = (): { name: string; rows: (string | number | null | undefined)[][] }[] => {
    const sections: { name: string; rows: (string | number | null | undefined)[][] }[] = [];

    sections.push({
      name: "Report Header",
      rows: [
        ["field", "value"],
        ["dashboard", "FireWatch Thailand"],
        ["generated_at", new Date().toISOString()],
        ["url", typeof window !== "undefined" ? window.location.href : ""],
        ["note", "Values exported from the persisted dashboard artifacts. Model outputs are predictions; FIRMS/GISTDA fields are observations."],
      ],
    });

    if (sciStats) {
      const sampleRows = scientificSampleSplits(sciStats.samples);
      sections.push({
        name: "Dataset Split",
        rows: [
          ["split", "n_rows", "positives", "positive_rate", "date_start", "date_end"],
          ...sampleRows.map((row) => [
            row.key,
            row.data.n,
            row.data.positives,
            row.data.positive_rate,
            ...row.data.date_range,
          ]),
          ["total_densified", sciStats.samples.total_densified, "", "", "", ""],
        ],
      });
      sections.push({
        name: "Bootstrap 95% Confidence Intervals (n=1000)",
        rows: [
          ["metric", "point", "ci_lower", "ci_upper", "std", "n_boot", "confidence"],
          ...(Object.entries(sciStats.ci_95) as [string, typeof sciStats.ci_95.roc_auc][]).map(
            ([k, v]) => [k, v.point, v.lower, v.upper, v.std, v.n_boot, v.confidence]
          ),
        ],
      });
      sections.push({
        name: "Confusion Matrix (@deployment threshold)",
        rows: [
          ["", "predicted_no_fire", "predicted_fire"],
          ["actual_no_fire", sciStats.confusion_matrix.tn, sciStats.confusion_matrix.fp],
          ["actual_fire", sciStats.confusion_matrix.fn, sciStats.confusion_matrix.tp],
        ],
      });
      sections.push({
        name: "Classification Statistics",
        rows: [
          ["metric", "value"],
          ...Object.entries(sciStats.classification_stats).map(([k, v]) => [k, v]),
        ],
      });
      sections.push({
        name: "ROC Curve (FPR vs TPR)",
        rows: [
          ["fpr", "tpr", "threshold"],
          ...sciStats.roc_curve.map((p) => [p.x, p.y, p.t ?? ""]),
        ],
      });
      sections.push({
        name: "Precision-Recall Curve",
        rows: [
          ["recall", "precision", "threshold"],
          ...sciStats.pr_curve.map((p) => [p.x, p.y, p.t ?? ""]),
        ],
      });
    }
    if (monthly.length) {
      sections.push({
        name: "Rolling Monthly AUC (in-sample descriptive diagnostic)",
        rows: [
          ["month", "auc", "positive_rate", "n_rows"],
          ...monthly.map((m) => [m.month, m.auc, m.positive_rate, m.n]),
        ],
      });
    }
    if (featureImportance.length) {
      sections.push({
        name: "Feature Importance (top 20)",
        rows: [
          ["rank", "feature", "importance"],
          ...featureImportance.map((f, i) => [i + 1, f.feature, f.importance]),
        ],
      });
    }
    if (provinces.length) {
      sections.push({
        name: "Provincial Breakdown (latest retained map candidates)",
        rows: [
          ["province", "critical", "high", "medium", "low", "total"],
          ...provinces.map((p) => [p.prov, p.CRITICAL, p.HIGH, p.MEDIUM, p.LOW, p.total]),
        ],
      });
    }
    return sections;
  };

  // ──── Combined: single CSV with all sections + section headers ────
  const exportCombined = () => {
    const sections = buildSections();
    const stamp = new Date().toISOString().slice(0, 10);
    downloadMultiSectionCsv(`firewatch_full_report_${stamp}.csv`, sections);
  };

  // ──── Separate: one CSV per section (current behaviour) ────
  const exportAll = () => {
    if (sciStats) {
      const sampleRows = scientificSampleSplits(sciStats.samples);
      downloadCsv("dataset_split.csv", [
        ["split", "n_rows", "positives", "positive_rate", "date_start", "date_end"],
        ...sampleRows.map((row) => [
          row.key,
          row.data.n,
          row.data.positives,
          row.data.positive_rate,
          ...row.data.date_range,
        ]),
      ]);
      downloadCsv("bootstrap_ci_95.csv", [
        ["metric", "point", "ci_lower", "ci_upper", "std", "n_boot"],
        ...(Object.entries(sciStats.ci_95) as [string, typeof sciStats.ci_95.roc_auc][]).map(
          ([k, v]) => [k, v.point, v.lower, v.upper, v.std, v.n_boot]
        ),
      ]);
      downloadCsv("confusion_matrix.csv", [
        ["", "predicted_no_fire", "predicted_fire"],
        ["actual_no_fire", sciStats.confusion_matrix.tn, sciStats.confusion_matrix.fp],
        ["actual_fire", sciStats.confusion_matrix.fn, sciStats.confusion_matrix.tp],
      ]);
      downloadCsv("classification_stats.csv", [
        ["metric", "value"],
        ...Object.entries(sciStats.classification_stats).map(([k, v]) => [k, v]),
      ]);
      downloadCsv("roc_curve.csv", [
        ["fpr", "tpr", "threshold"],
        ...sciStats.roc_curve.map((p) => [p.x, p.y, p.t ?? ""]),
      ]);
      downloadCsv("pr_curve.csv", [
        ["recall", "precision", "threshold"],
        ...sciStats.pr_curve.map((p) => [p.x, p.y, p.t ?? ""]),
      ]);
    }
    if (monthly.length) {
      downloadCsv("rolling_monthly_auc.csv", [
        ["month", "auc", "positive_rate", "n_rows"],
        ...monthly.map((m) => [m.month, m.auc, m.positive_rate, m.n]),
      ]);
    }
    if (featureImportance.length) {
      downloadCsv("feature_importance.csv", [
        ["rank", "feature", "importance"],
        ...featureImportance.map((f, i) => [i + 1, f.feature, f.importance]),
      ]);
    }
    if (provinces.length) {
      downloadCsv("provinces_breakdown.csv", [
        ["province", "critical", "high", "medium", "low", "total"],
        ...provinces.map((p) => [p.prov, p.CRITICAL, p.HIGH, p.MEDIUM, p.LOW, p.total]),
      ]);
    }
  };

  return (
    <div className="reports-page">
      <header className="notify-page-header" style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 16, flexWrap: "wrap" }}>
        <div>
          <h1>{t("page.reports.title", "📊 Reports · Scientific Analysis")}</h1>
          <p className="notify-page-subtitle">
            {t(sciStats
              ? "page.reports.subtitle"
              : "page.reports.subtitle.core")}
          </p>
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 6, alignItems: "flex-end" }}>
          <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
            <button
              type="button"
              className="action-btn primary"
              onClick={exportCombined}
              title={t("btn.export.combined")}
            >
              {t("btn.export.combined", "📄 All metrics in one file")}
            </button>
            <button
              type="button"
              className="action-btn"
              onClick={exportAll}
              title={t("btn.export.separate")}
            >
              {fmtTr(t("btn.export.separate"), {
                count: separateFileCount,
              })}
            </button>
          </div>
          <span style={{ fontSize: 10, color: "var(--text-3)" }}>
            {t("section.summary.hint")}
          </span>
        </div>
      </header>

      {/* Model Training Summary — top-of-page executive overview from real
          persisted artifacts. No fabricated numbers. */}
      <ModelTrainingSummary metrics={metrics} />

      {/* Plain-language summary table — for readers who skip technical sections */}
      {sciStats && (
        <section className="report-section">
          <div className="report-section-head">
            <h2>{t("section.summary")}</h2>
            <p className="report-section-hint">{t("section.summary.hint")}</p>
          </div>
          <table className="stats-table summary-friendly">
            <thead>
              <tr>
                <th>{t("section.summary.col.aspect")}</th>
                <th style={{ textAlign: "right" }}>{t("section.summary.col.result")}</th>
                <th>{t("section.summary.col.meaning")}</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>{t("section.summary.row.recall")}</td>
                <td className="mono right"><b style={{ color: recallLimitedByDrift ? "#60a5fa" : "#22c55e" }}>{(sciStats.classification_stats.sensitivity * 100).toFixed(0)}%</b></td>
                <td>
                  {fmtTr(t("section.summary.meaning.recall"), {
                    count: (sciStats.classification_stats.sensitivity * 100).toFixed(0),
                  })}
                  {" "}{t(recallLimitedByDrift
                    ? "section.summary.meaning.recall.watch_drift"
                    : "section.summary.meaning.recall.watch")}
                </td>
              </tr>
              <tr>
                <td>{t("section.summary.row.fpr")}</td>
                <td className="mono right"><b style={{ color: "#eab308" }}>{(sciStats.classification_stats.false_positive_rate * 100).toFixed(0)}%</b></td>
                <td>
                  {fmtTr(t("section.summary.meaning.fpr"), {
                    count: (sciStats.classification_stats.false_positive_rate * 100).toFixed(0),
                  })}
                </td>
              </tr>
              <tr>
                <td>{t("section.summary.row.auc")}</td>
                <td className="mono right"><b style={{ color: "#22c55e" }}>{(sciStats.ci_95.roc_auc.point * 100).toFixed(0)}%</b></td>
                <td>
                  {fmtTr(t("section.summary.meaning.auc"), {
                    auc: sciStats.ci_95.roc_auc.point.toFixed(3),
                  })}
                </td>
              </tr>
              <tr>
                <td>{t("section.summary.row.calib")}</td>
                <td className="mono right">
                  <b style={{ color: "#60a5fa" }}>
                    {sciStats.classification_stats.brier_score.toFixed(4)}
                  </b>
                </td>
                <td>
                  {fmtTr(t("section.summary.meaning.brier"), {
                    score: sciStats.classification_stats.brier_score.toFixed(4),
                  })}
                </td>
              </tr>
              <tr>
                <td>{t("section.summary.row.miss")}</td>
                <td className="mono right"><b style={{ color: "#ef4444" }}>{(sciStats.classification_stats.false_negative_rate * 100).toFixed(0)}%</b></td>
                <td>
                  {fmtTr(t("section.summary.meaning.miss"), {
                    count: (sciStats.classification_stats.false_negative_rate * 100).toFixed(0),
                  })}
                </td>
              </tr>
            </tbody>
          </table>
        </section>
      )}

      {/* Friendly confusion matrix — colored cards instead of just numbers */}
      {sciStats && (
        <section className="report-section">
          <div className="report-section-head">
            <h2>{t("section.confusion")}</h2>
            <p className="report-section-hint">
              {fmtTr(t("section.confusion.hint"), {
                n: (sciStats.confusion_matrix.tn + sciStats.confusion_matrix.fp + sciStats.confusion_matrix.fn + sciStats.confusion_matrix.tp).toLocaleString(),
              })}
            </p>
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(2, 1fr)", gap: 12 }}>
            <div style={{ padding: 14, background: "rgba(34, 197, 94, 0.10)", border: "1px solid rgba(34, 197, 94, 0.4)", borderRadius: 8 }}>
              <div style={{ fontSize: 12, fontWeight: 700, color: "#22c55e", marginBottom: 6 }}>{t("section.confusion.tn")}</div>
              <div style={{ fontSize: 22, fontWeight: 700 }}>{sciStats.confusion_matrix.tn.toLocaleString()}</div>
              <div style={{ fontSize: 11, color: "var(--text-3)", marginTop: 4 }}>{t("section.confusion.tn.exp")}</div>
            </div>
            <div style={{ padding: 14, background: "rgba(234, 179, 8, 0.10)", border: "1px solid rgba(234, 179, 8, 0.4)", borderRadius: 8 }}>
              <div style={{ fontSize: 12, fontWeight: 700, color: "#eab308", marginBottom: 6 }}>{t("section.confusion.fp")}</div>
              <div style={{ fontSize: 22, fontWeight: 700 }}>{sciStats.confusion_matrix.fp.toLocaleString()}</div>
              <div style={{ fontSize: 11, color: "var(--text-3)", marginTop: 4 }}>{t("section.confusion.fp.exp")}</div>
            </div>
            <div style={{ padding: 14, background: "rgba(239, 68, 68, 0.10)", border: "1px solid rgba(239, 68, 68, 0.4)", borderRadius: 8 }}>
              <div style={{ fontSize: 12, fontWeight: 700, color: "#ef4444", marginBottom: 6 }}>{t("section.confusion.fn")}</div>
              <div style={{ fontSize: 22, fontWeight: 700 }}>{sciStats.confusion_matrix.fn.toLocaleString()}</div>
              <div style={{ fontSize: 11, color: "var(--text-3)", marginTop: 4 }}>{t("section.confusion.fn.exp")}</div>
            </div>
            <div style={{ padding: 14, background: "rgba(34, 197, 94, 0.10)", border: "1px solid rgba(34, 197, 94, 0.4)", borderRadius: 8 }}>
              <div style={{ fontSize: 12, fontWeight: 700, color: "#22c55e", marginBottom: 6 }}>{t("section.confusion.tp")}</div>
              <div style={{ fontSize: 22, fontWeight: 700 }}>{sciStats.confusion_matrix.tp.toLocaleString()}</div>
              <div style={{ fontSize: 11, color: "var(--text-3)", marginTop: 4 }}>{t("section.confusion.tp.exp")}</div>
            </div>
          </div>
          <p className="report-section-hint" style={{ marginTop: 12 }}>
            {fmtTr(t("section.confusion.summary"), {
              recall: (sciStats.classification_stats.sensitivity * 100).toFixed(0),
              fpr: (sciStats.classification_stats.false_positive_rate * 100).toFixed(0),
            })}
          </p>
        </section>
      )}

      {/* Radar chart — metric balance overview */}
      {sciStats && (
        <section className="report-section">
          <div className="report-section-head" style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 10 }}>
            <div style={{ flex: 1 }}>
              <h2>{t("section.radar")}</h2>
              <p className="report-section-hint">{t("section.radar.hint")}</p>
            </div>
            <ChartToolbar
              containerRef={radarChartRef}
              fileStem="metrics-radar"
              getCsvRows={() => {
                const cs = sciStats.classification_stats;
                const auc = sciStats.ci_95.roc_auc.point;
                return [
                  ["axis", "value"],
                  ["Recall (sensitivity)", cs.sensitivity],
                  ["Specificity", cs.specificity],
                  ["AUC", auc],
                  ["Precision (PPV)", cs.ppv],
                  ["Negative predictive value", cs.npv],
                ];
              }}
            />
          </div>
          {(() => {
            const cs = sciStats.classification_stats;
            const auc = sciStats.ci_95.roc_auc.point;
            const data = [
              { axis: t("section.radar.axis.recall"),      value: cs.sensitivity },
              { axis: t("section.radar.axis.specificity"), value: cs.specificity },
              { axis: t("section.radar.axis.auc"),         value: auc },
              { axis: t("section.radar.axis.precision"),   value: cs.ppv },
              { axis: t("section.radar.axis.npv"),         value: cs.npv },
            ];
            return (
              <div ref={radarChartRef} style={{ height: 360 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <RadarChart data={data} outerRadius="75%">
                    <PolarGrid stroke="var(--border)" />
                    <PolarAngleAxis dataKey="axis" tick={{ fill: "var(--text-2)", fontSize: 11 }} />
                    <PolarRadiusAxis
                      domain={[0, 1]}
                      tick={{ fill: "var(--text-3)", fontSize: 10 }}
                      tickFormatter={(v) => `${Math.round(v * 100)}%`}
                      angle={90}
                    />
                    <Radar
                      name={t("section.radar.legend.current")}
                      dataKey="value"
                      stroke="var(--accent)"
                      fill="var(--accent)"
                      fillOpacity={0.25}
                      strokeWidth={2}
                    />
                    <Legend
                      wrapperStyle={{ fontSize: 11, color: "var(--text-2)" }}
                    />
                    <Tooltip
                      contentStyle={{
                        background: "var(--surface)",
                        border: "1px solid var(--border)",
                        borderRadius: 6,
                        fontSize: 12,
                      }}
                      formatter={(value) => {
                        const n = typeof value === "number" ? value : Number(value);
                        return Number.isFinite(n) ? `${(n * 100).toFixed(1)}%` : String(value);
                      }}
                    />
                  </RadarChart>
                </ResponsiveContainer>
              </div>
            );
          })()}
        </section>
      )}

      {/* Threshold analysis — P/R/F1 vs threshold to help operator pick */}
      {sciStats && sciStats.pr_curve && sciStats.pr_curve.length > 0 && (
        <section className="report-section">
          <div className="report-section-head" style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 10 }}>
            <div style={{ flex: 1 }}>
              <h2>{t("section.threshold")}</h2>
              <p className="report-section-hint">
                {deploymentThreshold == null
                  ? t("section.threshold.hint.no_deploy")
                  : fmtTr(t("section.threshold.hint"), {
                      threshold: deploymentThreshold.toFixed(4),
                    })}
              </p>
            </div>
            <ChartToolbar
              containerRef={thresholdChartRef}
              fileStem="threshold-analysis"
              getCsvRows={() => {
                const rows: (string | number)[][] = [["threshold", "precision", "recall", "f1"]];
                for (const p of sciStats.pr_curve) {
                  if (p.t == null) continue;
                  const f1 = (p.x + p.y) > 0 ? (2 * p.x * p.y) / (p.x + p.y) : 0;
                  rows.push([p.t, p.y, p.x, f1]);
                }
                return rows;
              }}
            />
          </div>
          {(() => {
            const rows = sciStats.pr_curve
              .filter((p) => p.t != null && p.t >= 0 && p.t <= 1)
              .map((p) => {
                const f1 = (p.x + p.y) > 0 ? (2 * p.x * p.y) / (p.x + p.y) : 0;
                return { threshold: p.t as number, precision: p.y, recall: p.x, f1 };
              })
              .sort((a, b) => a.threshold - b.threshold);
            return (
              <div ref={thresholdChartRef} style={{ height: 320 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={rows} margin={{ top: 16, right: 24, bottom: 32, left: 0 }}>
                    <CartesianGrid stroke="var(--border-soft)" strokeDasharray="3 3" />
                    <XAxis
                      dataKey="threshold"
                      type="number"
                      domain={[0, 1]}
                      ticks={[0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1]}
                      tick={{ fill: "var(--text-3)", fontSize: 11 }}
                      label={{ value: "Threshold", position: "insideBottom", offset: -8, style: { fill: "var(--text-3)", fontSize: 11 } }}
                    />
                    <YAxis
                      domain={[0, 1]}
                      tickFormatter={(v) => `${Math.round(v * 100)}%`}
                      tick={{ fill: "var(--text-3)", fontSize: 11 }}
                    />
                    {deploymentThreshold != null && (
                      <ReferenceLine
                        x={deploymentThreshold}
                        stroke="#eab308"
                        strokeDasharray="4 4"
                        label={{ value: t("section.threshold.deploy"), fill: "#eab308", fontSize: 10, position: "insideTopRight" }}
                      />
                    )}
                    <Tooltip
                      contentStyle={{
                        background: "var(--surface)",
                        border: "1px solid var(--border)",
                        borderRadius: 6,
                        fontSize: 12,
                        color: "var(--text)",
                      }}
                      formatter={(value, name) => {
                        const n = typeof value === "number" ? value : Number(value);
                        return [Number.isFinite(n) ? `${(n * 100).toFixed(1)}%` : String(value), name];
                      }}
                      labelFormatter={(label) => `Threshold: ${Number(label).toFixed(3)}`}
                    />
                    <Legend wrapperStyle={{ fontSize: 11, color: "var(--text-2)", paddingTop: 8 }} />
                    <Line type="monotone" dataKey="precision" name="Precision" stroke="#22c55e" strokeWidth={2} dot={false} />
                    <Line type="monotone" dataKey="recall" name="Recall" stroke="#3b82f6" strokeWidth={2} dot={false} />
                    <Line type="monotone" dataKey="f1" name="F1" stroke="var(--accent)" strokeWidth={2} dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            );
          })()}
          <div style={{ marginTop: 10, fontSize: 12, color: "var(--text-2)", lineHeight: 1.55 }}>
            {deploymentThreshold == null
              ? t("section.threshold.advice.no_deploy")
              : fmtTr(t("section.threshold.advice"), {
                  threshold: deploymentThreshold.toFixed(4),
                })}
          </div>
        </section>
      )}

      {/* Distribution by persisted absolute risk tier */}
      {tierDistribution.rows.some((r) => r.count > 0) && (
        <section className="report-section">
          <div className="report-section-head" style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 10 }}>
            <div style={{ flex: 1 }}>
              <h2>
                {t("section.distribution")} · {tierDistribution.latest ?? ""}
              </h2>
              <p className="report-section-hint">
                {t("section.distribution.hint")}
              </p>
            </div>
            <ChartToolbar
              containerRef={daysChartRef}
              fileStem="cells-by-risk-tier"
              shape={daysShape}
              onShapeChange={setDaysShape}
              getCsvRows={() => [
                ["risk_tier", "count"],
                ...tierDistribution.rows.map((r) => [r.tier, r.count]),
              ]}
            />
          </div>
          <div ref={daysChartRef} style={{ height: 280 }}>
            <ResponsiveContainer width="100%" height="100%">
              {daysShape === "bar" ? (
                <BarChart data={tierDistribution.rows} margin={{ top: 24, right: 16, bottom: 8, left: 0 }}>
                  <CartesianGrid stroke="var(--border-soft)" strokeDasharray="3 3" vertical={false} />
                  <XAxis dataKey="tier" tick={{ fill: "var(--text-3)", fontSize: 11 }} axisLine={{ stroke: "var(--border)" }} tickLine={false} />
                  <YAxis tick={{ fill: "var(--text-3)", fontSize: 11 }} axisLine={{ stroke: "var(--border)" }} tickLine={false}
                    label={{ value: t("section.distribution.yaxis"), angle: -90, position: "insideLeft", offset: 16, style: { fill: "var(--text-3)", fontSize: 11 } }} />
                  <Tooltip
                    cursor={{ fill: "rgba(255, 107, 53, 0.08)" }}
                    contentStyle={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 6, fontSize: 12, color: "var(--text)" }}
                    formatter={(value) => [`${value} ${t("section.distribution.tooltip.count")}`, ""]}
                  />
                  <Bar dataKey="count" radius={[6, 6, 0, 0]}>
                    {tierDistribution.rows.map((r) => (
                      <Cell key={r.tier} fill={r.color} />
                    ))}
                    <LabelList dataKey="count" position="top"
                      style={{ fill: "var(--text)", fontSize: 11, fontWeight: 600 }}
                      formatter={(v) => {
                        const n = typeof v === "number" ? v : Number(v);
                        return Number.isFinite(n) && n > 0 ? String(n) : "";
                      }}
                    />
                  </Bar>
                </BarChart>
              ) : (
                <LineChart data={tierDistribution.rows} margin={{ top: 24, right: 16, bottom: 8, left: 0 }}>
                  <CartesianGrid stroke="var(--border-soft)" strokeDasharray="3 3" />
                  <XAxis dataKey="tier" tick={{ fill: "var(--text-3)", fontSize: 11 }} axisLine={{ stroke: "var(--border)" }} tickLine={false} />
                  <YAxis tick={{ fill: "var(--text-3)", fontSize: 11 }} axisLine={{ stroke: "var(--border)" }} tickLine={false}
                    label={{ value: t("section.distribution.yaxis"), angle: -90, position: "insideLeft", offset: 16, style: { fill: "var(--text-3)", fontSize: 11 } }} />
                  <Tooltip
                    contentStyle={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 6, fontSize: 12, color: "var(--text)" }}
                    formatter={(value) => [`${value} ${t("section.distribution.tooltip.count")}`, ""]}
                  />
                  <Line type="monotone" dataKey="count" stroke="var(--accent)" strokeWidth={2.5} dot={{ r: 5, fill: "var(--accent)" }} />
                </LineChart>
              )}
            </ResponsiveContainer>
          </div>
          <div style={{
            marginTop: 10, padding: "10px 14px",
            background: "rgba(59, 130, 246, 0.08)",
            border: "1px solid rgba(59, 130, 246, 0.25)",
            borderRadius: 6, fontSize: 12, color: "var(--text-2)", lineHeight: 1.6,
          }}>
            {t("section.distribution.tiers")}
          </div>
        </section>
      )}

      {/* Scientific stats — technical section */}
      {sciStats ? (
        <StatisticsSection stats={sciStats} />
      ) : (
        <section className="report-section" role="status">
          <h2 style={{ fontSize: 14, marginBottom: 6 }}>
            {t("section.no_scistats.title")}
          </h2>
          <p style={{ color: "var(--text-3)", lineHeight: 1.55 }}>
            {t("section.no_scistats.hint")}
          </p>
        </section>
      )}

      {/* Section 1: Rolling AUC */}
      <section className="report-section">
        <div className="report-section-head" style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 10 }}>
          <div style={{ flex: 1 }}>
            <h2>{t("section.stability")} · {monthly.length}</h2>
            <p className="report-section-hint">{t("section.stability.dot")}</p>
          </div>
          {monthly.length > 0 && (
            <SmallExportBtn
              filename="rolling_monthly_auc.csv"
              getRows={() => [
                ["month", "auc", "positive_rate", "n_rows"],
                ...monthly.map((m) => [m.month, m.auc, m.positive_rate, m.n]),
              ]}
            />
          )}
        </div>
        {monthly.length > 0 ? (
          <RollingAucChart data={monthly} />
        ) : rollingLoadState === "loading" ? (
          <EmptyReport msg={t("section.stability.loading")} />
        ) : rollingLoadState === "error" ? (
          <div className="report-inline-error" role="alert">
            <span>{t("section.stability.error")}</span>
            <button
              type="button"
              className="action-btn"
              onClick={() => setRollingRetryKey((key) => key + 1)}
            >
              {t("analytics.error.retry")}
            </button>
          </div>
        ) : (
          <EmptyReport msg={t("section.stability.empty")} />
        )}
        {metrics?.stability_auc_mean != null && (
          <div className="report-stats-grid">
            <Stat label="AUC mean" value={metrics.stability_auc_mean?.toFixed(3)} />
            <Stat label="AUC std" value={metrics.stability_auc_std?.toFixed(3)} />
            <Stat label="AUC min" value={metrics.stability_auc_min?.toFixed(3)} />
            <Stat label="AUC max" value={metrics.stability_auc_max?.toFixed(3)} />
          </div>
        )}
      </section>

      {/* Section 2: Feature importance */}
      <section className="report-section">
        <div className="report-section-head" style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 10 }}>
          <div style={{ flex: 1 }}>
            <h2>{t("section.features")} — top {Math.min(featureImportance.length, 20)}</h2>
            <p className="report-section-hint">{t("section.features.hint")}</p>
          </div>
          {featureImportance.length > 0 && (
            <SmallExportBtn
              filename="feature_importance.csv"
              getRows={() => [
                ["rank", "feature", "importance"],
                ...featureImportance.map((f, i) => [i + 1, f.feature, f.importance]),
              ]}
            />
          )}
        </div>
        {featureImportance.length > 0 ? (
          <FeatureImportanceChart data={featureImportance} />
        ) : (
          <EmptyReport msg={t("section.features.empty")} />
        )}
      </section>

      {/* Section 3: Provincial breakdown */}
      <section className="report-section">
        <div className="report-section-head" style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 10 }}>
          <div style={{ flex: 1 }}>
            <h2>{t("section.provinces.title")}</h2>
            <p className="report-section-hint">{t("section.provinces.hint")}</p>
          </div>
          {provinces.length > 0 && (
            <SmallExportBtn
              filename="provinces_breakdown.csv"
              getRows={() => [
                ["province", "critical", "high", "medium", "low", "total"],
                ...provinces.map((p) => [p.prov, p.CRITICAL, p.HIGH, p.MEDIUM, p.LOW, p.total]),
              ]}
            />
          )}
        </div>
        {provinces.length > 0 ? (
          <ProvinceChart data={provinces} />
        ) : (
          <EmptyReport msg={t("section.provinces.empty")} />
        )}
      </section>

      <footer className="report-footer">
        <p style={{ fontSize: 11, color: "var(--text-3)" }}>
          {t("reports.raw_data")}: {" "}
          <code style={{ background: "var(--surface-2)", padding: "1px 4px", borderRadius: 3 }}>outputs/metadata/rolling_eval.json</code>{" · "}
          <code style={{ background: "var(--surface-2)", padding: "1px 4px", borderRadius: 3 }}>outputs/metadata/dataset_info.json</code>
        </p>
      </footer>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────
// Inline SVG charts (no chart library)
// ─────────────────────────────────────────────────────────────

interface RollingPoint {
  month: string;
  auc: number;
  positive_rate: number;
  n: number;
}

function RollingAucChart({ data }: { data: RollingPoint[] }) {
  const W = 800;
  const H = 240;
  const PAD_L = 50;
  const PAD_R = 20;
  const PAD_T = 20;
  const PAD_B = 50;
  const plotW = W - PAD_L - PAD_R;
  const plotH = H - PAD_T - PAD_B;
  const n = data.length;
  const xToPx = (i: number) => PAD_L + (n > 1 ? (i / (n - 1)) * plotW : plotW / 2);
  const yToPx = (auc: number) => PAD_T + (1 - (auc - 0.5) / 0.5) * plotH;

  const path = data
    .map((d, i) => `${i === 0 ? "M" : "L"} ${xToPx(i).toFixed(1)} ${yToPx(d.auc).toFixed(1)}`)
    .join(" ");

  return (
    <div style={{ overflowX: "auto" }}>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        style={{ width: "100%", maxWidth: 900, background: "var(--surface-2)", borderRadius: 6 }}
        role="img"
        aria-label="Rolling monthly AUC chart"
      >
        {/* Y-axis grid + labels */}
        {[0.5, 0.65, 0.75, 0.85, 1.0].map((t) => (
          <g key={t}>
            <line
              x1={PAD_L} y1={yToPx(t)} x2={W - PAD_R} y2={yToPx(t)}
              stroke="var(--border-soft)" strokeWidth={1}
            />
            <text x={PAD_L - 6} y={yToPx(t) + 3} fill="var(--text-3)" fontSize="10" textAnchor="end">
              {t.toFixed(2)}
            </text>
          </g>
        ))}
        {/* Baseline 0.5 */}
        <line
          x1={PAD_L} y1={yToPx(0.5)} x2={W - PAD_R} y2={yToPx(0.5)}
          stroke="#ef4444" strokeWidth={1.5} strokeDasharray="4 3"
        />
        <text x={W - PAD_R - 4} y={yToPx(0.5) - 4} fill="#ef4444" fontSize="9" textAnchor="end">
          random = 0.5
        </text>
        {/* Line */}
        <path d={path} fill="none" stroke="#22c55e" strokeWidth={2} />
        {/* Markers */}
        {data.map((d, i) => (
          <g key={d.month}>
            <circle
              cx={xToPx(i)}
              cy={yToPx(d.auc)}
              r={4}
              fill={d.auc >= 0.85 ? "#22c55e" : d.auc >= 0.70 ? "#eab308" : "#ef4444"}
              stroke="var(--surface)"
              strokeWidth={1.5}
            >
              <title>{`${d.month}: AUC ${d.auc.toFixed(3)} · positives ${(d.positive_rate * 100).toFixed(2)}% · n=${d.n.toLocaleString()}`}</title>
            </circle>
            {/* Month label every ~3rd */}
            {(i % Math.max(1, Math.floor(n / 8)) === 0 || i === n - 1) && (
              <text
                x={xToPx(i)} y={H - PAD_B + 14}
                fill="var(--text-3)" fontSize="9" textAnchor="middle"
                transform={`rotate(-25 ${xToPx(i)} ${H - PAD_B + 14})`}
              >
                {d.month}
              </text>
            )}
          </g>
        ))}
        {/* Axis labels */}
        <text x={PAD_L + plotW / 2} y={H - 6} fill="var(--text-3)" fontSize="11" textAnchor="middle">Month</text>
        <text
          x={14} y={PAD_T + plotH / 2}
          fill="var(--text-3)" fontSize="11" textAnchor="middle"
          transform={`rotate(-90 14 ${PAD_T + plotH / 2})`}
        >
          ROC-AUC
        </text>
      </svg>
    </div>
  );
}

function FeatureImportanceChart({
  data,
}: {
  data: { feature: string; importance: number }[];
}) {
  const W = 800;
  const ROW_H = 22;
  const PAD_L = 220;
  const PAD_R = 60;
  const PAD_T = 6;
  const H = data.length * ROW_H + PAD_T * 2;
  const plotW = W - PAD_L - PAD_R;
  const max = Math.max(...data.map((d) => d.importance), 1);

  return (
    <div style={{ overflowX: "auto" }}>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        style={{ width: "100%", background: "var(--surface-2)", borderRadius: 6 }}
        role="img"
        aria-label="Feature importance chart"
      >
        {data.map((d, i) => {
          const y = PAD_T + i * ROW_H;
          const barW = (d.importance / max) * plotW;
          const color = i < 3 ? "#22c55e" : i < 10 ? "#84cc16" : "#3b82f6";
          return (
            <g key={d.feature}>
              <text
                x={PAD_L - 8} y={y + ROW_H / 2 + 4}
                fill="var(--text-2)" fontSize="11" textAnchor="end"
                style={{ fontFamily: "ui-monospace, monospace" }}
              >
                {d.feature.length > 30 ? d.feature.slice(0, 30) + "…" : d.feature}
              </text>
              <rect
                x={PAD_L} y={y + 3}
                width={barW} height={ROW_H - 6}
                fill={color}
                fillOpacity={0.85}
                rx={2}
              >
                <title>{`${d.feature}: ${d.importance.toFixed(1)}`}</title>
              </rect>
              <text
                x={PAD_L + barW + 6} y={y + ROW_H / 2 + 4}
                fill="var(--text-2)" fontSize="11"
                style={{ fontFamily: "ui-monospace, monospace" }}
              >
                {d.importance.toFixed(1)}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

interface ProvincePoint {
  prov: string;
  CRITICAL: number;
  HIGH: number;
  MEDIUM: number;
  LOW: number;
  total: number;
}

function ProvinceChart({ data }: { data: ProvincePoint[] }) {
  const W = 800;
  const ROW_H = 26;
  const PAD_L = 160;
  const PAD_R = 40;
  const PAD_T = 10;
  const H = data.length * ROW_H + PAD_T * 2;
  const plotW = W - PAD_L - PAD_R;
  const max = Math.max(...data.map((d) => d.total), 1);

  return (
    <div style={{ overflowX: "auto" }}>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        style={{ width: "100%", background: "var(--surface-2)", borderRadius: 6 }}
        role="img"
        aria-label="Provinces by alert count"
      >
        {data.map((d, i) => {
          const y = PAD_T + i * ROW_H;
          let xOffset = PAD_L;
          const segments: { tier: keyof typeof TIER_COLORS; value: number }[] = [
            { tier: "CRITICAL", value: d.CRITICAL },
            { tier: "HIGH",     value: d.HIGH },
            { tier: "MEDIUM",   value: d.MEDIUM },
            { tier: "LOW",      value: d.LOW },
          ];
          return (
            <g key={d.prov}>
              <text
                x={PAD_L - 8} y={y + ROW_H / 2 + 4}
                fill="var(--text-2)" fontSize="11" textAnchor="end"
              >
                {d.prov.length > 22 ? d.prov.slice(0, 22) + "…" : d.prov}
              </text>
              {segments.map((s) => {
                if (s.value === 0) return null;
                const segW = (s.value / max) * plotW;
                const seg = (
                  <rect
                    key={s.tier}
                    x={xOffset} y={y + 4}
                    width={segW} height={ROW_H - 8}
                    fill={TIER_COLORS[s.tier]}
                    rx={2}
                  >
                    <title>{`${d.prov} ${s.tier}: ${s.value}`}</title>
                  </rect>
                );
                xOffset += segW;
                return seg;
              })}
              <text
                x={xOffset + 6} y={y + ROW_H / 2 + 4}
                fill="var(--text-2)" fontSize="11"
                style={{ fontFamily: "ui-monospace, monospace" }}
              >
                {d.total}
              </text>
            </g>
          );
        })}
      </svg>
      {/* Legend */}
      <div style={{ display: "flex", gap: 12, marginTop: 8, justifyContent: "center", flexWrap: "wrap" }}>
        {(["CRITICAL", "HIGH", "MEDIUM", "LOW"] as const).map((t) => (
          <span key={t} style={{ display: "inline-flex", alignItems: "center", gap: 4, fontSize: 11, color: "var(--text-2)" }}>
            <span style={{ width: 10, height: 10, background: TIER_COLORS[t], borderRadius: 2 }} />
            {t}
          </span>
        ))}
      </div>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string | undefined }) {
  return (
    <div>
      <div style={{ fontSize: 10, color: "var(--text-3)", textTransform: "uppercase", letterSpacing: "0.06em", fontWeight: 600 }}>
        {label}
      </div>
      <div style={{ fontSize: 22, fontWeight: 700, color: "var(--text)", fontVariantNumeric: "tabular-nums" }}>
        {value ?? "—"}
      </div>
    </div>
  );
}

function EmptyReport({ msg }: { msg: string }) {
  return (
    <div style={{
      padding: 28,
      background: "var(--surface-2)",
      borderRadius: 6,
      color: "var(--text-3)",
      fontSize: 12,
      textAlign: "center",
    }}>
      {msg}
    </div>
  );
}
