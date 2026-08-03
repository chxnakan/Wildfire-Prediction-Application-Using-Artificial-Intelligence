import { useEffect } from "react";
import type {
  FireFeature,
  GeoJsonMetadata,
  GistdaFeature,
  ReliabilityBin,
  UrgencyThresholds,
  ValidationMetrics,
} from "../types";

// Small inline SVG reliability diagram. Pure DOM, no chart library.
// X = mean predicted probability per bin; Y = actual positive rate.
// Red diagonal = perfect calibration. Closer to diagonal = better.
function ReliabilityDiagram({ bins }: { bins: ReliabilityBin[] }) {
  if (!bins.length) return null;
  const W = 280;
  const H = 200;
  const PAD = 30;
  const plotW = W - 2 * PAD;
  const plotH = H - 2 * PAD;
  const xToPx = (p: number) => PAD + p * plotW;
  const yToPx = (p: number) => H - PAD - p * plotH;
  const maxCount = Math.max(...bins.map((b) => b.count), 1);
  return (
    <svg width={W} height={H} style={{ background: "#1f2329", borderRadius: 6 }}>
      {/* Grid + axes */}
      {[0, 0.25, 0.5, 0.75, 1.0].map((t) => (
        <g key={t}>
          <line x1={xToPx(t)} y1={PAD} x2={xToPx(t)} y2={H - PAD} stroke="#2b3038" strokeWidth={1} />
          <line x1={PAD} y1={yToPx(t)} x2={W - PAD} y2={yToPx(t)} stroke="#2b3038" strokeWidth={1} />
          <text x={xToPx(t)} y={H - PAD + 14} fill="#6c707a" fontSize="10" textAnchor="middle">{t.toFixed(2)}</text>
          <text x={PAD - 6} y={yToPx(t) + 3} fill="#6c707a" fontSize="10" textAnchor="end">{t.toFixed(2)}</text>
        </g>
      ))}
      {/* Perfect calibration diagonal */}
      <line x1={xToPx(0)} y1={yToPx(0)} x2={xToPx(1)} y2={yToPx(1)} stroke="#ef4444" strokeWidth={1.5} strokeDasharray="4 3" />
      {/* Bin dots, sized by count */}
      {bins.map((b, i) => {
        const r = 3 + 6 * Math.sqrt(b.count / maxCount);
        return (
          <g key={i}>
            <circle
              cx={xToPx(b.mean_predicted)}
              cy={yToPx(b.actual_rate)}
              r={r}
              fill="#22c55e"
              fillOpacity={0.7}
              stroke="#fff"
              strokeWidth={0.5}
            >
              <title>
                {`bin ${(b.bin_lower * 100).toFixed(0)}–${(b.bin_upper * 100).toFixed(0)}%\n` +
                 `predicted ${(b.mean_predicted * 100).toFixed(1)}%  actual ${(b.actual_rate * 100).toFixed(1)}%\n` +
                 `n = ${b.count.toLocaleString()}`}
              </title>
            </circle>
          </g>
        );
      })}
      {/* Axis labels */}
      <text x={W / 2} y={H - 2} fill="#9aa0aa" fontSize="11" textAnchor="middle">Mean predicted probability</text>
      <text x={10} y={H / 2} fill="#9aa0aa" fontSize="11" textAnchor="middle" transform={`rotate(-90 10 ${H / 2})`}>Actual fire rate</text>
    </svg>
  );
}

interface Props {
  open: boolean;
  onClose: () => void;
  activeBaseDate: string;
  predicted: FireFeature[]; // current snapshot (post-province filter)
  observed: FireFeature[]; // all observed features
  liveFires: GistdaFeature[];
  metrics: ValidationMetrics | null;
  thresholds: UrgencyThresholds | null;
  metadata: GeoJsonMetadata | null;
  selectedProvince: string;
  selectedDay: string;
}

// Pop-up with the detail an operator might want before trusting a number on
// the dashboard: training metrics, snapshot composition, urgency thresholds,
// data freshness, etc. Reads everything from already-loaded state — no
// network call.
export default function InfoModal(props: Props) {
  useEffect(() => {
    if (!props.open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") props.onClose();
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [props.open, props.onClose]);

  if (!props.open) return null;

  const m = props.metrics ?? {};
  const isBinary = Boolean(
    m.task === "binary_fire_in_3d" ||
    typeof m.roc_auc === "number" ||
    typeof m.deployment_threshold === "number",
  );
  const hasLegacyRegressionMetrics = Boolean(
    typeof m.accuracy_within_1day === "number" ||
    typeof m.mae_days === "number" ||
    typeof m.rmse_days === "number" ||
    typeof m.r2 === "number",
  );
  const hasPerformanceMetrics = isBinary || hasLegacyRegressionMetrics;
  const acc = m.accuracy_within_1day;
  const mae = m.mae_days;
  const rmse = m.rmse_days;
  const r2 = m.r2;

  // Per-tier counts on the current snapshot
  const tiers = { CRITICAL: 0, HIGH: 0, MEDIUM: 0, LOW: 0 } as Record<string, number>;
  for (const f of props.predicted) {
    const u = f.properties.urgency_level;
    if (u && u in tiers) tiers[u]++;
  }

  // Risk-tier counts (High / Watch / Background) — the binary model has no
  // per-day resolution, so we distribute by absolute risk tier, not by day.
  const tierCounts: { label: string; count: number }[] = [
    { label: "High", count: 0 },
    { label: "Watch", count: 0 },
    { label: "Background", count: 0 },
  ];
  for (const f of props.predicted) {
    const u = f.properties.urgency_level;
    if (u === "HIGH") tierCounts[0].count++;
    else if (u === "MEDIUM") tierCounts[1].count++;
    else if (u === "LOW") tierCounts[2].count++;
  }

  // Province distribution
  const byProvince: Record<string, number> = {};
  for (const f of props.predicted) {
    const p = (f.properties.province ?? "").trim() || "(unassigned)";
    byProvince[p] = (byProvince[p] ?? 0) + 1;
  }
  const topProvinces = Object.entries(byProvince)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 5);

  const fmt = (v: number | undefined, d = 3) =>
    typeof v === "number" && isFinite(v) ? v.toFixed(d) : "—";
  const fmtPct = (v: number | undefined) =>
    typeof v === "number" ? `${(v * 100).toFixed(1)}%` : "—";
  const alwaysNegativeBaseline =
    typeof m.test_positive_rate === "number"
      ? 1 - m.test_positive_rate
      : null;
  const probabilityThresholds = Boolean(
    props.thresholds &&
    Number.isFinite(props.thresholds.HIGH) &&
    Number.isFinite(props.thresholds.MEDIUM) &&
    props.thresholds.HIGH >= 0 &&
    props.thresholds.HIGH <= 1 &&
    props.thresholds.MEDIUM >= 0 &&
    props.thresholds.MEDIUM <= 1,
  );

  return (
    <div className="info-modal-backdrop" onClick={props.onClose}>
      <div
        className="info-modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="model-details-title"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="info-modal-header">
          <h2 id="model-details-title">Model & Snapshot Details</h2>
          <button type="button" className="info-modal-close" onClick={props.onClose} aria-label="Close">
            ×
          </button>
        </div>

        <div className="info-modal-body">
          <section>
            <h3>Held-out test performance</h3>
            {!hasPerformanceMetrics ? (
              <p className="info-note" role="status">
                Model metrics are unavailable in the loaded artifact. Risk
                cells can still be inspected, but performance claims are
                intentionally withheld until the metrics payload is restored.
              </p>
            ) : isBinary ? (
              <>
                <p className="info-note" style={{ marginTop: 0, marginBottom: 10 }}>
                  Task: <b>binary classification</b> — "fire within next{" "}
                  {m.imminent_days ?? 3} days?"
                </p>
                <div className="info-grid">
                  <div><span>ROC-AUC</span><b>{fmt(m.roc_auc, 3)}</b></div>
                  <div><span>Avg Precision</span><b>{fmt(m.average_precision, 3)}</b></div>
                  <div><span>Test pos rate</span><b>{fmtPct(m.test_positive_rate)}</b></div>
                  <div><span>ECE (calibration)</span><b>{fmt(m.ece, 4)}</b></div>
                </div>
                <p className="info-note" style={{ marginTop: 4 }}>
                  <b>ROC-AUC</b> วัดคุณภาพ ranking (0.5=สุ่ม, 1.0=perfect).{" "}
                  <b>ECE</b> = Expected Calibration Error — ยิ่งต่ำยิ่งดี แต่ต้องดู
                  reliability curve และ seasonal drift ร่วมด้วย ไม่ใช่ตรารับรองว่า
                  ทุก probability ใช้ได้ทุกฤดู{" "}
                  <b>Test pos rate</b> = ภาคจริงมีไฟกี่% — เป็น baseline ของ P@top-K.
                </p>

                {typeof m.stability_auc_mean === "number" && (
                  <>
                    <h4 style={{ marginTop: 16, marginBottom: 6 }}>
                      Stability across {m.stability_valid_months ?? m.stability_months} months
                    </h4>
                    <div className="info-grid">
                      <div><span>AUC mean</span><b>{fmt(m.stability_auc_mean, 3)}</b></div>
                      <div><span>AUC std</span><b>{fmt(m.stability_auc_std, 3)}</b></div>
                      <div><span>AUC min</span><b>{fmt(m.stability_auc_min, 3)}</b></div>
                      <div><span>AUC max</span><b>{fmt(m.stability_auc_max, 3)}</b></div>
                    </div>
                    {m.rolling_by_month && m.rolling_by_month.length > 0 && (
                      <div style={{ marginTop: 6, fontSize: 11, color: "var(--text-3)" }}>
                        Rolling monthly AUC ranges {fmt(m.stability_auc_min, 2)}–{fmt(m.stability_auc_max, 2)};
                        ดูราย script <code style={{ background: "var(--surface-2)", padding: "1px 4px", borderRadius: 3 }}>scripts/rolling_eval.py</code>
                      </div>
                    )}
                  </>
                )}

                <h4 style={{ marginTop: 16, marginBottom: 6 }}>Deployment threshold</h4>
                <div className="info-grid">
                  <div><span>Threshold</span><b>{fmt(m.deployment_threshold, 2)}</b></div>
                  <div><span>Precision</span><b>{fmtPct(m.deployment_precision)}</b></div>
                  <div><span>Recall</span><b>{fmtPct(m.deployment_recall)}</b></div>
                  <div><span>F1</span><b>{fmtPct(m.deployment_f1)}</b></div>
                </div>
                <p className="info-note" style={{ marginTop: 4 }}>
                  Probability cutoff ที่ใช้แปลงเป็น alert/no-alert. ตัวเลขข้างต้น
                  คือ Precision/Recall ณ จุดนี้บน held-out test.
                </p>

                <h4 style={{ marginTop: 16, marginBottom: 6 }}>Watch-list quality (top-K)</h4>
                <div className="info-grid">
                  <div><span>P @ top-5%</span><b>{fmtPct(m.precision_at_top_5pct)}</b></div>
                  <div><span>P @ top-10%</span><b>{fmtPct(m.precision_at_top_10pct)}</b></div>
                  <div><span>P @ top-20%</span><b>{fmtPct(m.precision_at_top_20pct)}</b></div>
                </div>
                {typeof m.uplift_at_top_20pct === "number" && (
                  <p className="info-note" style={{ marginTop: 4 }}>
                    Watch-list lift @top-20%: <b>{m.uplift_at_top_20pct.toFixed(2)}×</b> —{" "}
                    เลือก 20% เสี่ยงสุดที่โมเดลให้คะแนน เจอไฟมากกว่าสุ่มเลือก{" "}
                    {m.uplift_at_top_20pct.toFixed(1)} เท่า
                  </p>
                )}

                {m.reliability_bins && m.reliability_bins.length > 0 && (
                  <>
                    <h4 style={{ marginTop: 16, marginBottom: 6 }}>Reliability curve</h4>
                    <ReliabilityDiagram bins={m.reliability_bins} />
                    <p className="info-note" style={{ marginTop: 4 }}>
                      แต่ละจุด = bin ของ probability โมเดล (X) vs อัตราเกิดไฟจริงในกลุ่มนั้น (Y).
                      ทแยงสีแดง = perfect calibration. จุดที่ใกล้เส้นหมายถึง
                      calibration ของ bin นั้นดีกว่า แต่ยังต้องดูจำนวนตัวอย่างและ
                      seasonal drift ก่อนใช้ probability ราย cell.
                    </p>
                  </>
                )}

                <p className="info-note" style={{ marginTop: 12 }}>
                  Evaluated on the held-out test slice of a chronological 4-way
                  purged split (train / val / calibration / test), at real class
                  balance. Train-only undersamples negatives 4:1. Calibration and
                  thresholds are fit on the calibration slice — never on test.
                </p>
              </>
            ) : (
              <>
                <p className="info-note" style={{ marginTop: 0 }}>
                  Legacy date-regression artifact only. The active 3-day risk
                  model does not produce a per-cell fire date.
                </p>
                <div className="info-grid">
                  <div><span>±1 day acc</span><b>{fmtPct(acc)}</b></div>
                  <div><span>MAE</span><b>{fmt(mae, 2)} d</b></div>
                  <div><span>RMSE</span><b>{fmt(rmse, 2)} d</b></div>
                  <div><span>R²</span><b>{fmt(r2, 3)}</b></div>
                </div>
                <p className="info-note">
                  From the held-out test slice of the chronological 4-way purged
                  split. The model never saw these rows during tuning or selection.
                </p>
              </>
            )}
          </section>

          <section>
            <h3>Current snapshot · {props.activeBaseDate}</h3>
            <div className="info-grid">
              <div><span>Predicted cells</span><b>{props.predicted.length}</b></div>
              <div><span>Observed (FIRMS)</span><b>{props.observed.length}</b></div>
              <div>
                <span>Live (GISTDA)</span>
                <b>{props.liveFires.length > 0 ? props.liveFires.length : "see feed status"}</b>
              </div>
              <div><span>Province filter</span><b>{props.selectedProvince === "all" ? "all" : props.selectedProvince}</b></div>
            </div>
          </section>

          <section>
            <h3>Urgency distribution</h3>
            <div className="info-grid">
              <div><span style={{ color: "#dc2626" }}>CRITICAL</span><b>{tiers.CRITICAL}</b></div>
              <div><span style={{ color: "#ea580c" }}>HIGH</span><b>{tiers.HIGH}</b></div>
              <div><span style={{ color: "#f59e0b" }}>MEDIUM</span><b>{tiers.MEDIUM}</b></div>
              <div><span style={{ color: "#10b981" }}>LOW</span><b>{tiers.LOW}</b></div>
            </div>
            {props.thresholds && (
              <p className="info-note">
                {probabilityThresholds
                  ? <>Absolute cutoffs from metadata: HIGH ≥{" "}
                      {(props.thresholds.HIGH * 100).toFixed(2)}%; WATCH
                      (legacy MEDIUM alias) ≥{" "}
                      {(props.thresholds.MEDIUM * 100).toFixed(2)}%; lower
                      values are Background.</>
                  : <>Legacy tier cutoffs: HIGH {props.thresholds.HIGH};
                      MEDIUM {props.thresholds.MEDIUM}. See artifact semantics
                      before interpreting their units.</>}
              </p>
            )}
          </section>

          <section>
            <h3>Risk-tier distribution</h3>
            <div className="info-bars">
              {tierCounts.map(({ label, count }) => {
                const max = Math.max(...tierCounts.map((t) => t.count), 1);
                const pct = (count / max) * 100;
                return (
                  <div key={label} className="info-bar-row">
                    <span className="info-bar-label">{label}</span>
                    <div className="info-bar-track"><div className="info-bar-fill" style={{ width: `${pct}%` }} /></div>
                    <span className="info-bar-count">{count}</span>
                  </div>
                );
              })}
            </div>
          </section>

          {topProvinces.length > 0 && (
            <section>
              <h3>Top 5 provinces by predicted cells</h3>
              <table className="info-table">
                <tbody>
                  {topProvinces.map(([name, count]) => (
                    <tr key={name}>
                      <td>{name}</td>
                      <td><b>{count}</b></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </section>
          )}

          <section>
            <h3>Glossary — what these numbers mean</h3>
            <dl className="info-dl">
              {isBinary && (
                <>
                  <dt>ROC-AUC</dt>
                  <dd>Area under the ROC curve. Measures how well the model <b>ranks</b> cells by fire risk regardless of threshold. <b>0.5 = random, 1.0 = perfect, ≥0.8 is good</b>. Robust to class imbalance — the right headline for "rare event" tasks like imminent-fire prediction.</dd>

                  <dt>Average Precision</dt>
                  <dd>Area under the precision-recall curve. More demanding than ROC-AUC when positives are rare. <b>Higher = better</b>; equals the test-set positive rate for a random model.</dd>

                  <dt>Accuracy (binary)</dt>
                  <dd>
                    Share of cells where the model's yes/no answer matches
                    reality at a stated threshold. It is misleading on this
                    imbalanced test population: an always-negative baseline
                    {alwaysNegativeBaseline == null
                      ? " is already high"
                      : ` is already ${(alwaysNegativeBaseline * 100).toFixed(1)}%`}.
                  </dd>

                  <dt>Precision</dt>
                  <dd>Of cells the model flagged as imminent fire, the fraction that really did burn within the window. <b>Higher = fewer false alarms</b>. The operational threshold is selected on the calibration partition, then evaluated once on test.</dd>

                  <dt>Recall</dt>
                  <dd>Of all real imminent fires in the test set, the fraction the model caught. <b>Higher = fewer missed fires</b>. Recall and precision trade off — pick a lower probability threshold to raise recall (more flags, more false alarms).</dd>

                  <dt>F1 score</dt>
                  <dd>Harmonic mean of precision and recall. <b>Best F1</b> is the maximum F1 over all probability thresholds — a single-number summary that balances false alarms against missed fires.</dd>

                  <dt>Best threshold</dt>
                  <dd>The persisted HIGH cutoff is selected before test on the calibration partition. A lower cutoff normally increases review load and test-set recall; changing production still requires a reviewed calibration/evaluation run.</dd>

                  <dt>Precision @ top-K%</dt>
                  <dd>If you take the K% of cells with the highest predicted probability and treat them as your watch-list, what fraction actually burns? This measures concentration in a relative watch-list and is separate from the absolute HIGH/WATCH thresholds.</dd>
                </>
              )}
              {!isBinary && (
                <>
                  <dt>±1 day accuracy</dt>
                  <dd>Legacy-only: the share of test-set rows where an older date-regression model landed within ±1 day of the recorded date. The active binary model does not make this claim.</dd>

                  <dt>MAE (mean absolute error, days)</dt>
                  <dd>Average distance between predicted and real fire dates, in days. <b>Lower = better</b>. MAE = 1.5 means predictions are off by 1½ days on average. Robust to outliers.</dd>

                  <dt>RMSE (root mean squared error, days)</dt>
                  <dd>Like MAE but squares the errors before averaging, so big misses count more. <b>Lower = better</b>. If RMSE is much higher than MAE, a few large errors dominate the picture.</dd>

                  <dt>R² (coefficient of determination)</dt>
                  <dd>Legacy-only coefficient of determination. <b>0 = no better than predicting the mean, 1 = perfect, negative = worse than predicting the mean.</b> It is not used by the active 3-day classifier.</dd>
                </>
              )}

              <dt>Evaluation protocol</dt>
              <dd>Whole-date, chronological <b>4-way purged split</b> (train / validation / calibration / test) with a 7-day purge gap between each — not a random or 60/20/20 row split. Thresholds and calibration are chosen only on the calibration slice; the test slice is scored exactly once and never used to pick a threshold.</dd>

              <dt>Risk tiers (HIGH / WATCH / Background)</dt>
              <dd>Absolute cutoffs on calibrated probability, selected on the calibration partition. <b>HIGH</b> is the precision-oriented operating point; <b>WATCH</b> is the lower monitoring cutoff. Its recall target describes calibration-time selection, not guaranteed held-out or live recall. <b>MEDIUM</b> in stored data is the WATCH compatibility alias. A low-risk day can legitimately show zero HIGH/WATCH cells.</dd>

              <dt>Probability (the model's actual output)</dt>
              <dd>Calibrated P(≥1 FIRMS hotspot in this cell within the next 3 days). The model is a binary "fire within 3 days" classifier — it does <b>not</b> know whether a fire would land on day 1, 2, or 3.</dd>

              <dt>days_until_fire / predicted_fire_date (deprecated)</dt>
              <dd><b>Deprecated.</b> These fields may still appear in stored historical GeoJSON but are <b>not forecasts</b> and are no longer used by any chart, alert, export, or validation. The model supports exactly one temporal claim: "a hotspot within the 3-day risk window".</dd>

              <dt>Historical fires (last 30 days, FIRMS)</dt>
              <dd>Literal count of NASA FIRMS hotspot detections in the cell over the last 30 days. This is supporting context, not proof for or against a forecast; weather and seasonal features can still change current risk.</dd>

              <dt>Hit rate vs FIRMS (strict model target)</dt>
              <dd>For each matured historical prediction, did the <b>same model grid cell</b> record a FIRMS hotspot on any day in its risk window (base_date+1 … base_date+3)? This is the model's exact binary target — no spatial radius, no ±1-day slack. Any separate proximity view is context only and is never treated as model precision/recall.</dd>
            </dl>
          </section>

          <section>
            <h3>Data sources</h3>
            <ul className="info-list">
              <li><b>NASA FIRMS VIIRS NRT</b> — hotspot detections + FRP / brightness, used for training labels and historical aggregates.</li>
              <li><b>GISTDA NRT VIIRS + MODIS</b> — a separate Thai satellite-hotspot feed used as an optional live overlay. It is useful corroborating context, but not fully independent proof of FIRMS because both share satellite-observation limitations.</li>
              <li><b>Open-Meteo ERA5</b> — daily reanalysis (temp / precip / wind), used as features only when the weather cache is present.</li>
              <li><b>Hansen GFC v1.11</b> — per-cell tree cover baseline (2000) + recent loss %, distinguishes wildfire risk from agricultural-burn signal.</li>
            </ul>
            <p className="info-note">
              Observation and metric records come from named real sources. Missing
              live data is labelled unavailable; model preprocessing may still
              apply documented missing-value handling.
            </p>
          </section>

          <section>
            <h3>Selected filter</h3>
            <p className="info-note">
              Tier filter: <b>{props.selectedDay === "all" ? "all" : props.selectedDay}</b> · Province: <b>{props.selectedProvince === "all" ? "all" : props.selectedProvince}</b>
            </p>
          </section>
        </div>
      </div>
    </div>
  );
}
