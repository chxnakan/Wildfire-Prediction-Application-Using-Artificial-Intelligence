import { useEffect, useMemo } from "react";
import type {
  FireFeature,
  ThresholdReport,
  ValidationMetrics,
} from "../types";
import { useLang } from "../utils/i18n";

interface Props {
  open: boolean;
  onClose: () => void;
  predicted: FireFeature[];
  metrics: ValidationMetrics | null;
}

interface OperatingPoint {
  name: "HIGH" | "WATCH";
  purpose: string;
  threshold: number | null;
  report: ThresholdReport | null;
  visibleCells: number;
}

const pct = (value: number | null | undefined): string =>
  typeof value === "number" && Number.isFinite(value)
    ? `${(value * 100).toFixed(2)}%`
    : "—";

/**
 * Read-only explanation of the two persisted operating points.
 *
 * This used to be an arbitrary threshold simulator. It inverted the legacy
 * pseudo-day field with the wrong formula and inferred precision/recall from
 * reliability-bin centres, producing numbers that were not measured metrics.
 * Operational thresholds now remain exactly as trained and persisted.
 */
export default function AlertSettings({
  open,
  onClose,
  predicted,
  metrics,
}: Props) {
  const { lang } = useLang();
  const copy = lang === "th"
    ? {
        title: "เกณฑ์ใช้งานจากโมเดล",
        intro:
          "หน้านี้เป็นข้อมูลแบบอ่านอย่างเดียว เกณฑ์ HIGH และ WATCH มาจาก artifact ที่ผ่านการ train ไม่ได้เปลี่ยนแผนที่หรือคำนวณ metric ใหม่ใน browser",
        purposeHigh: "แจ้งเตือนเร่งด่วน เน้น precision",
        purposeWatch: "รายการเฝ้าระวัง เน้น recall",
        threshold: "เกณฑ์ probability",
        cells: "cell ในมุมมองปัจจุบัน",
        precision: "Held-out precision",
        recall: "Held-out recall",
        rate: "Held-out alert rate",
        evidence: "ฐานประเมิน",
        unavailable: "artifact นี้ไม่มี operating point ดังกล่าว",
        background: "Background ในมุมมองปัจจุบัน",
        drift:
          "มี seasonal prevalence drift: ค่า held-out ด้านล่างจึงต่างจาก calibration period มาก โปรดอ่านคู่กับ drift note บนหน้าหลัก",
        fixed:
          "การเปลี่ยนเกณฑ์จริงต้องผ่าน calibration, held-out evaluation และ model-promotion gate — browser ไม่ควรสร้างค่าที่ดูดีขึ้นเอง",
        close: "ปิด",
      }
    : {
        title: "Persisted model operating points",
        intro:
          "This is a read-only explanation. HIGH and WATCH come from the trained artifact; this browser does not change the map or invent new metrics.",
        purposeHigh: "Urgent, precision-oriented alert",
        purposeWatch: "Recall-oriented monitoring list",
        threshold: "Probability cutoff",
        cells: "Cells in current view",
        precision: "Held-out precision",
        recall: "Held-out recall",
        rate: "Held-out alert rate",
        evidence: "Evaluation basis",
        unavailable: "This operating point is not present in the artifact",
        background: "Background in current view",
        drift:
          "Seasonal prevalence drift is flagged, so these held-out results differ materially from the calibration period. Read them with the drift note on the map.",
        fixed:
          "A real threshold change requires calibration, held-out evaluation, and the model-promotion gate. The browser must not create a better-looking operating point.",
        close: "Close",
      };

  useEffect(() => {
    if (!open) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  const counts = useMemo(() => {
    let high = 0;
    let watch = 0;
    let background = 0;
    for (const feature of predicted) {
      const level = feature.properties.urgency_level;
      if (level === "HIGH" || level === "CRITICAL") high += 1;
      else if (level === "MEDIUM") watch += 1;
      else if (level === "LOW") background += 1;
    }
    return { high, watch, background };
  }, [predicted]);

  if (!open) return null;

  const highReport = metrics?.high_test_metrics ?? (
    metrics?.deployment_threshold == null
      ? null
      : {
          threshold: metrics.deployment_threshold,
          precision: metrics.deployment_precision ?? NaN,
          recall: metrics.deployment_recall ?? NaN,
          f1: metrics.deployment_f1 ?? NaN,
          alert_rate: NaN,
        }
  );
  const operatingPoints: OperatingPoint[] = [
    {
      name: "HIGH",
      purpose: copy.purposeHigh,
      threshold:
        metrics?.high_test_metrics?.threshold ??
        metrics?.deployment_threshold ??
        null,
      report: highReport,
      visibleCells: counts.high,
    },
    {
      name: "WATCH",
      purpose: copy.purposeWatch,
      threshold:
        metrics?.watch_test_metrics?.threshold ??
        metrics?.watch_threshold ??
        null,
      report: metrics?.watch_test_metrics ?? null,
      visibleCells: counts.watch,
    },
  ];

  return (
    <div className="info-modal-backdrop" onClick={onClose}>
      <div
        className="info-modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="operating-points-title"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="info-modal-header">
          <h2 id="operating-points-title">⚙️ {copy.title}</h2>
          <button
            type="button"
            className="info-modal-close"
            onClick={onClose}
            aria-label={copy.close}
          >
            ×
          </button>
        </div>

        <div className="info-modal-body">
          <p style={{ color: "var(--text-2)", lineHeight: 1.6 }}>
            {copy.intro}
          </p>

          <div className="threshold-operating-grid">
            {operatingPoints.map((point) => (
              <section className="threshold-operating-card" key={point.name}>
                <h3>{point.name}</h3>
                <p>{point.purpose}</p>
                {point.threshold == null ? (
                  <div className="info-note">{copy.unavailable}</div>
                ) : (
                  <dl>
                    <div>
                      <dt>{copy.threshold}</dt>
                      <dd>{point.threshold.toFixed(4)}</dd>
                    </div>
                    <div>
                      <dt>{copy.cells}</dt>
                      <dd>{point.visibleCells.toLocaleString()}</dd>
                    </div>
                    <div>
                      <dt>{copy.precision}</dt>
                      <dd>{pct(point.report?.precision)}</dd>
                    </div>
                    <div>
                      <dt>{copy.recall}</dt>
                      <dd>{pct(point.report?.recall)}</dd>
                    </div>
                    <div>
                      <dt>{copy.rate}</dt>
                      <dd>{pct(point.report?.alert_rate)}</dd>
                    </div>
                    <div>
                      <dt>{copy.evidence}</dt>
                      <dd>
                        {point.report?.n != null
                          ? `${point.report.n.toLocaleString()} cell-days`
                          : "—"}
                      </dd>
                    </div>
                  </dl>
                )}
              </section>
            ))}
          </div>

          <p className="info-note">
            {copy.background}: <b>{counts.background.toLocaleString()}</b>
          </p>
          {metrics?.prevalence_drift?.drift_flag && (
            <p className="info-note drift">{copy.drift}</p>
          )}
          <p className="info-note">{copy.fixed}</p>
        </div>
      </div>
    </div>
  );
}
