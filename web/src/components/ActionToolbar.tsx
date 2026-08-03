import { useState } from "react";
import type { ValidationMetrics } from "../types";
import { useLang } from "../utils/i18n";

interface Props {
  exportCount: number;
  onExport: () => void;
  metrics: ValidationMetrics | null;
  onShowDetails: () => void;
  onShowAlertSettings: () => void;
}

// Top-of-sidebar action bar — quick operator actions:
//   • Export the current province/tier selection as CSV. The parent owns the
//     canonical exporter so this button and the lower export button cannot
//     disagree about filters or CSV escaping.
//   • Copy the current route URL (filters are intentionally not claimed to be
//     encoded in the link)
//   • Explain the persisted HIGH/WATCH operating points
//   • Open the technical details modal
export default function ActionToolbar({
  exportCount, onExport, metrics, onShowDetails, onShowAlertSettings,
}: Props) {
  const { t } = useLang();
  const [shareStatus, setShareStatus] = useState<string | null>(null);

  const handleShare = async () => {
    const url = window.location.href;
    try {
      if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(url);
      } else {
        // Clipboard API is restricted to secure contexts in several browsers;
        // the local demo commonly runs on http://0.0.0.0. Keep a scoped legacy
        // fallback so copying the page link still works there.
        const input = document.createElement("textarea");
        input.value = url;
        input.setAttribute("readonly", "");
        input.style.position = "fixed";
        input.style.opacity = "0";
        document.body.appendChild(input);
        let copied = false;
        try {
          input.select();
          copied = document.execCommand("copy");
        } finally {
          input.remove();
        }
        if (!copied) throw new Error("Copy command was rejected");
      }
      setShareStatus(`✓ ${t("sidebar.copied")}`);
      setTimeout(() => setShareStatus(null), 2000);
    } catch {
      setShareStatus(`❌ ${t("sidebar.copy_failed")}`);
      setTimeout(() => setShareStatus(null), 2000);
    }
  };

  const isCalibrated =
    metrics?.calibration_method != null ||
    metrics?.threshold_selected_on === "calib";

  return (
    <div className="action-toolbar">
      <button
        className="action-btn"
        type="button"
        onClick={onExport}
        disabled={exportCount === 0}
        title={exportCount > 0 ? `Export ${exportCount} selected cells as CSV` : "ไม่มี cells ให้ export"}
      >
        <span className="action-btn-icon">📥</span>
        <span>Export CSV</span>
      </button>
      <button
        className="action-btn"
        type="button"
        onClick={handleShare}
        title={t("sidebar.copy_page_link")}
      >
        <span className="action-btn-icon">🔗</span>
        <span>{shareStatus ?? t("btn.share")}</span>
      </button>
      <button
        className="action-btn"
        type="button"
        onClick={onShowAlertSettings}
        title="ดูเกณฑ์ HIGH / WATCH ที่บันทึกไว้"
        disabled={!isCalibrated}
      >
        <span className="action-btn-icon">⚙️</span>
        <span>Threshold</span>
      </button>
      <button
        className="action-btn"
        type="button"
        onClick={onShowDetails}
        title="ดู technical metrics + dataset details"
      >
        <span className="action-btn-icon">📊</span>
        <span>Details</span>
      </button>
    </div>
  );
}
