import { useEffect, useState } from "react";
import { fetchHealth, type LiveStatus } from "../api";
import { useLang } from "../utils/i18n";

// Header badge showing real-time API health. Polls /health every 60s with a
// 4-second timeout — if the backend doesn't respond we degrade to "offline"
// gracefully. Pulse animation only on "live" so a frozen page doesn't lie.
//
// "stale" appears when the API is not ready, the served GeoJSON is old, or
// its latest source-data date is old (`ready`, `geojson_age_days`, and
// `data_stale_days` from the current /health schema).

const POLL_INTERVAL_MS = 60_000;

export default function LiveStatusBadge() {
  const { t } = useLang();
  // Neutral until the first health response. Starting at OFFLINE caused a
  // false red flash on every healthy page load.
  const [status, setStatus] = useState<LiveStatus | "checking">("checking");
  const [lastCheck, setLastCheck] = useState<Date | null>(null);

  useEffect(() => {
    let cancelled = false;
    const tick = async () => {
      const s = await fetchHealth();
      if (cancelled) return;
      setStatus(s);
      setLastCheck(new Date());
    };
    tick();
    const id = window.setInterval(tick, POLL_INTERVAL_MS);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, []);

  const label =
    status === "live" ? "API LIVE" :
    status === "stale" ? "API DEGRADED" :
    status === "checking" ? "API CHECK" : "API OFFLINE";
  const description =
    status === "live" ? t("livestatus.live") :
    status === "stale" ? t("livestatus.stale") :
    status === "checking" ? t("analytics.feed.status.loading") :
    t("livestatus.offline");

  return (
    <div
      className={`live-status ${status}`}
      title={`${description}\nLast check: ${lastCheck?.toLocaleTimeString() ?? "—"}`}
      aria-live="polite"
      aria-busy={status === "checking"}
    >
      <span className={`live-dot ${status}`} aria-hidden="true" />
      <span className="live-label">{label}</span>
    </div>
  );
}
