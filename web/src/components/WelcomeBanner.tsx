import { useEffect, useState } from "react";
import { useLang } from "../utils/i18n";

// Lightweight first-visit welcome / legend banner.
//
// Shows once per browser (localStorage flag). Dismissable with X. Explains
// the two main marker colours and points the new operator to the docs.
// Avoids loading a heavyweight onboarding library (Intro.js, Shepherd) —
// for a 5-line banner that's overkill.
const STORAGE_KEY = "firewatch.welcomeDismissed.v4";

export default function WelcomeBanner() {
  const { t } = useLang();
  const [show, setShow] = useState(false);

  useEffect(() => {
    try {
      if (!window.localStorage.getItem(STORAGE_KEY)) setShow(true);
    } catch { /* storage disabled */ }
  }, []);

  const dismiss = () => {
    setShow(false);
    try { window.localStorage.setItem(STORAGE_KEY, "1"); } catch { /* */ }
  };

  if (!show) return null;
  return (
    <div
      className="welcome-banner"
      role="dialog"
      aria-label={t("welcome.aria")}
      aria-describedby="firewatch-welcome-description"
    >
      <div className="welcome-banner-icon" aria-hidden="true">👋</div>
      <div className="welcome-banner-body">
        <div className="welcome-banner-title">{t("welcome.title")}</div>
        <div className="welcome-banner-text" id="firewatch-welcome-description">
          <span>
            <span className="welcome-dot" style={{ background: "#10b981" }} />
            {t("welcome.model_risk")}
          </span>
          <span>
            <span className="welcome-dot" style={{ background: "#06b6d4" }} />
            {t("welcome.live_fires")}
          </span>
          <span style={{ color: "var(--text-3)" }}>{t("welcome.hint")}</span>
        </div>
      </div>
      <button
        type="button"
        className="welcome-banner-close"
        onClick={dismiss}
        aria-label={t("welcome.close")}
        title={t("welcome.close_forever")}
      >×</button>
    </div>
  );
}
