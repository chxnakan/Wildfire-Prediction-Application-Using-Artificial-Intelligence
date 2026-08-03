import {
  lazy,
  Suspense,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { fetchGeoJson } from "./api";
import AlertSettings from "./components/AlertSettings";
import AlertToasts from "./components/AlertToasts";
import InfoModal from "./components/InfoModal";
import LiveFiresPage from "./components/LiveFiresPage";
import LiveStatusBadge from "./components/LiveStatusBadge";
import MapView from "./components/MapView";
import NavTabs from "./components/NavTabs";
import Sidebar from "./components/Sidebar";
import ThemeToggle from "./components/ThemeToggle";
import WelcomeBanner from "./components/WelcomeBanner";
import { useFireAlerts, useFireSseStream } from "./utils/fireAlerts";
import { useHashRoute } from "./utils/hashRoute";
import { isActionableUrgency, matchesTierFilter } from "./utils/risk";
import {
  isVeryLowProbability,
  readProbability,
} from "./utils/probability";
import { usePersistedState } from "./utils/usePersistedState";
import type {
  AlertPageRoute,
  DaySelection,
  DisplayOptions,
  FireGeoJson,
  GistdaFeature,
  LiveFireMeta,
} from "./types";
import { exportCellsCsv } from "./utils/csv";
import { fetchLiveFires, LIVE_REFRESH_MS } from "./utils/gistda";
import { isInThailandBbox } from "./constants";
import LanguageToggle from "./components/LanguageToggle";
import { fmtTr, LanguageContext, makeT, useLang, type LangCtx as LangCtxValue, type Lang } from "./utils/i18n";

// Keep the default map/dashboard eager for the fastest operator landing view.
// Heavier secondary routes load only when their tab is opened.
const ComparePage = lazy(() => import("./components/ComparePage"));
const HotspotAnalyticsPage = lazy(() => import("./components/HotspotAnalyticsPage"));
const NotifyPage = lazy(() => import("./components/NotifyPage"));
const ReportsPage = lazy(() => import("./components/ReportsPage"));

function RouteLoadingFallback() {
  return (
    <main
      role="status"
      aria-live="polite"
      style={{ padding: 40, color: "var(--text-3)" }}
    >
      Loading page…
    </main>
  );
}

const DEFAULT_OPTIONS: DisplayOptions = {
  showObserved: false,
  // ON by default so real-time fire alerts (useFireAlerts) work out of
  // the box — operator opens the dashboard and starts getting alerts
  // within minutes of new GISTDA detections.
  showLiveFires: true,
  // showPredicted + showCellPins are always-on now (Sidebar doesn't surface
  // toggles for them). Keep heatRadius hard-coded at 33 to match the new
  // frontend default.
  showPredicted: true,
  showCellPins: true,
  // Start in presentation/operator focus mode: the latest snapshot currently
  // contains many genuine but visually unhelpful <0.01% cells. The sidebar
  // states the hidden count and lets users restore them; CSV keeps all cells.
  hideVeryLowRisk: true,
  heatRadius: 33,
};

export default function App() {
  const [storedLang, setLang] = usePersistedState<Lang>("firewatch:lang", "en");
  // localStorage is user-editable and may contain an old or malformed value.
  // Normalise before indexing DICT so one bad preference cannot white-screen
  // the entire application.
  const lang: Lang = storedLang === "th" ? "th" : "en";
  const langCtx = useMemo<LangCtxValue>(() => ({
    lang, setLang, t: makeT(lang),
  }), [lang, setLang]);

  return (
    <LanguageContext.Provider value={langCtx}>
      <AppInner />
    </LanguageContext.Provider>
  );
}

function AppInner() {
  const { t } = useLang();
  const [geojson, setGeojson] = useState<FireGeoJson | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [geojsonLoading, setGeojsonLoading] = useState(true);
  const geojsonRequestRef = useRef(0);

  const [selectedBaseDate, setSelectedBaseDate] = useState<string>("latest");
  const [selectedProvince, setSelectedProvince] = useState<string>("all");
  const [selectedDay, setSelectedDay] = useState<DaySelection>("all");
  const [options, setOptions] = useState<DisplayOptions>(DEFAULT_OPTIONS);

  // GISTDA live OBSERVED hotspots (not predictions) — served from OUR
  // backend cache (/live-hotspots). Refreshes every 60 s while the toggle
  // is on; the backend itself polls GISTDA every ~5 min.
  const [liveFires, setLiveFires] = useState<GistdaFeature[]>([]);
  const [liveFireMeta, setLiveFireMeta] = useState<LiveFireMeta>({
    status: "idle",
    count: 0,
    lastFetch: null,
    error: null,
  });
  const liveFireTimerRef = useRef<number | null>(null);
  const liveFireAbortRef = useRef<AbortController | null>(null);

  // Info modal visibility
  const [infoModalOpen, setInfoModalOpen] = useState(false);
  const [alertSettingsOpen, setAlertSettingsOpen] = useState(false);
  // Mobile sidebar drawer (desktop ignores this — sidebar always visible)
  const [sidebarOpen, setSidebarOpen] = useState(false);
  // Desktop sidebar collapse — persists across visits via localStorage so a
  // returning user keeps the layout they chose. Mobile uses sidebarOpen instead.
  const [sidebarCollapsed, setSidebarCollapsed] = usePersistedState<boolean>(
    "firewatch.sidebarCollapsed", false,
  );
  // Ctrl/Cmd+B toggles desktop sidebar collapse (VS Code convention)
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "b") {
        e.preventDefault();
        if (window.innerWidth >= 768) {
          setSidebarCollapsed((v) => !v);
        } else {
          setSidebarOpen((v) => !v);
        }
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [setSidebarCollapsed]);

  // Hash-based routing between Dashboard / Notify / Reports
  const [route, navigate] = useHashRoute();
  // Map visibility and feed availability are separate concerns. Hiding the
  // cyan overlay must not make the dedicated Live/Analytics routes empty.
  const liveFeedNeeded =
    options.showLiveFires || route === "live" || route === "analytics";
  // Close mobile drawer whenever the route changes
  useEffect(() => { setSidebarOpen(false); }, [route]);

  // ── Real-time fire alerts ──
  // MUST be called before any early-return below — React's Rules of Hooks
  // require hooks to run in the same order every render. Putting this after
  // `if (!geojson) return …` made the first render skip the hook entirely
  // and crashed the page with a "Rendered more hooks than during the
  // previous render" white-screen on data load.
  const fireAlerts = useFireAlerts(liveFires);

  // ── Server-Sent Events stream from FastAPI ──
  // Backend polls GISTDA every ~5 min server-side and pushes new fires; we react below
  // (after refreshLiveFires is declared) by triggering an immediate refresh
  // so the alert toast fires within seconds instead of waiting 5 min for
  // the next browser-side poll cycle.
  const sse = useFireSseStream();
  const lastSseFireRef = useRef<number>(0);

  const refreshLiveFires = useCallback(async () => {
    // Abort any in-flight fetch so a quick toggle on/off doesn't pile up requests.
    liveFireAbortRef.current?.abort();
    const controller = new AbortController();
    liveFireAbortRef.current = controller;
    setLiveFireMeta((m) => ({ ...m, status: "loading", error: null }));
    try {
      const result = await fetchLiveFires(controller.signal);
      const inThailand = result.features.filter((f) => {
        const lat = Number(f.attributes?.latitude);
        const lon = Number(f.attributes?.longitude);
        if (!Number.isFinite(lat) || !Number.isFinite(lon)) return false;
        return isInThailandBbox(lat, lon);
      });
      setLiveFires(inThailand);
      setLiveFireMeta({
        status: "ok",
        count: inThailand.length,
        lastFetch: new Date(),
        error: null,
        isStale: result.metadata.isStale,
        lastSuccessAt: result.metadata.lastSuccessAt,
      });
    } catch (err) {
      if (controller.signal.aborted) return;
      const msg = err instanceof Error ? err.message : String(err);
      // Retain the last successful hotspot list during transient failures —
      // an old map beats an empty one; the stale flag drives the UI hint.
      setLiveFireMeta((m) => ({ ...m, status: "error", error: msg, isStale: true }));
    }
  }, []);

  // SSE → immediate refresh: when the backend pushes a new fire event,
  // re-fetch GISTDA so useFireAlerts diffs the new payload + toasts fire
  // within ~1-2 seconds of detection (vs 5 min polling worst case).
  useEffect(() => {
    if (
      liveFeedNeeded &&
      sse.lastFireAt &&
      sse.lastFireAt !== lastSseFireRef.current
    ) {
      lastSseFireRef.current = sse.lastFireAt;
      refreshLiveFires();
    }
  }, [sse.lastFireAt, liveFeedNeeded, refreshLiveFires]);

  // Feed demand drives fetch + auto-refresh lifecycle. The map toggle controls
  // demand on the dashboard; dedicated live-data routes always need the feed.
  useEffect(() => {
    if (!liveFeedNeeded) {
      if (liveFireTimerRef.current != null) {
        window.clearTimeout(liveFireTimerRef.current);
        liveFireTimerRef.current = null;
      }
      liveFireAbortRef.current?.abort();
      setLiveFires([]);
      setLiveFireMeta({ status: "idle", count: 0, lastFetch: null, error: null });
      return;
    }
    let cancelled = false;
    const tick = async () => {
      await refreshLiveFires();
      if (cancelled || !liveFeedNeeded) return;
      liveFireTimerRef.current = window.setTimeout(tick, LIVE_REFRESH_MS);
    };
    tick();
    return () => {
      cancelled = true;
      liveFireAbortRef.current?.abort();
      if (liveFireTimerRef.current != null) {
        window.clearTimeout(liveFireTimerRef.current);
        liveFireTimerRef.current = null;
      }
    };
  }, [liveFeedNeeded, refreshLiveFires]);

  // Load GeoJSON once on mount, with an operator-visible retry path. A request
  // id prevents a slow earlier attempt from overwriting a successful retry.
  const loadGeoJson = useCallback(async () => {
    const requestId = ++geojsonRequestRef.current;
    setGeojsonLoading(true);
    setError(null);
    try {
      const payload = await fetchGeoJson();
      if (requestId !== geojsonRequestRef.current) return;
      setGeojson(payload);
    } catch (e) {
      if (requestId !== geojsonRequestRef.current) return;
      console.error(e);
      setError(e instanceof Error ? e.message : "Prediction API did not respond.");
    } finally {
      if (requestId === geojsonRequestRef.current) setGeojsonLoading(false);
    }
  }, []);

  useEffect(() => {
    loadGeoJson();
  }, [loadGeoJson]);

  const derived = useMemo(() => {
    if (!geojson) return null;
    // Two-stage Thailand filter:
    //   1. Bbox — fast first pass, drops obvious foreign cells.
    //   2. Province presence — for PREDICTED features only, also require a
    //      non-empty `province` field. risk_map.py only sets that when
    //      find_province() returns a Thai province for the cell (inside the
    //      77-province polygon). Bbox-alone leaks into S.Myanmar / N.Laos
    //      etc; province filter drops those.
    //   Observed FIRMS features keep just the bbox check — they're real
    //   detections and the user might still want to see cross-border ones
    //   for context (the Live Fires page has its own TH-only toggle).
    const rawFeatures = (geojson.features ?? []).filter((f) => {
      const coordinates = f?.geometry?.coordinates;
      if (!Array.isArray(coordinates) || coordinates.length < 2) return false;
      const lon = Number(coordinates[0]);
      const lat = Number(coordinates[1]);
      return Number.isFinite(lat) &&
        Number.isFinite(lon) &&
        isInThailandBbox(lat, lon);
    });
    const features = rawFeatures.filter((f) => {
      const prov = (f.properties.province ?? "").trim();
      // Predicted cells: require province annotation (drops S.Myanmar/N.Laos leaks
      // that fall inside the BBOX but outside Thailand's polygon).
      if (f.properties.source === "predicted") return prov.length > 0;
      // Observed FIRMS: bbox is sufficient — real hotspot coordinates are already
      // inside Thailand. Province annotation is present on newer snapshots
      // (risk_map.py ≥2026-05-26) but absent on older ones. Dropping province-less
      // observations would blank out the Compare page and Live Fires observed layer.
      return true;
    });

    const observed = features.filter((f) => f.properties.source === "observed");
    const predictedAll = features.filter(
      (f) => f.properties.source === "predicted"
    );

    const allBaseDates = [
      ...new Set(predictedAll.map((f) => f.properties.base_date).filter(Boolean) as string[]),
    ].sort();
    const latestBaseDate = allBaseDates[allBaseDates.length - 1] ?? "N/A";

    let activeBaseDate = selectedBaseDate;
    if (
      activeBaseDate === "latest" ||
      !allBaseDates.includes(activeBaseDate)
    ) {
      activeBaseDate = latestBaseDate;
    }

    const snapshotPredicted = predictedAll.filter(
      (f) => f.properties.base_date === activeBaseDate
    );

    const provinceSet = new Set(
      snapshotPredicted
        .map((f) => (f.properties.province ?? "").trim())
        .filter(Boolean)
    );
    const provinces = [...provinceSet].sort();

    let provinceFiltered = snapshotPredicted;
    let resolvedProvince = selectedProvince;
    if (selectedProvince !== "all") {
      if (provinceSet.has(selectedProvince)) {
        provinceFiltered = snapshotPredicted.filter(
          (f) => f.properties.province === selectedProvince
        );
      } else {
        // Snapshot doesn't contain the picked province — report this so the
        // App effect can reset state (next render).
        resolvedProvince = "all";
      }
    }

    // Risk-tier filter (High / Watch / Background) — replaces the former
    // day-0..7 pseudo-date buckets. Never a date filter.
    const dayFiltered = provinceFiltered.filter(
      (f) => matchesTierFilter(f.properties.urgency_level, selectedDay)
    );

    const mapPredicted = options.hideVeryLowRisk
      ? dayFiltered.filter(
          (feature) =>
            !isVeryLowProbability(readProbability(feature.properties)),
        )
      : dayFiltered;
    const hiddenVeryLowCount = dayFiltered.length - mapPredicted.length;

    // Status message — the focus filter changes only what the map draws.
    // Export keeps the complete tier/province selection.
    const daySelectorMessage = options.hideVeryLowRisk
      ? fmtTr(t("sidebar.showing.very_low_hidden"), {
          visible: mapPredicted.length,
          hidden: hiddenVeryLowCount,
        })
      : selectedDay === "all"
        ? fmtTr(t("sidebar.showing.all"), { n: provinceFiltered.length })
        : fmtTr(t("sidebar.showing.tier"), {
            n: dayFiltered.length,
            tier: selectedDay,
          });

    return {
      observed,
      predictedAll,                  // for grid-size detection
      snapshotPredicted: provinceFiltered, // for sidebar urgency / timeline / landcover
      selectedPredicted: dayFiltered,      // full selection for CSV export
      visiblePredicted: mapPredicted,      // map after optional focus filter
      hiddenVeryLowCount,
      allBaseDates,
      latestBaseDate,
      activeBaseDate,
      provinces,
      resolvedProvince,
      daySelectorMessage,
    };
  }, [
    geojson,
    selectedBaseDate,
    selectedProvince,
    selectedDay,
    options.hideVeryLowRisk,
    t,
  ]);

  // If the active province got reset because the snapshot dropped it, sync
  // the controlled state so the dropdown reflects "all".
  useEffect(() => {
    if (derived && derived.resolvedProvince !== selectedProvince) {
      setSelectedProvince(derived.resolvedProvince);
    }
  }, [derived, selectedProvince]);

  // If the active tier went empty in this snapshot, fall back to "all".
  useEffect(() => {
    if (!derived || selectedDay === "all") return;
    const count = derived.snapshotPredicted.filter(
      (f) => matchesTierFilter(f.properties.urgency_level, selectedDay)
    ).length;
    if (count === 0) setSelectedDay("all");
  }, [derived, selectedDay]);

  const onExportCsv = () => {
    if (!derived || !derived.selectedPredicted.length) {
      alert("Nothing to export — current filter has no cells.");
      return;
    }
    exportCellsCsv(
      derived.selectedPredicted,
      derived.activeBaseDate,
      selectedProvince,
      selectedDay
    );
  };

  const onBaseDateChange = (v: string) => {
    setSelectedBaseDate(v);
    // Reset day filter so the new snapshot's full prediction set is visible.
    setSelectedDay("all");
  };

  const meta = geojson?.metadata ?? {};

  // Keep the application shell and data-independent routes available even
  // when the prediction snapshot is loading or unavailable. Previously an
  // early full-page return removed navigation entirely, so a /#live or
  // /#analytics route could not show its independently fetched evidence.
  const actionableAlertCount = derived
    ? derived.predictedAll.filter(
        (f) =>
          f.properties.base_date === derived.latestBaseDate &&
          isActionableUrgency(f.properties.urgency_level),
      ).length
    : 0;
  const fireToasts = (
    <AlertToasts
      alerts={fireAlerts.activeAlerts}
      onDismiss={fireAlerts.dismissAlert}
      onDismissAll={fireAlerts.dismissAll}
      onFlyTo={(lat, lon) => {
        if (route !== "dashboard") navigate("dashboard");
        window.dispatchEvent(new CustomEvent("firewatch:flyto", { detail: { lat, lon } }));
      }}
    />
  );
  // The quick guide explains map layers, so keep it on the map route only.
  // Showing the fixed banner over Reports/Compare/Analytics obscured their
  // headline controls on a first visit.
  const welcomeBanner = route === "dashboard" ? <WelcomeBanner /> : null;

  if (route === "live") {
    return (
      <>
        <TopBar
          route={route}
          onNavigate={navigate}
          actionableAlertCount={actionableAlertCount}
          showSidebarToggle={false}
        />
        <LiveFiresPage
          liveFires={liveFires}
          observed={derived?.observed ?? []}
          observedAvailable={derived !== null}
          liveFireMeta={liveFireMeta}
          onRefresh={refreshLiveFires}
          onNavigateToMap={(lat, lon) => {
            navigate("dashboard");
            // Defer to next tick so MapView mounts before flyTo fires.
            setTimeout(() => {
              window.dispatchEvent(
                new CustomEvent("firewatch:flyto", { detail: { lat, lon } }),
              );
            }, 50);
          }}
        />
        {welcomeBanner}
        {fireToasts}
      </>
    );
  }

  if (route === "analytics") {
    return (
      <>
        <TopBar
          route={route}
          onNavigate={navigate}
          actionableAlertCount={actionableAlertCount}
          showSidebarToggle={false}
        />
        <Suspense fallback={<RouteLoadingFallback />}>
          <HotspotAnalyticsPage
            liveFires={liveFires}
            liveFireMeta={liveFireMeta}
          />
        </Suspense>
        {welcomeBanner}
        {fireToasts}
      </>
    );
  }

  if (error) {
    return (
      <>
        <TopBar
          route={route}
          onNavigate={navigate}
          actionableAlertCount={actionableAlertCount}
          showSidebarToggle={false}
        />
        <main
          role="alert"
          style={{ padding: "40px", color: "var(--text, #f5f5f5)" }}
        >
          <h2 style={{ color: "#f87171" }}>Prediction data unavailable</h2>
          <p>
            The model snapshot could not be loaded, so prediction-dependent
            pages show no replacement or placeholder risk values. Live Fires
            and Analytics remain available from the navigation above.
          </p>
          <p style={{ color: "var(--text-3, #a0a3aa)", fontSize: 12 }}>
            {error}
          </p>
          <button
            type="button"
            onClick={loadGeoJson}
            disabled={geojsonLoading}
            style={{
              marginTop: 8,
              padding: "8px 14px",
              color: "inherit",
              background: "var(--surface-2, #17181d)",
              border: "1px solid var(--border, #2a2b31)",
              borderRadius: 6,
              cursor: geojsonLoading ? "wait" : "pointer",
            }}
          >
            {geojsonLoading ? "Retrying…" : "Try again"}
          </button>
        </main>
        {fireToasts}
      </>
    );
  }

  if (geojsonLoading && !geojson) {
    return (
      <>
        <TopBar
          route={route}
          onNavigate={navigate}
          actionableAlertCount={actionableAlertCount}
          showSidebarToggle={false}
        />
        <main
          role="status"
          aria-live="polite"
          style={{ padding: "40px", color: "#a0a3aa" }}
        >
          Loading fire prediction data…
        </main>
        {fireToasts}
      </>
    );
  }

  if (!geojson || !derived) {
    return (
      <>
        <TopBar
          route={route}
          onNavigate={navigate}
          actionableAlertCount={actionableAlertCount}
          showSidebarToggle={false}
        />
        <main role="status" style={{ padding: "40px", color: "#a0a3aa" }}>
          No scored prediction snapshot is available.
        </main>
        {fireToasts}
      </>
    );
  }

  // ── Render route-specific content ──
  if (route === "notify") {
    return (
      <>
        <TopBar
          route={route}
          onNavigate={navigate}
          actionableAlertCount={actionableAlertCount}
          showSidebarToggle={false}
        />
        <Suspense fallback={<RouteLoadingFallback />}>
          <NotifyPage predictedAll={derived.predictedAll} />
        </Suspense>
        {welcomeBanner}
        {fireToasts}
      </>
    );
  }

  if (route === "compare") {
    return (
      <>
        <TopBar
          route={route}
          onNavigate={navigate}
          actionableAlertCount={actionableAlertCount}
          showSidebarToggle={false}
        />
        <Suspense fallback={<RouteLoadingFallback />}>
          <ComparePage
            allFeatures={derived.predictedAll}
            observedFeatures={derived.observed}
            metrics={meta.metrics ?? null}
          />
        </Suspense>
        {welcomeBanner}
        {fireToasts}
      </>
    );
  }

  if (route === "reports") {
    return (
      <>
        <TopBar
          route={route}
          onNavigate={navigate}
          actionableAlertCount={actionableAlertCount}
          showSidebarToggle={false}
        />
        <Suspense fallback={<RouteLoadingFallback />}>
          <ReportsPage
            metrics={meta.metrics ?? null}
            predictedAll={derived.predictedAll}
          />
        </Suspense>
        {welcomeBanner}
        {fireToasts}
      </>
    );
  }

  // Dashboard (default)
  return (
    <>
      <TopBar
        route={route}
        onNavigate={navigate}
        actionableAlertCount={actionableAlertCount}
        showSidebarToggle={true}
        sidebarOpen={sidebarOpen}
        onToggleSidebar={() => setSidebarOpen((v) => !v)}
        sidebarCollapsed={sidebarCollapsed}
        onToggleSidebarCollapsed={() => setSidebarCollapsed((v) => !v)}
      />
      {sidebarOpen && (
        <div
          className="sidebar-backdrop"
          onClick={() => setSidebarOpen(false)}
          aria-hidden="true"
        />
      )}
      <div
        id="sidebar-wrap"
        className={`${sidebarOpen ? "open" : ""}${sidebarCollapsed ? " desktop-collapsed" : ""}`}
      >
      <Sidebar
        activeBaseDate={derived.activeBaseDate}
        allBaseDates={derived.allBaseDates}
        selectedBaseDate={selectedBaseDate}
        onBaseDateChange={onBaseDateChange}
        provinces={derived.provinces}
        selectedProvince={selectedProvince}
        onProvinceChange={setSelectedProvince}
        selectedDay={selectedDay}
        onDayChange={setSelectedDay}
        predicted={derived.snapshotPredicted}
        predictedAll={derived.predictedAll}
        visibleCount={
          options.showPredicted ? derived.visiblePredicted.length : 0
        }
        hiddenVeryLowCount={derived.hiddenVeryLowCount}
        daySelectorMessage={
          options.showPredicted
            ? derived.daySelectorMessage
            : "Model risk layer is hidden; filters and CSV data are unchanged."
        }
        thresholds={meta.urgency_thresholds ?? null}
        metrics={meta.metrics ?? null}
        metadata={meta}
        options={options}
        onOptionsChange={(o) => setOptions((prev) => ({ ...prev, ...o }))}
        onExportCsv={onExportCsv}
        liveFireMeta={liveFireMeta}
        onShowInfoModal={() => setInfoModalOpen(true)}
        onShowAlertSettings={() => setAlertSettingsOpen(true)}
      />
      </div>

      <MapView
        observed={derived.observed}
        predictedAll={derived.predictedAll}
        predictedVisible={derived.visiblePredicted}
        liveFires={liveFires}
        thresholds={meta.urgency_thresholds ?? null}
        options={options}
      />

      <Legend
        showPredicted={options.showPredicted}
        showCritical={
          derived.snapshotPredicted.some(
            (feature) => feature.properties.urgency_level === "CRITICAL",
          ) ||
          (
            typeof meta.urgency_thresholds?.CRITICAL === "number" &&
            meta.urgency_thresholds.CRITICAL <= 1
          )
        }
        showObserved={options.showObserved}
        showLiveFire={options.showLiveFires && liveFires.length > 0}
      />

      <InfoModal
        open={infoModalOpen}
        onClose={() => setInfoModalOpen(false)}
        activeBaseDate={derived.activeBaseDate}
        predicted={derived.snapshotPredicted}
        observed={derived.observed}
        liveFires={liveFires}
        metrics={meta.metrics ?? null}
        thresholds={meta.urgency_thresholds ?? null}
        metadata={meta}
        selectedProvince={selectedProvince}
        selectedDay={selectedDay}
      />

      <AlertSettings
        open={alertSettingsOpen}
        onClose={() => setAlertSettingsOpen(false)}
        predicted={derived.snapshotPredicted}
        metrics={meta.metrics ?? null}
      />

      {welcomeBanner}

      {fireToasts}
    </>
  );
}

function TopBar({
  route, onNavigate, actionableAlertCount,
  showSidebarToggle = false, sidebarOpen = false, onToggleSidebar,
  sidebarCollapsed = false, onToggleSidebarCollapsed,
}: {
  route: AlertPageRoute;
  onNavigate: (r: AlertPageRoute) => void;
  actionableAlertCount: number;
  showSidebarToggle?: boolean;
  sidebarOpen?: boolean;
  onToggleSidebar?: () => void;
  sidebarCollapsed?: boolean;
  onToggleSidebarCollapsed?: () => void;
}) {
  const { t } = useLang();
  return (
    <header className="top-bar">
      <div className="top-bar-brand">
        {/* Mobile: hamburger ☰ that opens off-canvas drawer */}
        {showSidebarToggle && (
          <button
            type="button"
            className="hamburger-btn mobile-only"
            onClick={onToggleSidebar}
            aria-label={sidebarOpen ? t("sidebar.close") : t("sidebar.open")}
            aria-expanded={sidebarOpen}
          >
            <span>{sidebarOpen ? "✕" : "☰"}</span>
          </button>
        )}
        {/* Desktop: chevron that collapses sidebar to 0 width */}
        {showSidebarToggle && onToggleSidebarCollapsed && (
          <button
            type="button"
            className="hamburger-btn desktop-only"
            onClick={onToggleSidebarCollapsed}
            aria-label={sidebarCollapsed ? t("sidebar.expand") : t("sidebar.collapse")}
            aria-expanded={!sidebarCollapsed}
            title={sidebarCollapsed ? t("sidebar.expand.hint") : t("sidebar.collapse.hint")}
          >
            <span style={{ display: "inline-block", transition: "transform 0.2s" }}>
              {sidebarCollapsed ? "›" : "‹"}
            </span>
          </button>
        )}
        <span className="top-bar-logo" aria-hidden="true">🔥</span>
        <span className="top-bar-title">FireWatch Thailand</span>
      </div>
      <NavTabs
        active={route}
        onNavigate={onNavigate}
        actionableCount={actionableAlertCount}
      />
      <div className="top-bar-status">
        <LanguageToggle />
        <ThemeToggle />
        <LiveStatusBadge />
      </div>
    </header>
  );
}

function Legend({
  showPredicted,
  showCritical,
  showObserved,
  showLiveFire,
}: {
  showPredicted: boolean;
  showCritical: boolean;
  showObserved: boolean;
  showLiveFire: boolean;
}) {
  const { t } = useLang();
  if (!showPredicted && !showObserved && !showLiveFire) return null;
  return (
    <div className="legend">
      {showPredicted && (
        <div className="legend-title">{t("legend.title")}</div>
      )}
      <div className="legend-items">
        {showPredicted && (
          <>
            {showCritical && (
              <div className="legend-item">
                <div className="legend-color critical" />
                <span>{t("legend.critical")}</span>
              </div>
            )}
            <div className="legend-item">
              <div className="legend-color high" />
              <span>{t("legend.high")}</span>
            </div>
            <div className="legend-item">
              <div className="legend-color medium" />
              <span>{t("legend.watch")}</span>
            </div>
            <div className="legend-item">
              <div className="legend-color low" />
              <span>{t("legend.background")}</span>
            </div>
          </>
        )}
        {showObserved && (
          <div className="legend-item">
            <div className="legend-color observed" />
            <span>{t("legend.observed")}</span>
          </div>
        )}
        {showLiveFire && (
          <div className="legend-item">
            <div className="legend-color live-fire" />
            <span>{t("legend.live")}</span>
          </div>
        )}
      </div>
    </div>
  );
}
