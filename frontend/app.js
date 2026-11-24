// =====================================================================
// FireWatch Thailand — core: shared state, data loading, routing,
// theme and language switching, and small helpers used by every view.
// =====================================================================

const App = (() => {
  const VIEWS = ["map", "live", "analytics", "compare", "alerts", "reports"];
  const URGENCY = ["HIGH", "MEDIUM", "LOW"];

  const state = {
    view: "map",
    theme: document.documentElement.dataset.theme || "dark",
    lang: "en",
    metadata: null,       // /metadata  (dataset_info.json)
    performance: null,    // /performance {holdout, prospective}
    forecasts: null,      // /forecasts  {latest, forecasts:[...]}
    snapshot: "",         // selected snapshot file name; "" = latest
    geojson: null,
    cells: [],            // predicted cell properties + {lat, lon, province}
    observed: [],         // observed detections on the feature-cutoff day
    live: null,           // /alerts feed
    province: "",         // "" = Thailand, else a Thai province code
    threshold: null,      // display threshold; null = model threshold
    errors: {},
  };

  // ---------- events ----------
  const listeners = {};
  function on(event, fn) { (listeners[event] = listeners[event] || []).push(fn); }
  function emit(event, data) {
    (listeners[event] || []).forEach(fn => {
      try { fn(data); } catch (err) { console.error(`[${event}]`, err); }
    });
  }

  // ---------- helpers ----------
  async function getJSON(url) {
    const response = await fetch(url, { cache: "no-store" });
    if (!response.ok) throw new Error(`${url} → ${response.status}`);
    return response.json();
  }

  const ESC = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
  function esc(value) { return String(value ?? "").replace(/[&<>"']/g, c => ESC[c]); }

  function locale() { return state.lang === "th" ? "th-TH" : "en-GB"; }
  function num(value, digits = 0) {
    if (value === null || value === undefined || Number.isNaN(value)) return "—";
    return Number(value).toLocaleString(locale(), { minimumFractionDigits: digits, maximumFractionDigits: digits });
  }
  function pct(value, digits = 0) {
    if (value === null || value === undefined || Number.isNaN(value)) return "—";
    return num(value * 100, digits) + "%";
  }
  function fixed(value, digits = 3) {
    if (value === null || value === undefined || Number.isNaN(value)) return "—";
    return Number(value).toFixed(digits);
  }
  function parseISODate(iso) { return new Date(iso + "T00:00:00Z"); }
  function addDays(iso, days) {
    const d = parseISODate(iso);
    d.setUTCDate(d.getUTCDate() + days);
    return d.toISOString().slice(0, 10);
  }
  function daysBetween(fromIso, toIso) {
    return Math.round((parseISODate(toIso) - parseISODate(fromIso)) / 864e5);
  }
  function todayISO() {
    const now = new Date();
    return new Date(Date.UTC(now.getFullYear(), now.getMonth(), now.getDate())).toISOString().slice(0, 10);
  }
  function fmtDate(iso, opts = { day: "numeric", month: "short", year: "numeric" }) {
    if (!iso) return "—";
    return parseISODate(iso.slice(0, 10)).toLocaleDateString(locale(), { ...opts, timeZone: "UTC" });
  }
  function fmtDateTime(iso, opts = { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" }) {
    if (!iso) return "—";
    return new Date(iso).toLocaleString(locale(), opts);
  }
  function timeAgo(iso) {
    const minutes = Math.max(0, Math.round((Date.now() - new Date(iso)) / 60000));
    if (minutes < 1) return I18N.t("time.justNow");
    if (minutes < 60) return I18N.t("time.minutesAgo", { n: minutes });
    const hours = Math.round(minutes / 60);
    if (hours < 48) return I18N.t("time.hoursAgo", { n: hours });
    return I18N.t("time.daysAgo", { n: Math.round(hours / 24) });
  }

  function cssVar(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }

  function basemap() {
    return [RoadBasemap.create(state.lang)];
  }
  function urgencyColor(level) {
    return { HIGH: cssVar("--high"), MEDIUM: cssVar("--medium"), LOW: cssVar("--low") }[level] || cssVar("--faint");
  }

  function provinceName(code) {
    if (code === null || code === undefined || code === "OUT") return I18N.t("geo.outside");
    return Geo.name(code, state.lang) || code;
  }

  function toCSV(rows, columns) {
    const quote = v => {
      const s = v === null || v === undefined ? "" : String(v);
      return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
    };
    return [columns.map(c => quote(c.label || c.key)).join(","),
            ...rows.map(row => columns.map(c => quote(typeof c.value === "function" ? c.value(row) : row[c.key])).join(","))]
      .join("\n");
  }
  function download(filename, content, mime = "text/csv;charset=utf-8") {
    const blob = content instanceof Blob ? content : new Blob([content], { type: mime });
    const url = URL.createObjectURL(blob);
    const a = Object.assign(document.createElement("a"), { href: url, download: filename });
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 2000);
  }
  function downloadDataURL(filename, dataURL) {
    const a = Object.assign(document.createElement("a"), { href: dataURL, download: filename });
    document.body.appendChild(a);
    a.click();
    a.remove();
  }

  let toastTimer = null;
  function toast(message) {
    const el = document.getElementById("toast");
    el.textContent = message;
    el.classList.add("show");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => el.classList.remove("show"), 2600);
  }

  // ---------- modal ----------
  const modal = {
    onClose: null,
    open(title, html, onMount, onClose) {
      this.close();
      document.getElementById("modalTitle").textContent = title;
      document.getElementById("modalBody").innerHTML = html;
      document.getElementById("modal").hidden = false;
      this.onClose = onClose || null;
      if (onMount) onMount(document.getElementById("modalBody"));
      document.getElementById("modalClose").focus();
    },
    close() {
      const el = document.getElementById("modal");
      if (el.hidden) return;
      el.hidden = true;
      Charts.destroyWithin(document.getElementById("modalBody"));
      document.getElementById("modalBody").innerHTML = "";
      if (this.onClose) this.onClose();
      this.onClose = null;
    },
    isOpen() { return !document.getElementById("modal").hidden; },
  };

  // ---------- charts ----------
  const Charts = {
    registry: new Map(),
    make(canvas, config) {
      // Discard a delayed render targeting a view that has already been replaced.
      if (!canvas || !canvas.isConnected) return null;
      if (typeof Chart === "undefined") {
        canvas.replaceWith(Object.assign(document.createElement("div"), { className: "empty", textContent: I18N.t("common.chartsUnavailable") }));
        return null;
      }
      this.destroy(canvas);
      const muted = cssVar("--muted"), grid = cssVar("--chart-grid");
      Chart.defaults.color = muted;
      Chart.defaults.borderColor = grid;
      Chart.defaults.font.family = cssVar("--font");
      const chart = new Chart(canvas, config);
      this.registry.set(canvas, chart);
      return chart;
    },
    destroy(canvas) {
      const chart = this.registry.get(canvas);
      if (chart) { chart.destroy(); this.registry.delete(canvas); }
    },
    destroyWithin(root) {
      for (const [canvas, chart] of this.registry) {
        if (!document.body.contains(canvas) || root.contains(canvas)) { chart.destroy(); this.registry.delete(canvas); }
      }
    },
  };

  // ---------- model / snapshot accessors ----------
  function meta() { return state.geojson ? state.geojson.metadata : null; }
  function holdout() { return state.performance ? state.performance.holdout : null; }
  function horizonDays() { return meta()?.horizon_days ?? state.metadata?.max_prediction_days ?? 3; }
  function evaluationHorizon() { return holdout()?.horizon_days ?? 7; }
  function modelThreshold() { const m = meta(); return m ? m.threshold : 0.5; }
  function displayThreshold() { return state.threshold ?? modelThreshold(); }
  function isCalibrated() { return state.cells.length > 0 && typeof state.cells[0].probability === "number"; }

  function likelihoodLevel(score) {
    return score >= 0.7 ? "HIGH" : score >= 0.4 ? "MEDIUM" : "LOW";
  }

  // Legacy snapshots contain uncalibrated scores, which must not be labelled
  // as percentages. Keep the same bands but identify their actual measure.
  function tierText(key, vars = {}) {
    const calibrated = isCalibrated();
    const trainingText = key.startsWith("rp.") || key.startsWith("gloss.") ||
      ["det.task", "det.splitNote", "cmp.weatherHoldoutSub", "cmp.holdoutSub"].includes(key);
    const horizon = trainingText ? evaluationHorizon() : horizonDays();
    return I18N.t(key, {
      horizon,
      measure: I18N.t(calibrated ? "tier.probability" : "tier.score"),
      measureTitle: I18N.t(calibrated ? "tier.probabilityTitle" : "tier.scoreTitle", { horizon }),
      high: calibrated ? "≥70%" : "≥0.70",
      medium: calibrated ? "40–<70%" : "0.40–<0.70",
      low: calibrated ? "<40%" : "<0.40",
      ...vars,
    });
  }

  // Forecast state of a cell at the current display threshold.
  function cellView(cell) {
    const t = displayThreshold();
    const prob = typeof cell.probability === "number" ? cell.probability : null;
    const score = prob ?? cell.detection_score;
    if (Math.abs(t - modelThreshold()) < 1e-9) {
      return { flagged: cell.forecast_detection, urgency: cell.forecast_detection ? likelihoodLevel(score) : "NONE", days: cell.days_until_fire,
               date: cell.predicted_fire_date, prob, score };
    }
    if (score < t) return { flagged: false, urgency: "NONE", days: -1, date: null, prob, score };
    const days = Math.min(horizonDays(), Math.max(1, Math.round(cell.estimated_days_if_detection)));
    return { flagged: true, urgency: likelihoodLevel(score), days,
             date: addDays(cell.base_date, days), prob, score };
  }

  function matchesProvince(code) {
    if (!code) return false;
    if (!state.province) return true;
    return code === state.province;
  }

  // Flagged cells within the province filter, each with its current view.
  function flaggedCells({ ignoreProvince = false } = {}) {
    const out = [];
    for (const cell of state.cells) {
      if (!ignoreProvince && !matchesProvince(cell.province)) continue;
      const view = cellView(cell);
      if (view.flagged) out.push({ cell, view });
    }
    return out;
  }

  function reliabilityAt(p) {
    // Current held-out calibration cannot describe a different archived model.
    if (meta()?.model_version !== state.metadata?.model_version) return null;
    const h = holdout();
    const bins = h && h.diagnostics && h.diagnostics.reliability;
    if (!bins || p === null) return null;
    return bins.find(b => p >= b.lower && (p < b.upper || b.upper >= 1)) || null;
  }

  // ---------- data loading ----------
  async function loadSnapshot(name) {
    const latest = state.forecasts && state.forecasts.latest;
    const url = !name || name === latest ? "/geojson" : `/forecasts/${encodeURIComponent(name)}/geojson`;
    const geojson = await getJSON(url);
    await Geo.load();
    const cells = [], observed = [];
    for (const f of geojson.features) {
      const [lon, lat] = f.geometry.coordinates;
      const p = f.properties;
      const province = Geo.locate(lat, lon);
      if (!province) continue;
      if (p.source === "predicted") cells.push({ ...p, lat, lon, province });
      else if (p.source === "observed") observed.push({ ...p, lat, lon, province });
    }
    state.geojson = { ...geojson, features: geojson.features.filter(f => Geo.locate(f.geometry.coordinates[1], f.geometry.coordinates[0])) };
    state.cells = cells;
    state.observed = observed;
    state.snapshot = !name || name === latest ? "" : name;
    delete state.errors.geojson;
    emit("data");
  }

  async function selectSnapshot(name) {
    try {
      await loadSnapshot(name);
      syncHash();
    } catch (err) {
      console.error(err);
      toast(I18N.t("common.loadFailed"));
    }
  }

  // ---------- routing ----------
  function parseHash() {
    const raw = location.hash.replace(/^#/, "");
    const [view, query] = raw.split("?");
    return { view: VIEWS.includes(view) ? view : "map", params: new URLSearchParams(query || "") };
  }
  function hashFor(view) {
    if (view !== "map") return "#" + view;
    const params = new URLSearchParams();
    if (state.snapshot) params.set("snapshot", state.snapshot);
    if (state.province) params.set("province", state.province);
    if (state.threshold !== null) params.set("t", state.threshold.toFixed(2));
    const q = params.toString();
    return "#map" + (q ? "?" + q : "");
  }
  function syncHash() {
    const target = hashFor(state.view);
    if (location.hash !== target) history.replaceState(null, "", target);
  }
  function show(view) {
    state.view = view;
    document.body.dataset.view = view;
    document.querySelectorAll(".view").forEach(el => el.classList.toggle("active", el.dataset.view === view));
    document.querySelectorAll(".tab").forEach(el => {
      const active = el.dataset.view === view;
      el.classList.toggle("active", active);
      if (active) el.setAttribute("aria-current", "page"); else el.removeAttribute("aria-current");
    });
    syncHash();
    emit("show", view);
  }
  function go(view) {
    if (state.view === view) { syncHash(); return; }
    location.hash = hashFor(view);
  }

  // ---------- theme & language ----------
  function setTheme(theme) {
    state.theme = theme;
    document.documentElement.dataset.theme = theme;
    document.getElementById("themeToggle").textContent = theme === "dark" ? "🌙" : "☀️";
    try { localStorage.setItem("fw.theme", theme); } catch (e) { /* storage unavailable */ }
    emit("theme");
  }
  function setLang(lang) {
    state.lang = lang;
    I18N.setLang(lang);
    document.documentElement.lang = lang;
    document.getElementById("langToggle").textContent = lang === "th" ? "TH" : "EN";
    try { localStorage.setItem("fw.lang", lang); } catch (e) { /* storage unavailable */ }
    I18N.apply(document);
    emit("lang");
  }

  // ---------- live pill ----------
  function liveState() {
    const feed = state.live;
    if (state.errors.live) return "offline";
    if (!feed || feed.status === "starting") return "connecting";
    if (feed.status === "missing_key") return "nokey";
    if (feed.status === "error") return "offline";
    if (feed.stale || feed.status === "partial") return "delayed";
    return "live";
  }
  function updateLivePill() {
    const pill = document.getElementById("livePill");
    const s = liveState();
    pill.dataset.state = s;
    document.getElementById("livePillText").textContent = I18N.t("live.pill." + s);
    pill.title = I18N.t("live.pillTitle." + s);
  }

  // ---------- start ----------
  async function start() {
    let lang = "en";
    try { lang = localStorage.getItem("fw.lang") || "en"; } catch (e) { /* storage unavailable */ }
    setLang(lang === "th" ? "th" : "en");
    setTheme(state.theme === "light" ? "light" : "dark");

    const { view, params } = parseHash();
    state.province = params.get("province") || "";
    const t = parseFloat(params.get("t"));
    state.threshold = Number.isFinite(t) && t > 0 && t < 1 ? t : null;
    const requested = params.get("snapshot") || "";

    document.getElementById("themeToggle").addEventListener("click", () => setTheme(state.theme === "dark" ? "light" : "dark"));
    document.getElementById("langToggle").addEventListener("click", () => setLang(state.lang === "en" ? "th" : "en"));
    document.getElementById("livePill").addEventListener("click", () => go("live"));
    document.getElementById("sidebarToggle").addEventListener("click", () => {
      document.body.classList.toggle("sidebar-collapsed");
      emit("layout");
    });
    document.getElementById("modalClose").addEventListener("click", () => modal.close());
    document.getElementById("modal").addEventListener("click", e => { if (e.target.id === "modal") modal.close(); });
    document.addEventListener("keydown", e => { if (e.key === "Escape") modal.close(); });
    window.addEventListener("hashchange", () => show(parseHash().view));
    on("lang", updateLivePill);
    on("live", updateLivePill);
    if (window.matchMedia("(max-width: 760px)").matches) document.body.classList.add("sidebar-collapsed");

    show(view);
    updateLivePill();

    const results = await Promise.allSettled([
      getJSON("/metadata"), getJSON("/performance"), getJSON("/forecasts"), Geo.load(),
    ]);
    const [metadata, performance, forecasts] = results;
    if (metadata.status === "fulfilled") state.metadata = metadata.value; else state.errors.metadata = true;
    if (performance.status === "fulfilled") state.performance = performance.value; else state.errors.performance = true;
    if (forecasts.status === "fulfilled") state.forecasts = forecasts.value; else state.errors.forecasts = true;
    if (!Geo.byCode(state.province)) state.province = "";
    emit("meta");

    try {
      await loadSnapshot(requested);
    } catch (err) {
      console.error(err);
      if (requested) {
        try { await loadSnapshot(""); } catch (inner) { state.errors.geojson = true; emit("data"); }
      } else {
        state.errors.geojson = true;
        emit("data");
      }
    }
    syncHash();
  }

  document.addEventListener("DOMContentLoaded", start);

  return {
    state, on, emit, getJSON, esc, num, pct, fixed, addDays, daysBetween, todayISO, fmtDate, fmtDateTime,
    timeAgo, cssVar, basemap, urgencyColor, provinceName, toCSV, download, downloadDataURL, toast, modal, Charts,
    meta, holdout, horizonDays, evaluationHorizon, modelThreshold, displayThreshold, isCalibrated, likelihoodLevel, tierText, cellView, flaggedCells, matchesProvince,
    reliabilityAt, selectSnapshot, syncHash, go, URGENCY,
  };
})();
