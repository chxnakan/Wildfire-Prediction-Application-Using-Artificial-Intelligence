// =====================================================================
// Live Fires: NASA FIRMS near-real-time detections from /alerts,
// independent of the forecast model. Polls the server every minute.
// =====================================================================

const Live = (() => {
  const { state } = App;
  const t = (key, vars) => I18N.t(key, vars);
  const POLL_MS = 60000;
  const LIMIT = 2000;
  const filters = { satellite: "", province: "", hours: 24, search: "" };
  let knownIds = new Set();
  let initialized = false;
  let newIds = new Set();
  let fetchedAt = null;
  let detailsOpen = false;
  let timer = null;

  function firmsLink(a) {
    const day = a.observed_at.slice(0, 10);
    return `https://firms.modaps.eosdis.nasa.gov/map/#d:${day};@${a.longitude.toFixed(4)},${a.latitude.toFixed(4)},12.0z`;
  }

  async function poll() {
    clearTimeout(timer);
    try {
      const feed = await App.getJSON(`/alerts?scope=thailand&limit=${LIMIT}`);
      const fresh = initialized ? feed.alerts.filter(a => !knownIds.has(a.id)) : [];
      newIds = new Set(fresh.map(a => a.id));
      state.live = feed;
      delete state.errors.live;
      fetchedAt = new Date();
      // Do not establish a baseline from a pending or failed first request.
      if (feed.status === "ok" || feed.status === "partial") {
        initialized = true;
        feed.alerts.forEach(a => knownIds.add(a.id));
        if (knownIds.size > 10000) knownIds = new Set([...knownIds].slice(-5000));
      }
      if (fresh.length) App.emit("live:new", fresh);
    } catch (err) {
      state.errors.live = true;
    } finally {
      App.emit("live");
      timer = setTimeout(poll, POLL_MS);
    }
  }

  function statusMessage() {
    const feed = state.live;
    if (state.errors.live) return t("live.status.lost");
    if (!feed) return t("live.status.starting");
    if (["starting", "missing_key", "error", "partial"].includes(feed.status)) return t("live.status." + feed.status);
    if (feed.stale) return t("live.status.stale");
    return feed.total ? t("live.status.ok", { n: App.num(feed.total) }) : t("live.status.none");
  }

  function annotated() {
    return ((state.live && state.live.alerts) || []).map(a => ({ ...a, province: Geo.locate(a.latitude, a.longitude) }));
  }

  function filtered(rows) {
    const now = Date.now();
    const q = filters.search.trim().toLowerCase();
    return rows.filter(a => {
      if (!a.province) return false;
      if (filters.satellite && a.satellite !== filters.satellite) return false;
      if (filters.province && a.province !== filters.province) return false;
      if (now - new Date(a.observed_at) > filters.hours * 3600e3) return false;
      if (q) {
        const hay = [Geo.name(a.province, "en"), Geo.name(a.province, "th"), a.satellite,
                     a.latitude.toFixed(4), a.longitude.toFixed(4), a.confidence].join(" ").toLowerCase();
        if (!hay.includes(q)) return false;
      }
      return true;
    });
  }

  function pillHTML() {
    const s = document.getElementById("livePill").dataset.state;
    const cls = s === "live" ? "" : s === "delayed" || s === "connecting" ? "warn" : "bad";
    const total = state.live && state.live.status !== "missing_key" ? App.num(state.live.total) : "—";
    return `<span class="live-status-pill ${cls}">⚡ ${t("live.statusPill", { n: total })}</span>`;
  }

  function render() {
    const root = document.getElementById("livePage");
    const focused = document.activeElement && document.activeElement.id === "lfSearch";
    const caret = focused ? document.activeElement.selectionStart : null;
    const rows = annotated();
    const now = Date.now();
    const lastHour = rows.filter(a => now - new Date(a.observed_at) <= 3600e3).length;
    const bySat = sat => (state.live && state.live.by_satellite ? state.live.by_satellite[sat] || 0 : rows.filter(a => a.satellite === sat).length);
    const provinces = [...new Set(rows.map(a => a.province).filter(Boolean))]
      .map(code => ({ code, name: App.provinceName(code) }))
      .sort((a, b) => a.name.localeCompare(b.name, state.lang === "th" ? "th" : "en"));
    const feed = state.live;
    const truncated = feed && feed.total > feed.alerts.length;
    root.innerHTML = `
      <div class="page-head">
        <div>
          <h1 class="page-title">🔥 ${t("live.title")}</h1>
          <p class="page-sub">${t("live.subtitle")} <button type="button" class="link-btn" id="liveDetailsBtn">${t(detailsOpen ? "live.hideDetails" : "live.showDetails")}</button></p>
        </div>
        <div class="head-right">
          <div class="head-actions">${pillHTML()}<button type="button" class="btn sm" id="liveRefresh">↻ ${t("live.refresh")}</button></div>
          <div class="head-note">${fetchedAt ? t("live.updated", { time: fetchedAt.toLocaleTimeString({ th: "th-TH", zh: "zh-TW" }[state.lang] || "en-GB") }) : t("common.loading")}</div>
        </div>
      </div>
      <div class="details-panel" ${detailsOpen ? "" : "hidden"}>
        <div><b>${t("live.statusLabel")}:</b> ${App.esc(statusMessage())}</div>
        ${feed ? `<div><b>${t("live.area")}:</b> <span class="mono">${App.esc(feed.bbox)}</span> (${t("live.areaHelp")})</div>
        <div><b>${t("live.lastSuccess")}:</b> ${feed.last_success ? App.fmtDateTime(feed.last_success, { dateStyle: "medium", timeStyle: "short" }) : t("live.noSuccess")}</div>` : ""}
        <div>${t("live.latencyNote")}</div>
        ${truncated ? `<div>${t("live.truncated", { shown: feed.alerts.length, total: App.num(feed.total) })}</div>` : ""}
      </div>
      <div class="grid four section-gap">
        <div class="stat red"><div class="stat-label">${t("live.past1h")}</div><div class="stat-value">${App.num(lastHour)}</div></div>
        <div class="stat orange"><div class="stat-label">${t("live.past24h")}</div><div class="stat-value">${feed && feed.status !== "missing_key" ? App.num(feed.total) : "—"}</div></div>
        <div class="stat cyan"><div class="stat-label">NOAA-20</div><div class="stat-value">${App.num(bySat("NOAA20"))}</div></div>
        <div class="stat amber"><div class="stat-label">NOAA-21</div><div class="stat-value">${App.num(bySat("NOAA21"))}</div></div>
      </div>
      <div class="filters">
        <label class="check-pill"><input type="checkbox" id="lfThai" checked disabled> 🇹🇭 ${t("live.thailandOnly")}</label>
        <select class="select" id="lfSat">
          <option value="">${t("live.allSatellites")}</option>
          <option value="NOAA20" ${filters.satellite === "NOAA20" ? "selected" : ""}>VIIRS NOAA-20</option>
          <option value="NOAA21" ${filters.satellite === "NOAA21" ? "selected" : ""}>VIIRS NOAA-21</option>
        </select>
        <select class="select" id="lfProv">
          <option value="">${t("live.allProvinces")}</option>
          ${provinces.map(p => `<option value="${p.code}" ${filters.province === p.code ? "selected" : ""}>${App.esc(p.name)}</option>`).join("")}
        </select>
        <select class="select" id="lfHours">
          ${[1, 6, 24].map(h => `<option value="${h}" ${filters.hours === h ? "selected" : ""}>${t("live.pastHours", { n: h })}</option>`).join("")}
        </select>
        <input class="input" id="lfSearch" type="search" placeholder="${t("live.search")}" value="${App.esc(filters.search)}">
        <span class="count-note" id="lfCount"></span>
      </div>
      <div id="liveList"></div>`;
    renderList(rows);
    if (focused) {
      const input = document.getElementById("lfSearch");
      input.focus();
      input.setSelectionRange(caret, caret);
    }
  }

  function renderList(rows = annotated()) {
    const list = document.getElementById("liveList");
    if (!list) return;
    const shown = filtered(rows);
    document.getElementById("lfCount").textContent = t("live.count", { shown: App.num(shown.length), total: App.num(rows.length) });
    const feed = state.live;
    if (!shown.length) {
      list.innerHTML = `<div class="card empty">${App.esc(rows.length ? t("live.noMatch") : statusMessage())}</div>`;
      return;
    }
    list.innerHTML = shown.map(a => `
      <article class="incident ${newIds.has(a.id) ? "is-new" : ""}">
        <div class="inc-when">
          <span class="inc-ago">${App.timeAgo(a.observed_at)}</span>
          <span class="inc-date">${App.fmtDateTime(a.observed_at, { year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit" })}</span>
        </div>
        <div>
          <div class="inc-badges">
            <span class="src-badge">🛰️ FIRMS</span>
            <span class="mono-badge">${t("live.conf")}: ${App.esc(t("conf." + a.confidence))}</span>
            ${a.frp_mw !== null && a.frp_mw !== undefined ? `<span class="mono-badge">FRP ${App.num(a.frp_mw, 1)} MW</span>` : ""}
            ${newIds.has(a.id) ? `<span class="badge info">${t("live.new")}</span>` : ""}
          </div>
          <div class="inc-title">${a.province ? App.esc(App.provinceName(a.province)) : `${t("geo.outside")} <span class="muted">(${t("live.outsideHelp")})</span>`}</div>
          <div class="inc-meta"><span>${a.latitude.toFixed(4)}°N · ${a.longitude.toFixed(4)}°E</span><span>📡 VIIRS ${App.esc(a.satellite)}</span></div>
          <div class="inc-actions">
            <button type="button" class="btn" data-focus="${a.id}">📍 ${t("live.viewOnMap")}</button>
            <a class="btn" href="${firmsLink(a)}" target="_blank" rel="noopener">🔗 ${t("live.verify")}</a>
          </div>
        </div>
      </article>`).join("") + (feed && feed.total > feed.alerts.length
        ? `<p class="muted small" style="margin-top:8px">${t("live.truncated", { shown: feed.alerts.length, total: App.num(feed.total) })}</p>` : "");
  }

  const root = document.getElementById("livePage");
  root.addEventListener("click", e => {
    if (e.target.id === "liveRefresh") { poll(); return; }
    if (e.target.id === "liveDetailsBtn") { detailsOpen = !detailsOpen; render(); return; }
    const btn = e.target.closest("[data-focus]");
    if (btn) {
      const a = (state.live.alerts || []).find(x => x.id === btn.dataset.focus);
      if (a) MapView.focus(a.latitude, a.longitude, "live", a);
    }
  });
  root.addEventListener("change", e => {
    if (e.target.id === "lfSat") filters.satellite = e.target.value;
    else if (e.target.id === "lfProv") filters.province = e.target.value;
    else if (e.target.id === "lfHours") filters.hours = parseInt(e.target.value, 10);
    else return;
    renderList();
  });
  root.addEventListener("input", e => {
    if (e.target.id === "lfSearch") { filters.search = e.target.value; renderList(); }
  });

  const active = () => state.view === "live";
  App.on("live", () => { if (active()) render(); });
  App.on("lang", () => { if (active()) render(); });
  App.on("show", view => { if (view === "live") render(); });
  document.addEventListener("DOMContentLoaded", () => Geo.load().then(poll));

  return { firmsLink, poll };
})();
