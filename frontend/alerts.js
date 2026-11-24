// =====================================================================
// Alerts: the forecast watch list (flagged cells, most urgent first) and
// a session log of new live detections reported by the FIRMS feed.
// =====================================================================

const Alerts = (() => {
  const { state } = App;
  const t = (key, vars) => App.tierText(key, vars);
  const filters = { urgency: "", province: null, search: "" };
  const expanded = new Set();
  const PAGE = 40;
  const log = [];
  let rendered = [];

  App.on("live:new", items => {
    for (const a of items) log.unshift({ ...a, seen: new Date().toISOString() });
    log.splice(200);
    if (state.view === "alerts") renderLog();
  });

  function rows() {
    const q = filters.search.trim().toLowerCase();
    const province = filters.province ?? state.province;
    return App.flaggedCells({ ignoreProvince: true })
      .filter(({ cell, view }) => {
        if (filters.urgency && view.urgency !== filters.urgency) return false;
        if (province === "OUT" ? cell.province : province && cell.province !== province) return false;
        if (q) {
          const hay = [Geo.name(cell.province, "en"), Geo.name(cell.province, "th"), cell.lat.toFixed(3), cell.lon.toFixed(3)].join(" ").toLowerCase();
          if (!hay.includes(q)) return false;
        }
        return true;
      })
      .sort((a, b) => (b.view.prob ?? b.view.score) - (a.view.prob ?? a.view.score));
  }

  function rowHTML({ cell, view }, index) {
    const calibrated = view.prob !== null;
    return `<div class="alert-row ${view.urgency}">
      <div class="alert-tier">
        <div class="alert-level">${t("urgency.name." + view.urgency)}</div>
        <div class="alert-when">${t("al.estimated", { date: App.fmtDate(view.date), n: view.days })}</div>
      </div>
      <div class="alert-main">
        <b>${App.esc(App.provinceName(cell.province))}</b>
        <div class="inc-meta"><span>${cell.lat.toFixed(3)}°N · ${cell.lon.toFixed(3)}°E</span>${calibrated ? `<span>${t("pop.rawScore")} ${App.fixed(cell.detection_score, 3)}</span>` : ""}</div>
      </div>
      <div style="display:flex;align-items:center;gap:12px">
        <div class="alert-prob">${calibrated ? App.pct(view.prob) : App.fixed(view.score, 3)}<small>${t(calibrated ? "al.probability" : "al.score")}</small></div>
        <button type="button" class="btn sm" data-cell="${index}">📍 ${t("al.map")}</button>
      </div>
    </div>`;
  }

  function render() {
    const root = document.getElementById("alertsPage");
    const m = App.meta();
    if (!m) {
      root.innerHTML = `<div class="card empty">${t(state.errors.geojson ? "sb.dataMissing" : "common.loading")}</div>`;
      return;
    }
    const province = filters.province ?? state.province;
    const provinces = new Map();
    for (const { cell } of App.flaggedCells({ ignoreProvince: true })) {
      if (cell.province) provinces.set(cell.province, (provinces.get(cell.province) || 0) + 1);
    }
    const options = [...provinces.entries()].map(([code, n]) => ({ code, n, name: App.provinceName(code) }))
      .sort((a, b) => a.name.localeCompare(b.name, state.lang === "th" ? "th" : "en"));
    root.innerHTML = `
      <div class="page-head">
        <div>
          <h1 class="page-title">🔔 ${t("al.title")}</h1>
          <p class="page-sub">${t("al.subtitle", { date: m.base_date, t: App.displayThreshold().toFixed(2) })}</p>
        </div>
        <div class="head-right">${m.kind !== "prospective" ? `<span class="badge warn">${t("sb.kindHistorical")}</span>` : ""}</div>
      </div>
      <div class="filters">
        <div class="seg" role="group" aria-label="${t("al.urgencyFilter")}">
          ${["", "HIGH", "MEDIUM", "LOW"].map(u => `<button type="button" data-urgency="${u}" class="${filters.urgency === u ? "active" : ""}">${u ? t("urgency." + u) : t("al.all")}</button>`).join("")}
        </div>
        <select class="select" id="alProv">
          <option value="">${t("sb.allProvinces", { n: options.length })}</option>
          ${options.map(o => `<option value="${o.code}" ${province === o.code ? "selected" : ""}>${App.esc(o.name)} (${o.n})</option>`).join("")}
        </select>
        <input class="input" id="alSearch" type="search" placeholder="${t("al.search")}" value="${App.esc(filters.search)}">
        <span class="count-note" id="alCount"></span>
      </div>
      <div id="alGroups"></div>
      <div class="card section-gap">
        <div class="card-head"><div><div class="card-title">🛰️ ${t("al.logTitle")}</div><div class="card-sub">${t("al.logSub")}</div></div></div>
        <div id="alLog"></div>
      </div>`;
    renderGroups();
    renderLog();
  }

  function renderGroups() {
    const el = document.getElementById("alGroups");
    if (!el) return;
    rendered = rows();
    const total = App.flaggedCells({ ignoreProvince: true }).length;
    document.getElementById("alCount").textContent = t("al.count", { shown: App.num(rendered.length), total: App.num(total) });
    if (!rendered.length) {
      el.innerHTML = `<div class="card empty">${t("al.none")}</div>`;
      return;
    }
    let html = "";
    for (const level of App.URGENCY) {
      const group = rendered.map((r, i) => ({ r, i })).filter(({ r }) => r.view.urgency === level);
      if (!group.length) continue;
      const limit = expanded.has(level) ? group.length : PAGE;
      html += `<div class="group-title" style="color:${App.urgencyColor(level)}">${t("urgency.name." + level)} <span class="count">· ${App.num(group.length)}</span></div>`;
      html += group.slice(0, limit).map(({ r, i }) => rowHTML(r, i)).join("");
      if (group.length > limit) {
        html += `<button type="button" class="btn block" data-expand="${level}">${t("al.showAll", { n: App.num(group.length) })}</button>`;
      }
    }
    el.innerHTML = html;
  }

  function renderLog() {
    const el = document.getElementById("alLog");
    if (!el) return;
    if (!log.length) {
      el.innerHTML = `<div class="empty">${t("al.logEmpty")}</div>`;
      return;
    }
    el.innerHTML = log.slice(0, 50).map((a, i) => {
      const code = Geo.locate(a.latitude, a.longitude);
      return `<div class="alert-row" style="--ac:var(--live)">
        <div class="alert-tier"><div class="alert-level">${t("al.newDetection")}</div><div class="alert-when">${App.timeAgo(a.observed_at)}</div></div>
        <div class="alert-main"><b>${App.esc(App.provinceName(code))}</b><div class="inc-meta"><span>${a.latitude.toFixed(4)}°N · ${a.longitude.toFixed(4)}°E</span><span>VIIRS ${App.esc(a.satellite)}</span><span>${t("live.conf")}: ${App.esc(t("conf." + a.confidence))}</span></div></div>
        <button type="button" class="btn sm" data-log="${i}">📍 ${t("al.map")}</button>
      </div>`;
    }).join("");
  }

  const root = document.getElementById("alertsPage");
  root.addEventListener("click", e => {
    const seg = e.target.closest("[data-urgency]");
    if (seg) {
      filters.urgency = seg.dataset.urgency;
      root.querySelectorAll("[data-urgency]").forEach(b => b.classList.toggle("active", b === seg));
      renderGroups();
      return;
    }
    const more = e.target.closest("[data-expand]");
    if (more) { expanded.add(more.dataset.expand); renderGroups(); return; }
    const cellBtn = e.target.closest("[data-cell]");
    if (cellBtn) {
      const row = rendered[parseInt(cellBtn.dataset.cell, 10)];
      if (row) MapView.focus(row.cell.lat, row.cell.lon, "cell", row.cell);
      return;
    }
    const logBtn = e.target.closest("[data-log]");
    if (logBtn) {
      const a = log[parseInt(logBtn.dataset.log, 10)];
      if (a) MapView.focus(a.latitude, a.longitude, "live", a);
    }
  });
  root.addEventListener("change", e => {
    if (e.target.id === "alProv") { filters.province = e.target.value; renderGroups(); }
  });
  root.addEventListener("input", e => {
    if (e.target.id === "alSearch") { filters.search = e.target.value; renderGroups(); }
  });

  const active = () => state.view === "alerts";
  ["data", "filters", "lang", "theme"].forEach(evt => App.on(evt, () => { if (active()) render(); }));
  App.on("show", view => { if (view === "alerts") render(); });
  return { render };
})();
