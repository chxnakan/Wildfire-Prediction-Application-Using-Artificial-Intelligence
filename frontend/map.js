// =====================================================================
// Map view: glowing forecast markers, sidebar controls, popups with
// cell history and weather, and the Details / Threshold dialogs.
// =====================================================================

const MapView = (() => {
  const { state } = App;
  const t = (key, vars) => App.tierText(key, vars);
  const show = { predicted: true, observed: false, live: true };
  const THAILAND = Geo.bounds;

  // Whole zoom levels: fractional zoom rescales tiles and shows hairline seams.
  const map = L.map("map", { zoomControl: true, worldCopyJump: true, minZoom: 4 }).setView([13.2, 101.0], 6);
  let fitted = false;
  // The container has no size until the map view is first shown; fit Thailand then.
  function fitOnce() {
    map.invalidateSize();
    if (fitted || map.getSize().x === 0) return;
    map.fitBounds(THAILAND, { padding: [10, 10] });
    fitted = true;
  }
  map.createPane("outline").style.zIndex = 405;
  // One canvas for every marker layer: stacked canvases would swallow each other's clicks.
  // Draw order (observed → predicted → live) follows the order layers are added.
  const renderer = L.canvas({ padding: 0.4, tolerance: 4 });
  const layers = {
    observed: L.layerGroup().addTo(map),
    predicted: L.layerGroup().addTo(map),
    live: L.layerGroup().addTo(map),
  };
  let outline = null;
  let tiles = null;
  const popup = L.popup({ maxWidth: 290, autoPanPadding: [30, 30], closeButton: true });

  function setTiles() {
    if (tiles) tiles.forEach(layer => map.removeLayer(layer));
    tiles = App.basemap();
    tiles.forEach(layer => layer.addTo(map));
  }

  // ---------------- markers ----------------
  // The forecast grid is 0.1° (~4–5 px apart at zoom 6), so full-size markers and their
  // halos merge into blobs when zoomed out. Shrink markers, drop halos at low zoom, and
  // draw only the most important marker of a layer where several would overlap.
  function markerScale(z) {
    return z >= 8 ? 1 : z === 7 ? 0.85 : z === 6 ? 0.75 : 0.6;
  }

  function coreRadius(core, ring) {
    return Math.max(1.8, core * markerScale(map.getZoom())) + (ring ? 1 : 0);
  }

  // `items` must be ordered most important first. Keeps an item only if its marker core
  // (plus a small gap) does not overlap one already kept, using a pixel hash grid.
  function declutter(items, latlngOf, core, ring = false) {
    const zoom = map.getZoom();
    const minPx = 2 * coreRadius(core, ring) + 2;
    const grid = new Map();
    const kept = [];
    for (const item of items) {
      const p = map.project(latlngOf(item), zoom);
      const gx = Math.floor(p.x / minPx), gy = Math.floor(p.y / minPx);
      let clash = false;
      for (let dx = -1; dx <= 1 && !clash; dx++) {
        for (let dy = -1; dy <= 1 && !clash; dy++) {
          for (const q of grid.get(`${gx + dx},${gy + dy}`) || []) {
            if ((q.x - p.x) ** 2 + (q.y - p.y) ** 2 < minPx * minPx) { clash = true; break; }
          }
        }
      }
      if (clash) continue;
      const key = `${gx},${gy}`;
      if (!grid.has(key)) grid.set(key, []);
      grid.get(key).push(p);
      kept.push(item);
    }
    return kept;
  }

  function glow(layer, lat, lon, color, { core = 3, halo = 7, ring = false } = {}, onClick) {
    const z = map.getZoom();
    const s = markerScale(z);
    const r = coreRadius(core, false);
    const targets = [];
    if (z >= 7) {
      L.circleMarker([lat, lon], { renderer, radius: halo * s * 1.7, stroke: false, fillColor: color, fillOpacity: 0.05, interactive: false }).addTo(layer);
      targets.push(L.circleMarker([lat, lon], { renderer, radius: halo * s, stroke: false, fillColor: color, fillOpacity: 0.16 }).addTo(layer));
    }
    targets.push(L.circleMarker([lat, lon], ring
      ? { renderer, radius: r + 1, color, weight: z >= 7 ? 1.6 : 1.2, fill: true, fillOpacity: 0.1, fillColor: color }
      : { renderer, radius: r, color: "rgba(255,255,255,0.8)", weight: z >= 7 ? 0.8 : 0.6, fillColor: color, fillOpacity: 1 }).addTo(layer));
    if (onClick) targets.forEach(m => m.on("click", onClick));
  }

  function liveAlerts() {
    return (state.live && state.live.alerts) || [];
  }

  function redraw() {
    Object.values(layers).forEach(l => l.clearLayers());
    let visible = 0;
    if (show.observed) {
      const color = App.cssVar("--observed");
      const observed = declutter(state.observed.filter(o => App.matchesProvince(o.province)), o => [o.lat, o.lon], 2.6, true);
      for (const o of observed) {
        glow(layers.observed, o.lat, o.lon, color, { core: 2.6, halo: 5, ring: true },
             e => openPopup(e.latlng, observedPopup(o)));
      }
    }
    if (show.predicted) {
      const order = { LOW: 0, MEDIUM: 1, HIGH: 2 };
      const flagged = App.flaggedCells().sort((a, b) =>
        order[b.view.urgency] - order[a.view.urgency] || (b.view.prob ?? 0) - (a.view.prob ?? 0));
      // Keep the highest-risk cell where markers would overlap; draw HIGH last so it sits on top.
      const drawn = declutter(flagged, f => [f.cell.lat, f.cell.lon], 3.4).reverse();
      for (const { cell, view } of drawn) {
        glow(layers.predicted, cell.lat, cell.lon, App.urgencyColor(view.urgency),
             { core: view.urgency === "HIGH" ? 3.4 : 3, halo: view.urgency === "HIGH" ? 8 : 7 },
             e => openPopup(e.latlng, predictedPopup(cell), el => fillPredicted(el, cell)));
      }
      visible = flagged.length;
    }
    if (show.live) drawLive();
    updateCounts(visible);
  }

  function drawLive() {
    layers.live.clearLayers();
    if (!show.live) return;
    const color = App.cssVar("--live");
    const alerts = liveAlerts()
      .map(a => ({ a, code: Geo.locate(a.latitude, a.longitude) }))
      .filter(({ code }) => App.matchesProvince(code))
      .sort((x, y) => Date.parse(y.a.observed_at) - Date.parse(x.a.observed_at));
    // Keep the most recent detection where markers would overlap.
    for (const { a, code } of declutter(alerts, ({ a }) => [a.latitude, a.longitude], 3.2)) {
      glow(layers.live, a.latitude, a.longitude, color, { core: 3.2, halo: 8 },
           e => openPopup(e.latlng, livePopup(a, code)));
    }
  }

  function openPopup(latlng, html, onOpen) {
    popup.setLatLng(latlng).setContent(html).openOn(map);
    if (onOpen) onOpen(popup.getElement());
  }

  // ---------------- popups ----------------
  function predictedPopup(cell) {
    const v = App.cellView(cell);
    const color = App.urgencyColor(v.urgency);
    const thr = App.displayThreshold();
    const calibrated = v.prob !== null;
    const value = calibrated ? v.prob : v.score;
    let html = `<div style="--pc:${color}">
      <div class="pop-head"><span class="pop-dot"></span>${App.esc(t("urgency.name." + v.urgency))}</div>
      <div class="pop-label">${t(calibrated ? "pop.probability" : "pop.score")}</div>
      <div class="pop-big">${calibrated ? App.pct(v.prob) : App.fixed(v.score, 3)}</div>
      <div class="pop-bar"><span style="width:${Math.min(100, value * 100).toFixed(1)}%"></span><i style="left:${(thr * 100).toFixed(1)}%" title="${t("pop.thresholdTick", { t: thr.toFixed(2) })}"></i></div>`;
    if (calibrated) {
      const bin = App.reliabilityAt(v.prob);
      if (bin && bin.positive_rate !== null) {
        html += `<div class="pop-note">≈ ${t("pop.reliability", { n: Math.round(bin.positive_rate * 10) })}</div>`;
      }
    } else {
      html += `<div class="pop-note">${t("pop.notCalibrated")}</div>`;
    }
    html += `<div class="pop-rows">
        <div>${t("pop.estimated")}: <b>${App.fmtDate(v.date)}</b> <span class="muted">(${t("pop.dayN", { n: v.days })})</span></div>
        <div class="js-hist muted">${t("pop.historyLoading")}</div>
        <div>${t("pop.province")}: <b>${App.esc(App.provinceName(cell.province))}</b></div>
        <div>${t("pop.coords")}: ${cell.lat.toFixed(3)}°, ${cell.lon.toFixed(3)}°</div>
        ${calibrated ? `<div class="muted small">${t("pop.rawScore")}: ${App.fixed(cell.detection_score, 3)}</div>` : ""}
      </div>
      <div class="pop-sep"></div>
      <div class="pop-wx js-wx"><span class="muted">🌤 ${t("wx.loading")}</span></div>
    </div>`;
    return html;
  }

  async function fillPredicted(el, cell) {
    const m = App.meta();
    if (!el || !m) return;
    const hist = el.querySelector(".js-hist");
    const wx = el.querySelector(".js-wx");
    const snapshotName = state.snapshot || m.snapshot || (state.forecasts && state.forecasts.latest);
    const snapshotQuery = snapshotName ? `&snapshot_name=${encodeURIComponent(snapshotName)}` : "";
    App.getJSON(`/cell?lat_cell=${cell.lat_cell}&lon_cell=${cell.lon_cell}&until=${m.feature_cutoff}${snapshotQuery}`)
      .then(h => {
        if (!hist.isConnected) return;
        hist.classList.remove("muted");
        const year = h.history_days < 365 ? ` <span class="muted">(${t("pop.ofDays", { n: App.num(h.history_days) })})</span>` : "";
        hist.innerHTML = `${t("pop.past30", { n: `<b>${App.num(h.detections_30d)}</b>`, d: App.num(h.fire_days_30d) })}<br>
          <b>${App.num(h.fire_days_365d)}</b> ${t("pop.fireDaysYear")}${year}`;
      })
      .catch(() => { if (hist.isConnected) hist.textContent = t("pop.historyUnavailable"); });
    try {
      const w = await Weather.get(cell.lat, cell.lon, m.feature_cutoff);
      if (!wx.isConnected) return;
      const pos = temp => Math.max(0, Math.min(100, (temp - 10) / 35 * 100)).toFixed(1);
      wx.innerHTML = `
        <div class="wx-title">🌤 ${t("wx.title", { start: App.fmtDate(w.start), end: App.fmtDate(w.end) })}</div>
        <div>🌡️ ${t("wx.temp")}: <b>${App.num(w.tmin)}°C – ${App.num(w.tmax)}°C</b></div>
        <div class="temp-bar"><i style="left:${pos(w.tmin)}%"></i><i style="left:${pos(w.tmax)}%"></i></div>
        <div>🌧️ ${t("wx.rain7")}: <b>${App.num(w.rain, 1)} mm</b></div>
        <div>💨 ${t("wx.wind")}: <b>${App.num(w.wind, 1)} km/h</b></div>
        <div><span class="pop-chip">ⓘ ${t("wx.context")}</span></div>`;
    } catch (err) {
      if (wx.isConnected) wx.innerHTML = `<span class="muted">🌤 ${t("wx.unavailable")}</span>`;
    }
  }

  function observedPopup(o) {
    return `<div style="--pc:${App.cssVar("--observed")}">
      <div class="pop-head"><span class="pop-dot"></span>${t("pop.observedTitle")}</div>
      <div class="pop-rows">
        <div>${t("pop.observedDate")}: <b>${App.fmtDate(o.date)}</b></div>
        <div>${t("pop.detections")}: <b>${App.num(o.fire_count)}</b></div>
        <div>${t("pop.province")}: <b>${App.esc(App.provinceName(o.province))}</b></div>
        <div>${t("pop.coords")}: ${o.lat.toFixed(3)}°, ${o.lon.toFixed(3)}°</div>
      </div>
      <div class="pop-note" style="margin-top:8px">${t("pop.observedNote")}</div>
    </div>`;
  }

  function livePopup(a, code) {
    return `<div style="--pc:${App.cssVar("--live")}">
      <div class="pop-head"><span class="pop-dot"></span>${t("pop.liveTitle")}</div>
      <div class="pop-rows">
        <div>${t("pop.observedAt")}: <b>${App.fmtDateTime(a.observed_at)}</b> <span class="muted">(${App.timeAgo(a.observed_at)})</span></div>
        <div>${t("pop.satellite")}: <b>VIIRS ${App.esc(a.satellite)}</b></div>
        <div>${t("pop.confidence")}: <b>${App.esc(t("conf." + a.confidence))}</b></div>
        ${a.frp_mw !== null && a.frp_mw !== undefined ? `<div>FRP: <b>${App.num(a.frp_mw, 1)} MW</b></div>` : ""}
        <div>${t("pop.province")}: <b>${App.esc(App.provinceName(code))}</b></div>
        <div>${t("pop.coords")}: ${a.latitude.toFixed(4)}°, ${a.longitude.toFixed(4)}°</div>
      </div>
      <div style="margin-top:8px"><a class="pop-link" href="${Live.firmsLink(a)}" target="_blank" rel="noopener">🔗 ${t("live.verify")}</a></div>
    </div>`;
  }

  // ---------------- sidebar ----------------
  function snapshotOptions() {
    const list = (state.forecasts && state.forecasts.forecasts) || [];
    const m = App.meta();
    const latest = list.find(f => f.latest);
    const latestLabel = latest ? `${latest.base_date} · ${latest.model_version.slice(0, 6)}` : (m ? m.base_date : "—");
    let html = `<option value="">${t("sb.latest", { label: latestLabel })}</option>`;
    for (const f of list) {
      const tags = [];
      if (f.latest) tags.push(t("sb.tagLatest"));
      if (!f.calibrated) tags.push(t("sb.tagUncalibrated"));
      const age = App.daysBetween(f.base_date, App.todayISO());
      html += `<option value="${App.esc(f.name)}" ${state.snapshot === f.name ? "selected" : ""}>${f.base_date} · ${f.model_version.slice(0, 6)} — ${t("sb.flaggedN", { n: App.num(f.flagged) })} · ${t("sb.daysBehind", { n: App.num(age) })}${tags.length ? " · " + tags.join(", ") : ""}</option>`;
    }
    return html;
  }

  function provinceOptions() {
    const counts = new Map();
    for (const { cell } of App.flaggedCells({ ignoreProvince: true })) {
      if (cell.province) counts.set(cell.province, (counts.get(cell.province) || 0) + 1);
    }
    if (state.province && state.province !== "OUT" && !counts.has(state.province)) counts.set(state.province, 0);
    const rows = [...counts.entries()]
      .map(([code, n]) => ({ code, n, name: App.provinceName(code) }))
      .sort((a, b) => a.name.localeCompare(b.name, App.state.lang === "th" ? "th" : "en"));
    let html = `<option value="">${t("sb.allProvinces", { n: rows.length })}</option>`;
    for (const r of rows) {
      html += `<option value="${r.code}" ${state.province === r.code ? "selected" : ""}>${App.esc(r.name)} (${r.n})</option>`;
    }
    return html;
  }

  function performanceHTML() {
    const h = App.holdout();
    if (!h || !h.model || !h.model.classification) {
      return `<div class="note-box warn">${t("perf.unavailable")}</div>`;
    }
    const c = h.model.classification;
    const d = h.diagnostics || {};
    const cal = h.calibration;
    const baselines = Object.values(h.baselines || {}).map(b => b.classification).filter(Boolean);
    const bestBaselineAP = Math.max(...baselines.map(b => b.average_precision || 0));
    const auc = d.roc_auc ?? c.roc_auc;
    const badges = [`<span class="badge info">⏱ ${t("perf.badgeHoldout")}</span>`];
    if (cal && typeof cal.test_ece_calibrated === "number") {
      const good = cal.test_ece_calibrated < 0.05;
      badges.push(`<span class="badge ${good ? "" : "warn"}">${good ? "✓" : "⚠"} ${t("perf.badgeCalibrated", { ece: App.fixed(cal.test_ece_calibrated, 3) })}</span>`);
    } else {
      badges.push(`<span class="badge warn">⚠ ${t("perf.badgeUncalibrated")}</span>`);
    }
    if (baselines.length) {
      badges.push(c.average_precision > bestBaselineAP
        ? `<span class="badge">★ ${t("perf.badgeBeats")}</span>`
        : `<span class="badge neutral">${t("perf.badgeNotBeats")}</span>`);
    }
    const split = h.split || {};
    const recallTicks = baselines.filter(b => b.recall).map(b => `<span class="tick" style="left:${b.recall * 100}%" title="${t("perf.baselineTick")}"></span>`).join("");
    const aucPos = v => Math.max(0, Math.min(100, (v - 0.5) / 0.5 * 100));
    const aucTicks = baselines.filter(b => b.roc_auc).map(b => `<span class="tick" style="left:${aucPos(b.roc_auc)}%" title="${t("perf.baselineTick")}"></span>`).join("");
    const repeat = h.baselines && h.baselines.repeat_recent_detection_and_median_days;
    const aucClass = auc >= 0.8 ? "green" : "";
    return `
      <div class="badges">${badges.join("")}</div>
      <div class="perf-cards">
        <div class="perf-card">
          <div class="pc-label"><span>🔥 ${t("perf.detection")}</span><span title="${t("perf.detectionHelp")}">?</span></div>
          <div class="pc-value">${App.pct(c.recall)}</div>
          <div class="pc-desc">${t("perf.detectionDesc", { recall: App.pct(c.recall), t: App.fixed(c.threshold, 2), start: split.test_start || "—", end: split.test_end || "—" })}</div>
          <div class="scale"><span class="fill" style="width:${c.recall * 100}%"></span>${recallTicks}<span class="mark" style="left:${c.recall * 100}%"></span></div>
          <div class="scale-labels"><span>0</span><span>1</span></div>
        </div>
        <div class="perf-card ${aucClass}">
          <div class="pc-label"><span>📊 ${t("perf.ranking")}</span><span title="${t("perf.rankingHelp")}">?</span></div>
          <div class="pc-value">${App.fixed(auc, 3)}</div>
          <div class="pc-desc">${t("perf.rankingDesc", { base: repeat && repeat.classification ? App.fixed(repeat.classification.roc_auc, 3) : "—" })}</div>
          <div class="scale"><span class="fill" style="width:${aucPos(auc)}%"></span>${aucTicks}<span class="mark" style="left:${aucPos(auc)}%"></span></div>
          <div class="scale-labels"><span>0.5</span><span>1</span></div>
        </div>
      </div>`;
  }

  function baseDateHTML() {
    const m = App.meta();
    if (!m) {
      return `<div class="note-box warn">${t(state.errors.geojson ? "sb.dataMissing" : "common.loading")}</div>`;
    }
    const age = App.daysBetween(m.base_date, App.todayISO());
    const ageClass = age <= 1 ? "" : age <= 7 ? "warn" : "old";
    const kind = m.kind === "prospective"
      ? `<span class="badge info">${t("sb.kindProspective")}</span>`
      : `<span class="badge warn">${t("sb.kindHistorical")}</span>`;
    const note = t(m.kind === "prospective" ? "sb.noteProspective" : "sb.noteHistorical",
                   { cutoff: App.fmtDate(m.feature_cutoff), day1: App.fmtDate(App.addDays(m.base_date, 1)) });
    return `
      <div class="base-row"><span class="label-caps">${t("sb.baseDate")}</span>
        <span class="badge ${ageClass}">${t(age === 1 ? "sb.ageOne" : "sb.age", { n: App.num(age) })}</span>${kind}</div>
      <div class="base-date">${m.base_date}</div>
      <p class="base-note">${note}</p>
      <p class="issued">🛰️ ${t("sb.issued")}: <b>${App.fmtDateTime(m.issued_at)}</b> · ${t("sb.model")} <span class="mono">${App.esc((m.model_version || "").slice(0, 8))}</span></p>`;
  }

  function buildSidebar() {
    const sb = document.getElementById("sidebar");
    const scroll = sb.scrollTop;
    const liveWarn = ["offline", "nokey", "delayed"].includes(liveStateKey());
    const thr = state.threshold;
    sb.innerHTML = `
      <div class="sb-head">
        <h1 class="sb-title">🔥 <span>${t("sb.title")}</span></h1>
        <p class="sb-sub">${t("sb.subtitle")}</p>
      </div>
      <div class="sb-pills">
        <button type="button" class="layer-pill ${show.predicted ? "on" : ""}" data-layer="predicted" aria-pressed="${show.predicted}"><span class="pill-dot"></span>🔮 ${t("sb.layerPredicted")}</button>
        <button type="button" class="layer-pill ${show.observed ? "on" : ""}" data-layer="observed" aria-pressed="${show.observed}"><span class="pill-dot"></span>🛰️ ${t("sb.layerObserved")}</button>
        <button type="button" class="layer-pill ${show.live ? "on" : ""}" data-layer="live" aria-pressed="${show.live}"><span class="pill-dot"></span>🔥 ${t("sb.layerLive")}${liveWarn ? ` <span class="warn" title="${t("live.pillTitle." + liveStateKey())}">⚠</span>` : ""}</button>
      </div>
      <div class="tiles">
        <div class="tile high"><b id="cntHigh">0</b><span>${t("urgency.HIGH")}</span></div>
        <div class="tile medium"><b id="cntMedium">0</b><span>${t("urgency.MEDIUM")}</span></div>
        <div class="tile low"><b id="cntLow">0</b><span>${t("urgency.LOW")}</span></div>
        <div class="tile visible"><b id="cntVisible">0</b><span>${t("sb.visible")}</span></div>
      </div>
      <div class="sb-section">
        ${baseDateHTML()}
        <label class="field-label" for="snapshotSelect">${t("sb.viewFrom")}</label>
        <select id="snapshotSelect" class="select">${snapshotOptions()}</select>
        <label class="field-label" for="provinceSelect">${t("sb.filterProvince")}</label>
        <select id="provinceSelect" class="select">${provinceOptions()}</select>
        ${thr !== null ? `<div class="thr-badge"><span>⚙️ ${t("sb.displayThreshold", { t: thr.toFixed(2), m: App.modelThreshold().toFixed(2) })}</span><button type="button" data-act="reset-threshold" title="${t("thr.reset")}">✕</button></div>` : ""}
      </div>
      <div class="actions">
        <button type="button" class="btn" data-act="csv">📥 ${t("sb.exportCsv")}</button>
        <button type="button" class="btn" data-act="share">🔗 ${t("sb.share")}</button>
        <button type="button" class="btn" data-act="threshold">⚙️ ${t("sb.threshold")}</button>
        <button type="button" class="btn" data-act="details">📊 ${t("sb.details")}</button>
      </div>
      <div class="sb-section perf">
        <div class="perf-head"><span class="label-caps">🔥 ${t("perf.title")}</span><button type="button" class="info-btn" data-act="details" title="${t("sb.details")}">i</button></div>
        ${performanceHTML()}
      </div>
      <div class="sb-foot">${t("sb.footer")}</div>`;
    sb.scrollTop = scroll;
    updateCounts(show.predicted ? App.flaggedCells().length : 0);
  }

  function liveStateKey() {
    return document.getElementById("livePill").dataset.state || "connecting";
  }

  function updateCounts(visible) {
    const counts = { HIGH: 0, MEDIUM: 0, LOW: 0 };
    for (const { view } of App.flaggedCells()) if (view.urgency in counts) counts[view.urgency]++;
    const set = (id, v) => { const el = document.getElementById(id); if (el) el.textContent = App.num(v); };
    set("cntHigh", counts.HIGH);
    set("cntMedium", counts.MEDIUM);
    set("cntLow", counts.LOW);
    set("cntVisible", visible);
  }

  function buildLegend() {
    document.getElementById("legend").innerHTML = `
      <div class="legend-title">${t("legend.title")}</div>
      <div class="legend-item"><span class="legend-dot" style="background:var(--high)"></span>${t("legend.high")}</div>
      <div class="legend-item"><span class="legend-dot" style="background:var(--medium)"></span>${t("legend.medium")}</div>
      <div class="legend-item"><span class="legend-dot" style="background:var(--low)"></span>${t("legend.low")}</div>
      <div class="legend-item"><span class="legend-dot ring"></span>${t("legend.observed")}</div>
      <div class="legend-item"><span class="legend-dot" style="background:var(--live)"></span>${t("legend.live")}</div>`;
  }

  function setProvince(code, { refit = true } = {}) {
    code = Geo.byCode(code) ? code : "";
    const previous = refit ? state.province : "";
    state.province = code;
    if (outline) { map.removeLayer(outline); outline = null; }
    const p = code && code !== "OUT" ? Geo.byCode(code) : null;
    if (p) {
      outline = L.geoJSON(p.feature, {
        pane: "outline",
        style: { color: App.cssVar("--accent"), weight: 2, dashArray: "6 4", fillColor: App.cssVar("--accent"), fillOpacity: 0.05 },
        interactive: false,
      }).addTo(map);
      map.fitBounds(outline.getBounds(), { padding: [40, 40], maxZoom: 9 });
    } else if (!code && previous) {
      map.fitBounds(THAILAND, { padding: [10, 10] });
    }
    App.syncHash();
    App.emit("filters");
  }

  // ---------------- actions ----------------
  function exportCSV() {
    const m = App.meta();
    if (!m) return;
    const rows = App.flaggedCells().map(({ cell, view }) => ({ cell, view }));
    const csv = App.toCSV(rows, [
      { label: "province", value: r => r.cell.province ? Geo.name(r.cell.province, "en") : "Outside Thailand" },
      { label: "province_th", value: r => r.cell.province ? Geo.name(r.cell.province, "th") : "" },
      { label: "province_code", value: r => r.cell.province || "" },
      { label: "lat", value: r => r.cell.lat.toFixed(3) },
      { label: "lon", value: r => r.cell.lon.toFixed(3) },
      { label: "probability_7d", value: r => r.view.prob === null ? "" : r.view.prob.toFixed(4) },
      { label: "raw_model_score", value: r => r.cell.detection_score.toFixed(4) },
      { label: App.isCalibrated() ? "probability_level" : "score_level", value: r => r.view.urgency },
      { label: "estimated_first_detection", value: r => r.view.date },
      { label: "days_after_base", value: r => r.view.days },
      { label: "base_date", value: () => m.base_date },
      { label: "threshold", value: () => App.displayThreshold().toFixed(2) },
      { label: "model_version", value: () => m.model_version },
      { label: "kind", value: () => m.kind },
    ]);
    App.download(`wildfire_prediction_forecast_${m.base_date}_${m.model_version.slice(0, 6)}${state.province ? "_" + state.province : ""}.csv`, csv);
  }

  async function share() {
    App.syncHash();
    try {
      await navigator.clipboard.writeText(location.href);
      App.toast(t("sb.linkCopied"));
    } catch (err) {
      window.prompt(t("sb.copyThis"), location.href);
    }
  }

  // ---------------- details modal ----------------
  function metric(label, value, cls = "") {
    return `<div class="m-metric ${cls}"><span>${label}</span><b>${value}</b></div>`;
  }

  function openDetails() {
    const h = App.holdout();
    const m = App.meta();
    let html = "";
    if (h && h.model && h.model.classification) {
      const c = h.model.classification;
      const d = h.diagnostics || {};
      const cal = h.calibration || {};
      const split = h.split || {};
      const top = d.precision_at_top || {};
      html += `
        <div class="m-section">${t("det.heldOut")}</div>
        <p class="m-note" style="margin:0 0 10px">${t("det.task")}</p>
        <div class="m-grid">
          ${metric("ROC-AUC", App.fixed(d.roc_auc ?? c.roc_auc, 3))}
          ${metric(t("det.ap"), App.fixed(c.average_precision, 3))}
          ${metric(t("det.posRate"), App.pct(d.positive_rate, 1))}
          ${metric(t("det.ece"), App.fixed(d.ece, 4))}
        </div>
        <p class="m-note">${t("det.metricsNote")}</p>
        <div class="m-sub">${t("det.deployment")}</div>
        <div class="m-grid">
          ${metric(t("det.threshold"), App.fixed(c.threshold, 2))}
          ${metric(t("det.precision"), App.pct(c.precision, 1))}
          ${metric(t("det.recall"), App.pct(c.recall, 1))}
          ${metric("F1", App.pct(c.f1, 1))}
        </div>
        <p class="m-note">${t("det.thresholdNote")}</p>
        <div class="m-sub">${t("det.watchList")}</div>
        <div class="m-grid">
          ${["5", "10", "20"].filter(k => top[k]).map(k => metric(t("det.pAtTop", { k }), App.pct(top[k].precision, 1))).join("")}
        </div>
        ${top["20"] ? `<p class="m-note">${t("det.liftNote", { lift: App.fixed(top["20"].lift, 2) })}</p>` : ""}
        <div class="m-sub">${t("det.reliability")}</div>
        <div class="m-chart"><canvas id="relChart"></canvas></div>
        <p class="m-note">${t("det.reliabilityNote")}</p>
        <p class="m-note">${t("det.splitNote", { ts: split.test_start, te: split.test_end, rows: App.num(split.test_rows), trs: split.train_start, tre: split.train_end })}
          ${cal.method ? t("det.calibrationNote", { raw: App.fixed(cal.test_ece_raw, 3), cal: App.fixed(cal.test_ece_calibrated, 3) }) : ""}</p>
        <div class="m-divider"></div>`;
    } else {
      html += `<div class="note-box warn">${t("perf.unavailable")}</div><div class="m-divider"></div>`;
    }
    if (m) {
      const counts = { HIGH: 0, MEDIUM: 0, LOW: 0, NONE: 0 };
      const tiers = [0, 0, 0, 0];
      const provinces = new Map();
      for (const cell of state.cells) {
        if (!App.matchesProvince(cell.province)) continue;
        const v = App.cellView(cell);
        counts[v.flagged ? v.urgency : "NONE"]++;
        const p = v.prob ?? v.score;
        tiers[p >= 0.75 ? 0 : p >= 0.5 ? 1 : p >= 0.25 ? 2 : 3]++;
        if (v.flagged && cell.province) provinces.set(cell.province, (provinces.get(cell.province) || 0) + 1);
      }
      const total = tiers.reduce((a, b) => a + b, 0) || 1;
      const flagged = counts.HIGH + counts.MEDIUM + counts.LOW;
      const live = state.live && state.live.status !== "missing_key" ? App.num(state.live.total) : "—";
      const observed = state.observed.filter(o => App.matchesProvince(o.province)).length;
      const tierColors = ["var(--red)", "var(--orange)", "var(--amber)", "var(--green)"];
      const tierLabels = ["≥ 0.75", "0.50 – 0.75", "0.25 – 0.50", "< 0.25"];
      const top5 = [...provinces.entries()].sort((a, b) => b[1] - a[1]).slice(0, 5);
      html += `
        <div class="m-section">${t("det.snapshot", { date: m.base_date })}</div>
        <div class="m-grid">
          ${metric(t("det.flaggedCells"), App.num(flagged))}
          ${metric(t("det.observed", { date: App.fmtDate(m.feature_cutoff) }), App.num(observed))}
          ${metric(t("det.live"), live)}
          ${metric(t("det.provinceFilter"), state.province ? App.esc(App.provinceName(state.province === "OUT" ? null : state.province)) : t("det.all"))}
        </div>
        <div class="m-section" style="margin-top:18px">${t("det.urgencyDist")}</div>
        <div class="m-grid">
          ${metric(`<span>${t("urgency.HIGH")}</span>`, App.num(counts.HIGH), "high")}
          ${metric(`<span>${t("urgency.MEDIUM")}</span>`, App.num(counts.MEDIUM), "medium")}
          ${metric(`<span>${t("urgency.LOW")}</span>`, App.num(counts.LOW), "low")}
          ${metric(t("det.below"), App.num(counts.NONE))}
        </div>
        <p class="m-note">${t("det.urgencyNote", { t: App.displayThreshold().toFixed(2) })}</p>
        <div class="m-section" style="margin-top:18px">${t("det.tiers")}</div>
        ${tiers.map((n, i) => `<div class="tier-row"><span>${tierLabels[i]}</span><div class="tier-track"><span style="width:${(n / total * 100).toFixed(1)}%;background:${tierColors[i]}"></span></div><b>${App.num(n)}</b></div>`).join("")}
        <p class="m-note">${t(App.isCalibrated() ? "det.tiersNote" : "det.tiersNoteRaw")}</p>
        <div class="m-section" style="margin-top:18px">${t("det.topProvinces")}</div>
        <div class="prov-list">${top5.length ? top5.map(([code, n]) => `<div><span>${App.esc(App.provinceName(code))}</span><b>${App.num(n)}</b></div>`).join("") : `<div class="muted">—</div>`}</div>
        <div class="m-section" style="margin-top:22px">${t("det.glossary")}</div>
        <dl class="glossary">
          ${["auc", "ap", "precision", "recall", "f1", "ece", "brier", "lift"].map(k => `<dt>${t("gloss." + k + ".t")}</dt><dd>${t("gloss." + k + ".d")}</dd>`).join("")}
        </dl>`;
    }
    App.modal.open(t("det.title"), html, body => {
      const canvas = body.querySelector("#relChart");
      const bins = h && h.diagnostics && h.diagnostics.reliability;
      if (!canvas || !bins) return;
      const max = Math.max(...bins.map(b => b.count));
      App.Charts.make(canvas, {
        type: "scatter",
        data: {
          datasets: [
            { type: "line", data: [{ x: 0, y: 0 }, { x: 1, y: 1 }], borderColor: App.cssVar("--red"), borderDash: [5, 4], borderWidth: 1.5, pointRadius: 0 },
            { data: bins.filter(b => b.count).map(b => ({ x: b.mean_probability, y: b.positive_rate, n: b.count })),
              backgroundColor: App.cssVar("--green"), borderColor: "rgba(255,255,255,.7)", borderWidth: 1,
              pointRadius: ctx => 3 + 9 * Math.sqrt((ctx.raw ? ctx.raw.n : 0) / max) },
          ],
        },
        options: {
          responsive: true, maintainAspectRatio: false, animation: false,
          plugins: { legend: { display: false }, tooltip: { callbacks: { label: ctx => ctx.raw.n ? `${t("det.binTip", { p: App.pct(ctx.raw.x, 1), y: App.pct(ctx.raw.y, 1), n: App.num(ctx.raw.n) })}` : "" } } },
          scales: {
            x: { min: 0, max: 1, title: { display: true, text: t("det.meanProb") }, ticks: { stepSize: 0.25 } },
            y: { min: 0, max: 1, title: { display: true, text: t("det.actualRate") }, ticks: { stepSize: 0.25 } },
          },
        },
      });
    });
  }

  // ---------------- threshold modal ----------------
  function openThreshold() {
    const h = App.holdout();
    const table = (h && h.diagnostics && h.diagnostics.threshold_table) || [];
    const model = App.modelThreshold();
    let value = App.displayThreshold();
    const html = `
      <p class="m-note" style="margin-top:0">${t("thr.intro", { m: model.toFixed(2) })}</p>
      <div class="slider-row">
        <input type="range" id="thrRange" min="0.05" max="0.95" step="0.05" value="${value.toFixed(2)}" aria-label="${t("sb.threshold")}">
        <output id="thrOut">${value.toFixed(2)}</output>
      </div>
      <div class="m-grid" id="thrStats"></div>
      <p class="m-note">${t("thr.statsNote")}${App.isCalibrated() ? "" : " " + t("thr.uncalibratedNote")}</p>
      <div class="modal-actions">
        <button type="button" class="btn" id="thrReset">${t("thr.reset", { m: model.toFixed(2) })}</button>
        <button type="button" class="btn primary" id="thrApply">${t("thr.apply")}</button>
      </div>`;
    App.modal.open(t("thr.title"), html, body => {
      const range = body.querySelector("#thrRange");
      const render = () => {
        value = parseFloat(range.value);
        body.querySelector("#thrOut").textContent = value.toFixed(2);
        const row = table.find(r => Math.abs(r.threshold - value) < 1e-6);
        let shown = 0;
        for (const cell of state.cells) {
          if (!App.matchesProvince(cell.province)) continue;
          const score = typeof cell.probability === "number" ? cell.probability : cell.detection_score;
          if (score >= value) shown++;
        }
        body.querySelector("#thrStats").innerHTML = [
          metric(t("thr.precision"), row ? App.pct(row.precision, 1) : "—"),
          metric(t("thr.recall"), row ? App.pct(row.recall, 1) : "—"),
          metric(t("thr.flaggedShare"), row ? App.pct(row.flagged_share, 1) : "—"),
          metric(t("thr.shownNow"), App.num(shown)),
        ].join("");
      };
      range.addEventListener("input", render);
      render();
      body.querySelector("#thrReset").addEventListener("click", () => { range.value = model.toFixed(2); render(); });
      body.querySelector("#thrApply").addEventListener("click", () => {
        state.threshold = Math.abs(value - model) < 1e-9 ? null : value;
        App.modal.close();
        App.syncHash();
        App.emit("filters");
        buildSidebar();
      });
    });
  }

  // ---------------- events ----------------
  document.getElementById("sidebar").addEventListener("click", e => {
    const pill = e.target.closest("[data-layer]");
    if (pill) {
      const key = pill.dataset.layer;
      show[key] = !show[key];
      pill.classList.toggle("on", show[key]);
      pill.setAttribute("aria-pressed", show[key]);
      redraw();
      return;
    }
    const act = e.target.closest("[data-act]");
    if (!act) return;
    const action = act.dataset.act;
    if (action === "csv") exportCSV();
    else if (action === "share") share();
    else if (action === "threshold") openThreshold();
    else if (action === "details") openDetails();
    else if (action === "reset-threshold") { state.threshold = null; App.syncHash(); App.emit("filters"); buildSidebar(); }
  });
  document.getElementById("sidebar").addEventListener("change", e => {
    if (e.target.id === "snapshotSelect") App.selectSnapshot(e.target.value);
    else if (e.target.id === "provinceSelect") setProvince(e.target.value);
  });

  App.on("data", () => {
    buildSidebar();
    buildLegend();
    if (state.province) setProvince(state.province); else redraw();
  });
  App.on("meta", buildSidebar);
  App.on("filters", redraw);
  // Marker size and decluttering depend on zoom.
  map.on("zoomend", redraw);
  App.on("lang", () => {
    tiles.forEach(layer => layer.setLanguage(state.lang));
    buildSidebar(); buildLegend(); map.closePopup();
  });
  App.on("theme", () => { redraw(); if (outline) setProvince(state.province, { refit: false }); });
  let liveWarned = null;
  App.on("live", () => {
    drawLive();
    // Rebuild only when the live-layer warning changes, so an open dropdown is not reset every minute.
    const warn = ["offline", "nokey", "delayed"].includes(liveStateKey());
    if (warn !== liveWarned) { liveWarned = warn; buildSidebar(); }
  });
  App.on("show", view => { if (view === "map") setTimeout(fitOnce, 0); });
  App.on("layout", () => setTimeout(() => map.invalidateSize(), 280));

  setTiles();
  buildLegend();
  buildSidebar();

  // Used by other views: switch to the map, centre on a point and open a popup.
  function focus(lat, lon, kind, item) {
    App.go("map");
    setTimeout(() => {
      fitted = true;
      map.invalidateSize();
      map.setView([lat, lon], Math.max(map.getZoom(), 10));
      const latlng = L.latLng(lat, lon);
      if (kind === "live") openPopup(latlng, livePopup(item, Geo.locate(lat, lon)));
      else if (kind === "cell") openPopup(latlng, predictedPopup(item), el => fillPredicted(el, item));
    }, 60);
  }

  return { map, focus, openDetails };
})();

// Weather context from Open-Meteo (no key). Requests go directly from the browser.
const Weather = (() => {
  const cache = new Map();
  function get(lat, lon, endIso) {
    const key = `${lat.toFixed(2)},${lon.toFixed(2)},${endIso}`;
    if (cache.has(key)) return cache.get(key);
    const start = App.addDays(endIso, -6);
    const age = App.daysBetween(endIso, App.todayISO());
    const base = age > 5 ? "https://archive-api.open-meteo.com/v1/archive" : "https://api.open-meteo.com/v1/forecast";
    const params = new URLSearchParams({
      latitude: lat.toFixed(3), longitude: lon.toFixed(3), start_date: start, end_date: endIso,
      daily: "temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max", timezone: "Asia/Bangkok",
    });
    const promise = fetch(`${base}?${params}`)
      .then(r => { if (!r.ok) throw new Error("weather " + r.status); return r.json(); })
      .then(data => {
        const d = data.daily || {};
        const n = (d.time || []).length;
        const valid = arr => (arr || []).filter(v => typeof v === "number");
        const lastValid = arr => { const v = valid(arr); return v.length ? v[v.length - 1] : null; };
        if (!n || lastValid(d.temperature_2m_max) === null) throw new Error("weather empty");
        return {
          start: d.time[0], end: d.time[n - 1],
          tmin: lastValid(d.temperature_2m_min), tmax: lastValid(d.temperature_2m_max),
          rain: valid(d.precipitation_sum).reduce((a, b) => a + b, 0),
          wind: Math.max(...valid(d.wind_speed_10m_max)),
        };
      });
    cache.set(key, promise);
    promise.catch(() => cache.delete(key));
    return promise;
  }
  return { get };
})();
