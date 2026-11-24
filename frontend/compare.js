// =====================================================================
// Predicted vs Actual: a retrospective check of the selected snapshot,
// held-out test results against baselines, and prospective scoring status.
// =====================================================================

const Compare = (() => {
  const { state } = App;
  const t = (key, vars) => App.tierText(key, vars);
  // Deduplicate only requests in flight; even completed outcomes can change
  // when late satellite observations arrive.
  const outcomes = new Map();
  let refreshTimer = null;
  let renderVersion = 0;
  let miniMap = null;

  function snapshotName() {
    const m = App.meta();
    return state.snapshot || (m && m.snapshot) || (state.forecasts && state.forecasts.latest) || null;
  }

  async function loadOutcomes(name) {
    if (outcomes.has(name)) return outcomes.get(name);
    const promise = App.getJSON(`/forecasts/${encodeURIComponent(name)}/outcomes?scope=thailand`)
      .finally(() => { if (outcomes.get(name) === promise) outcomes.delete(name); });
    outcomes.set(name, promise);
    return promise;
  }

  function kpi(cls, label, value, sub = "") {
    return `<div class="stat ${cls}"><div class="stat-label">${label}</div><div class="stat-value">${value}</div>${sub ? `<div class="stat-sub">${sub}</div>` : ""}</div>`;
  }

  function retroHTML(result) {
    if (!result) return `<div class="empty">${t("common.loading")}</div>`;
    if (result.status === "no_candidates") return `<div class="empty">${t("cmp.noCandidates")}</div>`;
    if (result.error) return `<div class="note-box warn">${t("cmp.unavailable")}</div>`;
    if (result.status === "pending") return `<div class="note-box">${t("cmp.pending", { end: result.window_end })}</div>`;
    if (result.status === "unknown_coverage") {
      return `<div class="note-box warn">${t("cmp.unknownCoverage", { dates: result.missing_dates.join(", ") })}</div>`;
    }
    const c = result.metrics.classification;
    const timing = result.metrics.timing;
    return `
      <div class="grid three">
        ${kpi("green", t("cmp.hits"), App.num(c.true_positive), t("cmp.hitsSub"))}
        ${kpi("amber", t("cmp.falseAlarms"), App.num(c.false_positive), t("cmp.falseAlarmsSub"))}
        ${kpi("red", t("cmp.misses"), App.num(c.false_negative), t("cmp.missesSub"))}
        ${kpi("blue", t("det.precision"), App.pct(c.precision, 1), t("cmp.precisionSub"))}
        ${kpi("violet", t("det.recall"), App.pct(c.recall, 1), t("cmp.recallSub"))}
        ${kpi("cyan", t("cmp.mae"), timing && timing.mae_days !== null ? `${App.num(timing.mae_days, 2)} ${t("cmp.days")}` : "—", t("cmp.maeSub"))}
      </div>
      <div class="grid two section-gap">
        <div style="position:relative;height:380px;border-radius:9px;overflow:hidden;border:1px solid var(--border)"><div id="cmpMap" style="position:absolute;inset:0"></div></div>
        <div>
          <div class="m-section">${t("cmp.legend")}</div>
          <div class="legend-item"><span class="legend-dot" style="background:var(--green)"></span>${t("cmp.legendHit")}</div>
          <div class="legend-item"><span class="legend-dot" style="background:var(--amber)"></span>${t("cmp.legendFalse")}</div>
          <div class="legend-item"><span class="legend-dot ring" style="border-color:var(--red)"></span>${t("cmp.legendMiss")}</div>
          <p class="m-note">${t("cmp.populationNote")}</p>
          <p class="m-note">${t("cmp.thresholdNote", { t: App.fixed(result.threshold, 2) })}</p>
        </div>
      </div>`;
  }

  function drawMiniMap(result) {
    const el = document.getElementById("cmpMap");
    if (miniMap) { miniMap.remove(); miniMap = null; }
    if (!el || !result || result.status !== "ok") return;
    // This map is replaced on language/filter changes; avoid a zoom transition
    // firing after its container has been removed.
    miniMap = L.map(el, { zoomControl: true, attributionControl: true, preferCanvas: true, zoomAnimation: false }).setView([13.2, 101], 5);
    App.basemap().forEach(layer => layer.addTo(miniMap));
    const green = App.cssVar("--green"), amber = App.cssVar("--amber"), red = App.cssVar("--red");
    const bounds = [];
    for (const c of result.cells) {
      if (!Geo.locate(c.lat_grid, c.lon_grid)) continue;
      const hit = c.flagged && c.actual_day > 0;
      const style = c.flagged
        ? { radius: 3.5, color: "rgba(255,255,255,.75)", weight: 1, fillColor: hit ? green : amber, fillOpacity: 0.95 }
        : { radius: 3.5, color: red, weight: 1.6, fill: false };
      const tip = c.flagged
        ? (hit ? t("cmp.tipHit", { p: c.predicted_day, a: c.actual_day }) : t("cmp.tipFalse", { p: c.predicted_day }))
        : t("cmp.tipMiss", { a: c.actual_day });
      L.circleMarker([c.lat_grid, c.lon_grid], style).bindTooltip(tip).addTo(miniMap);
      bounds.push([c.lat_grid, c.lon_grid]);
    }
    if (bounds.length) miniMap.fitBounds(bounds, { padding: [20, 20], maxZoom: 8 });
  }

  function baselineRows(h) {
    const rows = [{ key: "model", label: t("cmp.model"), data: h.model, hl: true }];
    const names = { training_prevalence_and_median_days: "cmp.basePrevalence", repeat_recent_detection_and_median_days: "cmp.baseRepeat" };
    for (const [key, value] of Object.entries(h.baselines || {})) rows.push({ key, label: t(names[key] || key), data: value });
    return rows;
  }

  function holdoutHTML(h) {
    if (!h || !h.model || !h.model.classification) return `<div class="note-box warn">${t("perf.unavailable")}</div>`;
    const c = h.model.classification;
    const timing = h.model.timing || {};
    const rows = baselineRows(h);
    const split = h.split || {};
    return `
      <div class="table-wrap">
        <table class="data">
          <thead><tr><th>${t("cmp.method")}</th><th>${t("det.precision")}</th><th>${t("det.recall")}</th><th>F1</th><th>${t("det.ap")}</th><th>ROC-AUC</th><th>Brier</th><th>${t("cmp.maeShort")}</th></tr></thead>
          <tbody>${rows.map(r => {
            const k = r.data.classification || {}, tm = r.data.timing || {};
            return `<tr class="${r.hl ? "hl" : ""}"><td>${App.esc(r.label)}</td><td>${App.pct(k.precision, 1)}</td><td>${App.pct(k.recall, 1)}</td><td>${App.fixed(k.f1, 3)}</td><td>${App.fixed(k.average_precision, 3)}</td><td>${App.fixed(k.roc_auc, 3)}</td><td>${App.fixed(k.brier_score, 3)}</td><td>${App.fixed(tm.mae_days, 2)}</td></tr>`;
          }).join("")}</tbody>
        </table>
      </div>
      <p class="m-note">${t("cmp.holdoutNote", { ts: split.test_start, te: split.test_end, rows: App.num(split.test_rows), t: App.fixed(c.threshold, 2) })}</p>
      <div class="grid two section-gap">
        <div><div class="chart-box short"><canvas id="cmpBars"></canvas></div></div>
        <div>
          <div class="m-section">${t("cmp.confusion")}</div>
          <div class="cm">
            <div class="hdr"></div><div class="hdr">${t("cmp.actualYes")}</div><div class="hdr">${t("cmp.actualNo")}</div>
            <div class="hdr">${t("cmp.predYes")}</div><div class="cell good"><b>${App.num(c.true_positive)}</b>${t("cmp.tp")}</div><div class="cell bad"><b>${App.num(c.false_positive)}</b>${t("cmp.fp")}</div>
            <div class="hdr">${t("cmp.predNo")}</div><div class="cell bad"><b>${App.num(c.false_negative)}</b>${t("cmp.fn")}</div><div class="cell good"><b>${App.num(c.true_negative)}</b>${t("cmp.tn")}</div>
          </div>
          <div class="m-grid" style="margin-top:12px">
            <div class="m-metric"><span>${t("cmp.mae")}</span><b>${App.fixed(timing.mae_days, 2)} ${t("cmp.days")}</b></div>
            <div class="m-metric"><span>RMSE</span><b>${App.fixed(timing.rmse_days, 2)} ${t("cmp.days")}</b></div>
            <div class="m-metric"><span>${t("cmp.within1")}</span><b>${App.pct(timing.within_one_day, 1)}</b></div>
            <div class="m-metric"><span>${t("cmp.baselineDays")}</span><b>${App.fixed(h.baseline_median_days, 1)} ${t("cmp.days")}</b></div>
          </div>
        </div>
      </div>`;
  }

  function prospectiveHTML(p) {
    if (!p) return `<div class="note-box">${t("cmp.noProspective")}</div>`;
    const runs = p.evaluated_runs || [];
    return `
      <div class="grid four">
        ${kpi("green", t("cmp.evaluated"), App.num(runs.length))}
        ${kpi("amber", t("cmp.pendingRuns"), App.num(p.pending_runs))}
        ${kpi("blue", t("cmp.historicalExcluded"), App.num(p.historical_runs_excluded))}
        ${kpi("red", t("cmp.unknownExcluded"), App.num((p.unknown_coverage_runs || 0) + (p.invalid_runs_excluded || 0)))}
      </div>
      ${runs.length ? `<div class="table-wrap section-gap"><table class="data"><thead><tr><th>${t("cmp.baseDate")}</th><th>${t("sb.model")}</th><th>${t("det.precision")}</th><th>${t("det.recall")}</th><th>${t("cmp.maeShort")}</th></tr></thead><tbody>
        ${runs.map(r => `<tr><td>${r.base_date}</td><td class="mono">${App.esc(r.model_version.slice(0, 8))}</td><td>${App.pct(r.metrics.classification.precision, 1)}</td><td>${App.pct(r.metrics.classification.recall, 1)}</td><td>${App.fixed(r.metrics.timing.mae_days, 2)}</td></tr>`).join("")}
      </tbody></table></div>` : ""}
      <p class="m-note">${t("cmp.prospectiveNote")}</p>`;
  }

  async function render() {
    clearTimeout(refreshTimer);
    const version = ++renderVersion;
    const root = document.getElementById("comparePage");
    App.Charts.destroyWithin(root);
    if (miniMap) { miniMap.remove(); miniMap = null; }
    const m = App.meta();
    const h = App.holdout();
    const name = snapshotName();
    root.innerHTML = `
      <div class="page-head">
        <div>
          <h1 class="page-title">🆚 ${t("cmp.title")}</h1>
          <p class="page-sub">${t("cmp.subtitle")}</p>
        </div>
      </div>
      <div class="card accent">
        <div class="card-head">
          <div>
            <div class="card-title">${t("cmp.retroTitle", { date: m ? m.base_date : "—", model: m ? (m.model_version || "").slice(0, 8) : "—" })}</div>
            <div class="card-sub" id="cmpWindow">${m ? t("cmp.retroSub", { base: m.base_date, start: App.addDays(m.base_date, 1), end: App.addDays(m.base_date, App.horizonDays()) }) : ""}</div>
          </div>
          ${m && !m.prospective ? `<span class="badge warn">${t("cmp.retroBadge")}</span>` : ""}
        </div>
        <div id="cmpRetro">${name ? retroHTML(null) : `<div class="empty">${t("sb.dataMissing")}</div>`}</div>
        ${m && !m.prospective ? `<div class="note-box warn">${t("cmp.retroWarning")}</div>` : ""}
      </div>
      <div class="card">
        <div class="card-head"><div><div class="card-title">${t("cmp.holdoutTitle")}</div><div class="card-sub">${t(h && h.weather ? "cmp.weatherHoldoutSub" : "cmp.holdoutSub")}</div></div></div>
        ${holdoutHTML(h)}
      </div>
      <div class="card">
        <div class="card-head"><div><div class="card-title">${t("cmp.prospectiveTitle")}</div><div class="card-sub">${t("cmp.prospectiveSub")}</div></div></div>
        ${prospectiveHTML(state.performance && state.performance.prospective)}
      </div>`;
    drawBars(h);
    if (!name) return;
    let result;
    try { result = await loadOutcomes(name); } catch (err) { result = { error: true }; }
    if (version !== renderVersion || state.view !== "compare" || snapshotName() !== name) return;
    const retro = document.getElementById("cmpRetro");
    if (!retro) return;
    retro.innerHTML = retroHTML(result);
    drawMiniMap(result);
    refreshTimer = setTimeout(() => { if (active()) render(); }, 60000);
  }

  function drawBars(h) {
    const canvas = document.getElementById("cmpBars");
    if (!canvas || !h) return;
    const rows = baselineRows(h);
    const palette = [App.cssVar("--accent"), App.cssVar("--faint"), App.cssVar("--blue")];
    App.Charts.make(canvas, {
      type: "bar",
      data: {
        labels: [t("det.ap"), "ROC-AUC", "F1"],
        datasets: rows.map((r, i) => {
          const k = r.data.classification || {};
          return { label: r.label, data: [k.average_precision, k.roc_auc, k.f1].map(v => v ?? 0), backgroundColor: palette[i % palette.length], borderRadius: 3 };
        }),
      },
      options: { responsive: true, maintainAspectRatio: false, animation: { duration: 300 },
                 plugins: { legend: { position: "bottom", labels: { boxWidth: 12 } } },
                 scales: { y: { beginAtZero: true, max: 1 } } },
    });
  }

  const active = () => state.view === "compare";
  ["data", "lang", "theme", "meta"].forEach(evt => App.on(evt, () => { if (active()) render(); }));
  App.on("show", view => {
    if (view === "compare") render();
    else {
      clearTimeout(refreshTimer);
      renderVersion++;
      if (miniMap) { miniMap.remove(); miniMap = null; }
    }
  });
  return { render };
})();
