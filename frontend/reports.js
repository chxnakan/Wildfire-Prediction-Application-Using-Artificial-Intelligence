// =====================================================================
// Reports: training provenance, urgency distribution, feature importance,
// province breakdown and documented limitations, with CSV/PNG/JSON export.
// =====================================================================

const Reports = (() => {
  const { state } = App;
  const t = (key, vars) => App.tierText(key, vars);
  let chartType = "bar";

  function kv(rows) {
    return `<dl class="kv">${rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("")}</dl>`;
  }

  function urgencyCounts() {
    const counts = { HIGH: 0, MEDIUM: 0, LOW: 0, NONE: 0 };
    for (const cell of state.cells) {
      if (!App.matchesProvince(cell.province)) continue;
      const v = App.cellView(cell);
      counts[v.flagged ? v.urgency : "NONE"]++;
    }
    return counts;
  }

  function provinceRows() {
    const map = new Map();
    for (const { cell, view } of App.flaggedCells({ ignoreProvince: true })) {
      const key = cell.province || "OUT";
      const row = map.get(key) || { code: key, HIGH: 0, MEDIUM: 0, LOW: 0, total: 0 };
      row[view.urgency]++;
      row.total++;
      map.set(key, row);
    }
    return [...map.values()].sort((a, b) => b.total - a.total);
  }

  function summaryHTML() {
    const md = state.metadata;
    const h = App.holdout();
    if (!md || !h) return `<div class="note-box warn">${t("perf.unavailable")}</div>`;
    const split = h.split || {};
    const cal = h.calibration || {};
    const params = (md.model && md.model.params) || {};
    const totalDays = md.observation_start && md.observation_end ? App.daysBetween(md.observation_start, md.observation_end) + 1 : null;
    const sources = (h.sources || []).length;
    const dataset = kv([
      [t("rp.source"), t(md.context ? "rp.sourceContextValue" : md.weather ? "rp.sourceWeatherValue" : "rp.sourceValue")],
      [t("rp.files"), App.num(sources)],
      [t("rp.dateRange"), `${md.observation_start} → ${md.observation_end}`],
      [t("rp.totalDays"), `${App.num(totalDays)} <small>(${t("rp.noRecords", { n: App.num(md.regional_days_without_records) })})</small>`],
      [t("rp.cells"), `${App.num(md.candidate_grids)} <small>× ${md.grid_size}° ${t("rp.grid")}</small>`],
      [t("rp.gridDays"), App.num(md.daily_rows)],
      [t("rp.detections"), `${App.num(md.unique_detections)} <small>(${t("rp.duplicates", { n: App.num(md.duplicates_removed) })})</small>`],
      [t("rp.features"), `${App.num((md.features || []).length)} <small>(${t(md.feature_selection ? "rp.researchFeaturesNote" : md.weather ? "rp.weatherFeaturesNote" : "rp.featuresNote")})</small>`],
    ]);
    const model = kv([
      [t("rp.type"), App.esc(md.model ? md.model.type : "—")],
      [t("rp.task"), t("rp.taskValue")],
      [t("rp.horizon"), t("rp.horizonValue", { n: md.max_prediction_days || App.evaluationHorizon() })],
      [t("rp.version"), `<span class="mono">${App.esc(md.model_version || "—")}</span>`],
      ["n_estimators", App.num(params.n_estimators)],
      ["learning_rate", params.learning_rate ?? "—"],
      ["num_leaves", App.num(params.num_leaves)],
      ["min_child_samples", App.num(params.min_child_samples)],
      ["random_state", params.random_state ?? "—"],
      [t("det.threshold"), App.fixed(h.model.classification.threshold, 2)],
    ]);
    const training = kv([
      [t("rp.split"), `${t("rp.splitValue")} <small>(${t("rp.noShuffle")})</small>`],
      [t("rp.train"), `${split.train_start} → ${split.train_end} <small>(${App.num(split.train_rows)})</small>`],
      [t("rp.test"), `${split.test_start} → ${split.test_end} <small>(${App.num(split.test_rows)})</small>`],
      [t("rp.calibration"), cal.method ? t("rp.calibrationValue", { k: cal.folds || "—" }) : t("perf.badgeUncalibrated")],
      [t("rp.testEce"), cal.method ? `${App.fixed(cal.test_ece_raw, 3)} → ${App.fixed(cal.test_ece_calibrated, 3)} <small>(${t("rp.rawToCal")})</small>` : "—"],
      [t("rp.baselines"), t("rp.baselinesValue")],
      [t("rp.trainedAt"), md.trained_at ? App.fmtDateTime(md.trained_at, { dateStyle: "medium", timeStyle: "short" }) : "—"],
      [t("rp.trainTime"), md.training_seconds ? t("rp.seconds", { n: App.num(md.training_seconds, 1) }) : "—"],
    ]);
    const beats = Object.values(h.baselines || {}).every(b => (b.classification.average_precision || 0) < h.model.classification.average_precision);
    return `
      <div class="card-head">
        <div>
          <div class="card-title" style="font-size:19px">🧪 ${t("rp.summaryTitle")}</div>
          <div class="card-sub">${t("rp.summarySub", { at: md.trained_at ? App.fmtDateTime(md.trained_at, { dateStyle: "medium", timeStyle: "short" }) : "—" })}</div>
        </div>
        <div class="summary-badges">
          ${cal.method ? `<span class="badge ${cal.test_ece_calibrated < 0.05 ? "" : "warn"}">${cal.test_ece_calibrated < 0.05 ? "✓" : "⚠"} ${t("rp.badgeCalibrated")}</span>` : ""}
          <span class="badge">✓ ${t("rp.badgeSplit")}</span>
          <span class="badge ${beats ? "" : "neutral"}">${beats ? "✓" : "•"} ${t(beats ? "perf.badgeBeats" : "perf.badgeNotBeats")}</span>
        </div>
      </div>
      <div class="grid three">
        <div class="kv-card"><h4>📦 ${t("rp.dataset")}</h4>${dataset}</div>
        <div class="kv-card"><h4>🧠 ${t("rp.model")}</h4>${model}</div>
        <div class="kv-card"><h4>🎓 ${t("rp.training")}</h4>${training}</div>
      </div>
      <p class="limit-note">${t("rp.limitNote", { p: App.pct(h.model.classification.precision), r: App.pct(h.model.classification.recall), t: App.fixed(h.model.classification.threshold, 2), prev: App.pct(h.diagnostics ? h.diagnostics.positive_rate : null, 1) })}</p>`;
  }

  function featureHTML() {
    const rows = (state.metadata && state.metadata.feature_importance) || [];
    if (!rows.length) return `<div class="empty">${t("rp.noImportance")}</div>`;
    const max = Math.max(...rows.map(r => r.gain_pct)) || 1;
    const colors = ["var(--green)", "var(--lime)", "var(--blue)"];
    return `<div class="hbars">${rows.map((r, i) => {
      const color = colors[Math.min(2, Math.floor(i / Math.ceil(rows.length / 3)))];
      return `<div class="hbar" title="${App.esc(r.feature)} · ${t("rp.splits", { n: App.num(r.splits) })}"><span class="hbar-label mono">${App.esc(r.feature)}</span>
        <div class="hbar-track"><div class="hbar-fill" style="width:${Math.max(0.4, r.gain_pct / max * 88)}%"><span style="width:100%;background:${color}"></span></div><span class="hbar-val">${App.num(r.gain_pct, 1)}%</span></div></div>`;
    }).join("")}</div>
    <p class="m-note">${t("rp.featureNote")}</p>`;
  }

  function provinceHTML() {
    const all = provinceRows();
    const outside = all.find(r => r.code === "OUT");
    const rows = all.filter(r => r.code !== "OUT").slice(0, 15);
    if (!rows.length) return `<div class="empty">${t("al.none")}</div>`;
    const max = Math.max(...rows.map(r => r.total));
    return `<div class="hbars">${rows.map(r => `<div class="hbar"><span class="hbar-label">${App.esc(App.provinceName(r.code === "OUT" ? null : r.code))}</span>
      <div class="hbar-track"><div class="hbar-fill" style="width:${r.total / max * 88}%">
        ${App.URGENCY.filter(u => r[u]).map(u => `<span style="width:${r[u] / r.total * 100}%;background:${App.urgencyColor(u)}" title="${t("urgency." + u)}: ${r[u]}"></span>`).join("")}
      </div><span class="hbar-val">${App.num(r.total)}</span></div></div>`).join("")}</div>
      <p class="m-note">${t("rp.provinceNote")}${outside ? " " + t("rp.outsideNote", { n: App.num(outside.total) }) : ""}</p>`;
  }

  function limitationsHTML() {
    const h = App.holdout();
    const items = (h && h.limitations) || [];
    if (!items.length) return `<div class="empty">—</div>`;
    return `<ul class="limits">${items.map(item => `<li>${App.esc(item)}</li>`).join("")}</ul>
      <p class="m-note">${t("rp.limitationsNote")}</p>`;
  }

  function render() {
    const root = document.getElementById("reportsPage");
    App.Charts.destroyWithin(root);
    const m = App.meta();
    root.innerHTML = `
      <div class="page-head">
        <div>
          <h1 class="page-title">📊 ${t("rp.title")}</h1>
          <p class="page-sub">${t("rp.subtitle")}</p>
        </div>
        <div class="head-right">
          <div class="head-actions">
            <button type="button" class="btn primary" data-dl="all">📄 ${t("rp.allMetrics")}</button>
            <button type="button" class="btn" data-dl="metadata">📂 ${t("rp.metadataJson")}</button>
          </div>
          <div class="head-note">${t("rp.headNote")}</div>
        </div>
      </div>
      <div class="card accent">${summaryHTML()}</div>
      <div class="card">
        <div class="card-head">
          <div><div class="card-title">${t("rp.urgencyTitle", { date: m ? m.base_date : "—" })}</div><div class="card-sub">${t("rp.urgencySub", { t: App.displayThreshold().toFixed(2) })}</div></div>
          <div class="card-tools">
            <div class="seg"><button type="button" data-chart="bar" class="${chartType === "bar" ? "active" : ""}">📊 ${t("rp.bar")}</button><button type="button" data-chart="line" class="${chartType === "line" ? "active" : ""}">📈 ${t("rp.line")}</button></div>
            <button type="button" class="btn sm" data-dl="png">🖼️ PNG</button>
            <button type="button" class="btn sm" data-dl="urgency-csv">📥 CSV</button>
          </div>
        </div>
        <div class="chart-box"><canvas id="rpUrgency"></canvas></div>
        <div class="note-box">${t("rp.urgencyNote")}</div>
      </div>
      <div class="card">
        <div class="card-head">
          <div><div class="card-title">${t("rp.featureTitle", { n: ((state.metadata && state.metadata.feature_importance) || []).length })}</div><div class="card-sub">${t("rp.featureSub")}</div></div>
          <button type="button" class="btn sm" data-dl="features-csv">📥 CSV</button>
        </div>
        ${featureHTML()}
      </div>
      <div class="card">
        <div class="card-head">
          <div><div class="card-title">🇹🇭 ${t("rp.provinceTitle")}</div><div class="card-sub">${t("rp.provinceSub")}</div></div>
          <button type="button" class="btn sm" data-dl="province-csv">📥 CSV</button>
        </div>
        ${provinceHTML()}
      </div>
      <div class="card">
        <div class="card-head"><div><div class="card-title">⚠️ ${t("rp.limitations")}</div></div></div>
        ${limitationsHTML()}
      </div>`;
    drawUrgency();
  }

  function drawUrgency() {
    const canvas = document.getElementById("rpUrgency");
    if (!canvas) return;
    const c = urgencyCounts();
    const colors = [App.urgencyColor("HIGH"), App.urgencyColor("MEDIUM"), App.urgencyColor("LOW"), App.cssVar("--faint")];
    const data = [c.HIGH, c.MEDIUM, c.LOW, c.NONE];
    App.Charts.make(canvas, {
      type: chartType,
      data: {
        labels: [t("legend.high"), t("legend.medium"), t("legend.low"), t("det.below")],
        datasets: [{ data, backgroundColor: chartType === "bar" ? colors : "rgba(249,115,22,.15)", borderColor: chartType === "bar" ? colors : App.cssVar("--accent"),
                     borderWidth: chartType === "bar" ? 0 : 2, borderRadius: 4, pointBackgroundColor: colors, pointRadius: 5, fill: chartType === "line", tension: 0.25, maxBarThickness: 260 }],
      },
      options: {
        responsive: true, maintainAspectRatio: false, animation: { duration: 300 },
        plugins: { legend: { display: false } },
        scales: { y: { beginAtZero: true, title: { display: true, text: t("rp.cellCount") } } },
      },
      plugins: [{
        id: "valueLabels",
        afterDatasetsDraw(chart) {
          const { ctx } = chart;
          ctx.save();
          ctx.fillStyle = App.cssVar("--text");
          ctx.font = `700 12px ${App.cssVar("--font")}`;
          ctx.textAlign = "center";
          chart.getDatasetMeta(0).data.forEach((el, i) => ctx.fillText(App.num(data[i]), el.x, el.y - 8));
          ctx.restore();
        },
      }],
    });
  }

  function chartPNG(canvas, filename) {
    const out = document.createElement("canvas");
    out.width = canvas.width;
    out.height = canvas.height;
    const ctx = out.getContext("2d");
    ctx.fillStyle = App.cssVar("--card");
    ctx.fillRect(0, 0, out.width, out.height);
    ctx.drawImage(canvas, 0, 0);
    App.downloadDataURL(filename, out.toDataURL("image/png"));
  }

  function handleDownload(kind) {
    const m = App.meta();
    const stamp = m ? m.base_date : "snapshot";
    if (kind === "all") {
      App.download(`firewatch_metrics_${stamp}.json`, JSON.stringify({ metadata: state.metadata, performance: state.performance, snapshot: m }, null, 2), "application/json");
    } else if (kind === "metadata") {
      App.download(`firewatch_dataset_info.json`, JSON.stringify(state.metadata, null, 2), "application/json");
    } else if (kind === "png") {
      const canvas = document.getElementById("rpUrgency");
      if (canvas) chartPNG(canvas, `firewatch_${App.isCalibrated() ? "likelihood" : "score"}_${stamp}.png`);
    } else if (kind === "urgency-csv") {
      const c = urgencyCounts();
      const rows = ["HIGH", "MEDIUM", "LOW", "NONE"].map(k => ({ tier: k === "NONE" ? "below_threshold" : k, cells: c[k] }));
      App.download(`firewatch_${App.isCalibrated() ? "likelihood" : "score"}_${stamp}.csv`, App.toCSV(rows, [{ key: "tier" }, { key: "cells" }]));
    } else if (kind === "features-csv") {
      const rows = (state.metadata && state.metadata.feature_importance) || [];
      App.download("firewatch_feature_importance.csv", App.toCSV(rows, [{ key: "feature" }, { key: "gain_pct" }, { key: "splits" }]));
    } else if (kind === "province-csv") {
      const rows = provinceRows().map(r => ({ ...r, name_en: r.code === "OUT" ? "Outside Thailand" : Geo.name(r.code, "en"), name_th: r.code === "OUT" ? "" : Geo.name(r.code, "th") }));
      App.download(`firewatch_provinces_${stamp}.csv`, App.toCSV(rows, [
        { key: "code" }, { key: "name_en" }, { key: "name_th" }, { key: "HIGH" }, { key: "MEDIUM" }, { key: "LOW" }, { key: "total" }]));
    }
  }

  const root = document.getElementById("reportsPage");
  root.addEventListener("click", e => {
    const dl = e.target.closest("[data-dl]");
    if (dl) { handleDownload(dl.dataset.dl); return; }
    const ct = e.target.closest("[data-chart]");
    if (ct) {
      chartType = ct.dataset.chart;
      root.querySelectorAll("[data-chart]").forEach(b => b.classList.toggle("active", b === ct));
      drawUrgency();
    }
  });

  const active = () => state.view === "reports";
  ["data", "filters", "lang", "theme", "meta"].forEach(evt => App.on(evt, () => { if (active()) render(); }));
  App.on("show", view => { if (view === "reports") render(); });
  return { render };
})();
