// =====================================================================
// Analytics: charts for the selected forecast snapshot and the regional
// detection history used to build it.
// =====================================================================

const Analytics = (() => {
  const { state } = App;
  const t = (key, vars) => App.tierText(key, vars);
  let history = null;
  let historyError = false;

  const verticalMarker = {
    id: "verticalMarker",
    afterDatasetsDraw(chart, _args, opts) {
      if (opts.index === undefined || opts.index < 0) return;
      const { ctx, chartArea, scales } = chart;
      const x = scales.x.getPixelForValue(opts.index);
      ctx.save();
      ctx.strokeStyle = opts.color;
      ctx.setLineDash([5, 4]);
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(x, chartArea.top);
      ctx.lineTo(x, chartArea.bottom);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = opts.color;
      ctx.font = `600 11px ${App.cssVar("--font")}`;
      ctx.textAlign = x > chartArea.right - 120 ? "right" : "left";
      ctx.fillText(opts.label, x + (ctx.textAlign === "right" ? -6 : 6), chartArea.top + 12);
      ctx.restore();
    },
  };

  async function loadHistory() {
    if (history || historyError) return;
    try { history = (await App.getJSON("/history/daily?scope=thailand")).days; }
    catch (err) { historyError = true; }
  }

  function snapshotCells() {
    return state.cells.filter(c => App.matchesProvince(c.province)).map(cell => ({ cell, view: App.cellView(cell) }));
  }

  function render() {
    const root = document.getElementById("analyticsPage");
    App.Charts.destroyWithin(root);
    const m = App.meta();
    if (!m) {
      root.innerHTML = `<div class="card empty">${t(state.errors.geojson ? "sb.dataMissing" : "common.loading")}</div>`;
      return;
    }
    const rows = snapshotCells();
    const flagged = rows.filter(r => r.view.flagged);
    const calibrated = App.isCalibrated();
    const meanProb = flagged.length ? flagged.reduce((s, r) => s + (r.view.prob ?? r.view.score), 0) / flagged.length : null;
    const observed = state.observed.filter(o => App.matchesProvince(o.province)).reduce((s, o) => s + o.fire_count, 0);
    const filterLabel = state.province ? App.provinceName(state.province === "OUT" ? null : state.province) : t("det.all");
    root.innerHTML = `
      <div class="page-head">
        <div>
          <h1 class="page-title">📈 ${t("an.title")}</h1>
          <p class="page-sub">${t("an.subtitle", { date: m.base_date, model: (m.model_version || "").slice(0, 8), kind: t(m.kind === "prospective" ? "sb.kindProspective" : "sb.kindHistorical") })}</p>
        </div>
        <div class="head-right"><span class="badge neutral">${t("an.filter", { p: App.esc(filterLabel) })}</span></div>
      </div>
      <div class="grid four">
        <div class="stat blue"><div class="stat-label">${t("an.candidates")}</div><div class="stat-value">${App.num(rows.length)}</div><div class="stat-sub">${t("an.candidatesSub")}</div></div>
        <div class="stat orange"><div class="stat-label">${t("an.flagged")}</div><div class="stat-value">${App.num(flagged.length)}</div><div class="stat-sub">${t("an.flaggedSub", { t: App.displayThreshold().toFixed(2) })}</div></div>
        <div class="stat amber"><div class="stat-label">${t(calibrated ? "an.meanProb" : "an.meanScore")}</div><div class="stat-value">${meanProb === null ? "—" : calibrated ? App.pct(meanProb) : App.fixed(meanProb, 3)}</div><div class="stat-sub">${t("an.meanProbSub")}</div></div>
        <div class="stat red"><div class="stat-label">${t("an.observed")}</div><div class="stat-value">${App.num(observed)}</div><div class="stat-sub">${t("an.observedSub", { date: App.fmtDate(m.feature_cutoff) })}</div></div>
      </div>
      <div class="grid two section-gap">
        <div class="card">
          <div class="card-head"><div><div class="card-title">${t("an.timeline")}</div><div class="card-sub">${t("an.timelineSub")}</div></div></div>
          <div class="chart-box"><canvas id="anTimeline"></canvas></div>
        </div>
        <div class="card">
          <div class="card-head"><div><div class="card-title">${t(calibrated ? "an.histogram" : "an.histogramRaw")}</div><div class="card-sub">${t("an.histogramSub")}</div></div></div>
          <div class="chart-box"><canvas id="anHist"></canvas></div>
        </div>
        <div class="card">
          <div class="card-head"><div><div class="card-title">${t("an.urgency")}</div><div class="card-sub">${t("an.urgencySub")}</div></div></div>
          <div class="chart-box"><canvas id="anUrgency"></canvas></div>
        </div>
        <div class="card">
          <div class="card-head"><div><div class="card-title">${t("an.provinces")}</div><div class="card-sub">${t("an.provincesSub")}</div></div></div>
          <div class="chart-box"><canvas id="anProv"></canvas></div>
        </div>
      </div>
      <div class="card section-gap">
        <div class="card-head"><div><div class="card-title">${t("an.history")}</div><div class="card-sub">${t("an.historySub")}</div></div></div>
        <div class="chart-box tall"><canvas id="anHistory"></canvas></div>
      </div>`;
    drawCharts(rows, flagged, m);
  }

  function drawCharts(rows, flagged, m) {
    const colors = { HIGH: App.urgencyColor("HIGH"), MEDIUM: App.urgencyColor("MEDIUM"), LOW: App.urgencyColor("LOW") };
    const muted = App.cssVar("--faint");
    const accent = App.cssVar("--accent");
    const base = { responsive: true, maintainAspectRatio: false, animation: { duration: 300 } };

    // Timeline: flagged cells by estimated day of first detection.
    const days = Array.from({ length: App.horizonDays() }, (_, i) => i + 1);
    const perDay = days.map(d => flagged.filter(r => r.view.days === d).length);
    App.Charts.make(document.getElementById("anTimeline"), {
      type: "bar",
      data: {
        labels: days.map(d => [`+${d}`, App.fmtDate(App.addDays(m.base_date, d), { day: "numeric", month: "short" })]),
        datasets: [{ data: perDay, borderRadius: 4, backgroundColor: days.map(d => colors[d <= 2 ? "HIGH" : d <= 4 ? "MEDIUM" : "LOW"]) }],
      },
      options: { ...base, plugins: { legend: { display: false } }, scales: { y: { beginAtZero: true, title: { display: true, text: t("an.cells") } } } },
    });

    // Histogram of probabilities (or raw scores) across all candidates.
    const bins = 20;
    const counts = new Array(bins).fill(0);
    for (const r of rows) {
      const p = r.view.prob ?? r.view.score;
      counts[Math.min(bins - 1, Math.floor(p * bins))]++;
    }
    const thr = App.displayThreshold();
    App.Charts.make(document.getElementById("anHist"), {
      type: "bar",
      data: {
        labels: counts.map((_, i) => (i / bins).toFixed(2)),
        datasets: [{ data: counts, borderRadius: 2, backgroundColor: counts.map((_, i) => i / bins >= thr - 1e-9 ? accent : muted) }],
      },
      options: {
        ...base,
        plugins: { legend: { display: false }, tooltip: { callbacks: { title: items => `${(items[0].dataIndex / bins).toFixed(2)} – ${((items[0].dataIndex + 1) / bins).toFixed(2)}` } },
                   verticalMarker: { index: thr * bins - 0.5, color: App.cssVar("--text-2"), label: t("an.thresholdLine", { t: thr.toFixed(2) }) } },
        scales: { y: { type: "logarithmic", title: { display: true, text: t("an.cellsLog") } }, x: { ticks: { maxRotation: 0, autoSkip: true, maxTicksLimit: 6 } } },
      },
      plugins: [verticalMarker],
    });

    // Urgency doughnut.
    const urgency = { HIGH: 0, MEDIUM: 0, LOW: 0 };
    flagged.forEach(r => urgency[r.view.urgency]++);
    App.Charts.make(document.getElementById("anUrgency"), {
      type: "doughnut",
      data: {
        labels: [t("legend.high"), t("legend.medium"), t("legend.low"), t("det.below")],
        datasets: [{ data: [urgency.HIGH, urgency.MEDIUM, urgency.LOW, rows.length - flagged.length],
                     backgroundColor: [colors.HIGH, colors.MEDIUM, colors.LOW, App.cssVar("--border-2")],
                     borderColor: App.cssVar("--card"), borderWidth: 2 }],
      },
      options: { ...base, cutout: "62%", plugins: { legend: { position: "right" } } },
    });

    // Top provinces (ignores the province filter).
    const byProvince = new Map();
    for (const { cell } of App.flaggedCells({ ignoreProvince: true })) {
      if (cell.province) byProvince.set(cell.province, (byProvince.get(cell.province) || 0) + 1);
    }
    const top = [...byProvince.entries()].sort((a, b) => b[1] - a[1]).slice(0, 10);
    App.Charts.make(document.getElementById("anProv"), {
      type: "bar",
      data: {
        labels: top.map(([code]) => App.provinceName(code)),
        datasets: [{ data: top.map(([, n]) => n), backgroundColor: top.map(([code]) => code === state.province ? accent : App.cssVar("--blue")), borderRadius: 3 }],
      },
      options: { ...base, indexAxis: "y", plugins: { legend: { display: false } }, scales: { x: { beginAtZero: true } } },
    });

    drawHistory(m);
  }

  async function drawHistory(m) {
    const canvas = document.getElementById("anHistory");
    await loadHistory();
    if (!canvas || !canvas.isConnected) return;
    if (!history) {
      canvas.parentElement.innerHTML = `<div class="empty">${t("an.historyUnavailable")}</div>`;
      return;
    }
    const labels = history.map(d => d.date);
    App.Charts.make(canvas, {
      type: "line",
      data: {
        labels,
        datasets: [
          { label: t("an.detections"), data: history.map(d => d.detections), borderColor: App.cssVar("--accent"), backgroundColor: "rgba(249,115,22,.12)", fill: true, pointRadius: 0, borderWidth: 1.5, tension: 0.25 },
          { label: t("an.activeCells"), data: history.map(d => d.cells), borderColor: App.cssVar("--cyan"), pointRadius: 0, borderWidth: 1.2, tension: 0.25 },
        ],
      },
      options: {
        responsive: true, maintainAspectRatio: false, animation: false,
        interaction: { mode: "index", intersect: false },
        plugins: { legend: { position: "top", align: "end" }, verticalMarker: { index: labels.indexOf(m.feature_cutoff), color: App.cssVar("--text-2"), label: t("an.baseMarker", { date: m.feature_cutoff }) } },
        scales: { x: { ticks: { maxTicksLimit: 10, maxRotation: 0 } }, y: { beginAtZero: true } },
      },
      plugins: [verticalMarker],
    });
  }

  const active = () => state.view === "analytics";
  ["data", "filters", "lang", "theme", "meta"].forEach(evt => App.on(evt, () => { if (active()) render(); }));
  App.on("show", view => { if (view === "analytics") render(); });
  return { render };
})();
