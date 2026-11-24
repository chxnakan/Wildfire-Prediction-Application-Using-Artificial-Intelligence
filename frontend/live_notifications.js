// In-page notifications from the existing live:new feed; no email or push access.
const LiveNotifications = (() => {
  const STORAGE_KEY = "firewatch.live-notifications.v1";
  const RETENTION_MS = 48 * 60 * 60 * 1000;
  const WINDOW_MS = 24 * 60 * 60 * 1000;
  const MAX_IDS = 20000;
  const seen = new Map();
  const pending = new Map();
  const root = document.getElementById("liveNotification");
  let latest = null;

  function readSeen(now) {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (!raw || raw.length > 2000000) return;
      const saved = JSON.parse(raw);
      if (saved.version !== 1 || !Array.isArray(saved.ids)) return;
      for (const row of saved.ids.slice(-MAX_IDS)) {
        if (!Array.isArray(row) || typeof row[0] !== "string" || row[0].length > 128 ||
            !Number.isFinite(row[1]) || row[1] < now - RETENTION_MS || row[1] > now) continue;
        seen.set(row[0], Math.max(seen.get(row[0]) || 0, row[1]));
      }
    } catch (_) { /* Storage can be disabled; session deduplication still works. */ }
  }

  function saveSeen(now) {
    for (const [id, time] of seen) if (time < now - RETENTION_MS) seen.delete(id);
    const entries = [...seen].sort((a, b) => a[1] - b[1]).slice(-MAX_IDS);
    seen.clear();
    for (const [id, time] of entries) seen.set(id, time);
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify({ version: 1, ids: entries })); }
    catch (_) { /* Keep the in-memory map if browser storage is unavailable. */ }
  }

  function visible(item, now) {
    const observed = Date.parse(item.observed_at);
    return Number.isFinite(observed) && observed <= now && observed >= now - WINDOW_MS &&
      App.matchesProvince(Geo.locate(item.latitude, item.longitude));
  }

  function setText(id, value) {
    const el = document.getElementById(id);
    if (el.textContent !== value) el.textContent = value;
  }

  function render() {
    const now = Date.now();
    for (const [id, item] of pending) if (!visible(item, now)) pending.delete(id);
    latest = [...pending.values()].sort((a, b) => Date.parse(b.observed_at) - Date.parse(a.observed_at))[0] || null;
    root.hidden = !latest;
    if (!latest) return;
    setText("liveNotificationTitle", I18N.t("notify.title", { n: App.num(pending.size) }));
    setText("liveNotificationLocation", I18N.t("notify.location", {
      province: App.provinceName(Geo.locate(latest.latitude, latest.longitude)),
      lat: App.fixed(latest.latitude, 4), lon: App.fixed(latest.longitude, 4),
    }));
    setText("liveNotificationTime", I18N.t("notify.time", { time: App.fmtDateTime(latest.observed_at) }));
    setText("liveNotificationNote", I18N.t("notify.note"));
    setText("liveNotificationMap", I18N.t("notify.map"));
    setText("liveNotificationDismiss", I18N.t("notify.dismiss"));
    document.getElementById("liveNotificationClose").setAttribute("aria-label", I18N.t("notify.close"));
  }

  function receive(items) {
    if (!Array.isArray(items) || !["ok", "partial"].includes(App.state.live?.status)) return;
    const now = Date.now();
    // Re-read under the browser lock so multiple open tabs do not repeat a batch.
    readSeen(now);
    for (const item of items) {
      if (!item || typeof item.id !== "string" || !/^[a-zA-Z0-9_-]{1,128}$/.test(item.id) ||
          !Number.isFinite(item.latitude) || !Number.isFinite(item.longitude) ||
          !Number.isFinite(Date.parse(item.observed_at))) continue;
      if (seen.has(item.id)) continue;
      seen.set(item.id, now);
      if (visible(item, now)) pending.set(item.id, item);
    }
    saveSeen(now);
    render();
  }

  function dismiss() {
    pending.clear();
    latest = null;
    root.hidden = true;
  }

  App.on("live:new", items => {
    if (navigator.locks?.request) {
      navigator.locks.request(STORAGE_KEY, () => receive(items)).catch(() => receive(items));
    } else receive(items);
  });
  // The live feed deliberately treats its first successful response as a baseline.
  // Only subsequent live:new events can open this notification.
  App.on("live", render);
  App.on("lang", render);
  App.on("filters", render);
  document.getElementById("liveNotificationClose").addEventListener("click", dismiss);
  document.getElementById("liveNotificationDismiss").addEventListener("click", dismiss);
  document.getElementById("liveNotificationMap").addEventListener("click", () => {
    render();
    if (!latest) return;
    const item = latest;
    MapView.focus(item.latitude, item.longitude, "live", item);
    dismiss();
  });
  return {};
})();
