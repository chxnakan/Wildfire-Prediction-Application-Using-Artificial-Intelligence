import L from "leaflet";
import "leaflet.markercluster";
import { useEffect, useMemo, useRef } from "react";
import { fetchCellWeather } from "../api";
import {
  OBSERVED_DOT_FRAC,
  URGENCY_COLORS,
  URGENCY_DOT_FRAC,
} from "../constants";
import type {
  DisplayOptions,
  FireFeature,
  GistdaFeature,
  UrgencyLevel,
  UrgencyThresholds,
} from "../types";
import { createHeatmapLayer } from "../utils/heatmap";
import { LIVE_FIRE_COLOR } from "../utils/gistda";
import { fmtTr, useLang } from "../utils/i18n";
import {
  formatProbabilityPercent,
  probabilityPercent,
  readProbability,
} from "../utils/probability";
import { thaiFireSeasonForDate } from "../utils/seasonality";
import { isActionableUrgency } from "../utils/risk";
import {
  AnchoredCircle,
  clampPxForFrac,
  clampedRadiusMeters,
  dotRadiusMeters,
  gridSizeMeters,
} from "../utils/markers";

type Translator = (key: string, fallback?: string) => string;

// Popup HTML is assembled as strings and injected via Leaflet bindPopup /
// innerHTML. Any dynamic value that is (or could become) attacker-influenced
// — place names, upstream weather `reason` text, dates — MUST be escaped
// before interpolation so it renders as text, never markup. Numbers formatted
// with toFixed and our own i18n strings are safe and don't need this.
function esc(v: unknown): string {
  return String(v ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function fireWeatherLevelKey(
  level: string | null | undefined,
): "very_high" | "high" | "moderate" | "low" | "unknown" {
  switch ((level ?? "").trim().toLowerCase()) {
    case "สูงมาก":
    case "very high":
      return "very_high";
    case "สูง":
    case "high":
      return "high";
    case "ปานกลาง":
    case "moderate":
      return "moderate";
    case "ต่ำ":
    case "low":
      return "low";
    default:
      return "unknown";
  }
}

interface MapViewProps {
  observed: FireFeature[];
  predictedAll: FireFeature[]; // all predicted features for grid-size detection
  predictedVisible: FireFeature[]; // current day-filter applied
  liveFires: GistdaFeature[];
  thresholds: UrgencyThresholds | null;
  options: DisplayOptions;
  predictionEmptyState: {
    title: string;
    body: string;
    actionLabel?: string;
    onAction?: () => void;
  } | null;
}

// Render order: heatmap below, then observed / predicted / live-fire dots on
// top so popups stay clickable. Live fires render last so they sit above
// the historical-observation layer when both are on.
const LAYER_ORDER = ["heatmap", "observed", "predicted", "livefire"] as const;
type LayerKey = (typeof LAYER_ORDER)[number];

export default function MapView(props: MapViewProps) {
  const { t } = useLang();
  const containerRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<L.Map | null>(null);
  const layersRef = useRef<Record<LayerKey, L.Layer | null>>({
    heatmap: null,
    observed: null,
    predicted: null,
    livefire: null,
  });

  // Theme-aware tile layer — swaps the basemap when the user toggles light/dark
  // mode (ThemeToggle sets <html data-theme="light">). Kept as a ref so we can
  // remove/add it cleanly without recreating the whole map.
  const tileLayerRef = useRef<L.TileLayer | null>(null);

  const tileUrlFor = (theme: "light" | "dark") =>
    theme === "light"
      ? "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
      : "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png";

  const currentTheme = (): "light" | "dark" =>
    document.documentElement.getAttribute("data-theme") === "light" ? "light" : "dark";

  // One-time map init.
  useEffect(() => {
    if (mapRef.current || !containerRef.current) return;
    const map = L.map(containerRef.current, { center: [15.0, 101.0], zoom: 6 });
    tileLayerRef.current = L.tileLayer(tileUrlFor(currentTheme()), {
      attribution: "&copy; OpenStreetMap contributors &copy; CARTO",
    }).addTo(map);

    map.on("zoomend", () => reclampAllMarkers(layersRef.current, map));
    mapRef.current = map;

    // Leaflet does not automatically notice CSS-only container changes.
    // Collapsing/opening the sidebar could otherwise leave blank tiles or an
    // offset map until the next browser resize.
    const resizeObserver =
      typeof ResizeObserver !== "undefined"
        ? new ResizeObserver(() => {
            window.requestAnimationFrame(() => {
              if (mapRef.current) mapRef.current.invalidateSize({ pan: false });
            });
          })
        : null;
    resizeObserver?.observe(containerRef.current);

    // Watch for theme changes from ThemeToggle and swap tiles live.
    const observer = new MutationObserver(() => {
      if (!mapRef.current || !tileLayerRef.current) return;
      mapRef.current.removeLayer(tileLayerRef.current);
      tileLayerRef.current = L.tileLayer(tileUrlFor(currentTheme()), {
        attribution: "&copy; OpenStreetMap contributors &copy; CARTO",
      }).addTo(mapRef.current);
    });
    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["data-theme"],
    });

    // Listen for "firewatch:flyto" events dispatched by AlertToasts so
    // clicking an alert pans the map to that hotspot.
    const onFlyTo = (e: Event) => {
      const detail = (e as CustomEvent<{ lat: number; lon: number }>).detail;
      if (!detail || !mapRef.current) return;
      mapRef.current.flyTo([detail.lat, detail.lon], 11, { duration: 1.2 });
    };
    window.addEventListener("firewatch:flyto", onFlyTo);

    return () => {
      window.removeEventListener("firewatch:flyto", onFlyTo);
      observer.disconnect();
      resizeObserver?.disconnect();
      map.remove();
      mapRef.current = null;
      tileLayerRef.current = null;
    };
  }, []);

  // Stable grid-size for sizing dots (detected from the full set, not just visible).
  const gridM = useMemo(() => gridSizeMeters(props.predictedAll), [props.predictedAll]);

  // Rebuild layers whenever inputs change.
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;

    clearLayers(layersRef.current, map);

    if (props.options.showObserved) {
      layersRef.current.observed = buildObservedLayer(map, props.observed, t);
    }

    if (props.options.showPredicted) {
      // Background rows are contextual candidates, not alerts. Even when an
      // operator opts in to their pins, never give them an alert-like heat
      // halo; the heat layer is reserved for cells crossing HIGH/WATCH.
      const heatmapFeatures = props.predictedVisible.filter((feature) =>
        isActionableUrgency(feature.properties.urgency_level),
      );
      layersRef.current.heatmap = createHeatmapLayer(
        heatmapFeatures,
        props.thresholds,
        props.options.heatRadius
      );
      if (props.options.showCellPins) {
        layersRef.current.predicted = buildPredictedLayer(map, props.predictedVisible, gridM, t);
      }
    }

    if (props.options.showLiveFires && props.liveFires.length > 0) {
      layersRef.current.livefire = buildLiveFireLayer(map, props.liveFires, gridM, t);
    }

    addLayers(layersRef.current, map);
  }, [
    props.observed,
    props.predictedVisible,
    props.liveFires,
    props.thresholds,
    props.options.showObserved,
    props.options.showPredicted,
    props.options.showCellPins,
    props.options.showLiveFires,
    props.options.heatRadius,
    gridM,
    t,
  ]);

  return (
    <>
      <div id="map" ref={containerRef} />
      {props.predictionEmptyState && (
        <div className="map-prediction-empty" role="status" aria-live="polite">
          <strong>{props.predictionEmptyState.title}</strong>
          <span>{props.predictionEmptyState.body}</span>
          {props.predictionEmptyState.actionLabel &&
            props.predictionEmptyState.onAction && (
              <button type="button" onClick={props.predictionEmptyState.onAction}>
                {props.predictionEmptyState.actionLabel}
              </button>
            )}
        </div>
      )}
    </>
  );
}

// ───────────────────── helpers ─────────────────────

function clearLayers(layers: Record<LayerKey, L.Layer | null>, map: L.Map) {
  for (const k of LAYER_ORDER) {
    if (layers[k]) {
      map.removeLayer(layers[k]!);
      layers[k] = null;
    }
  }
}

function addLayers(layers: Record<LayerKey, L.Layer | null>, map: L.Map) {
  for (const k of LAYER_ORDER) if (layers[k]) map.addLayer(layers[k]!);
}

function reclampAllMarkers(layers: Record<LayerKey, L.Layer | null>, map: L.Map) {
  for (const k of ["observed", "predicted", "livefire"] as const) {
    const layer = layers[k];
    if (!layer) continue;
    const eachLayer = (layer as unknown as { eachLayer?: (cb: (m: L.Layer) => void) => void })
      .eachLayer;
    if (!eachLayer) continue;
    eachLayer.call(layer, (m) => {
      const marker = m as AnchoredCircle;
      if (marker._baseRadiusM == null || typeof marker.setRadius !== "function") return;
      marker.setRadius(
        clampedRadiusMeters(map, marker._anchorLat, marker._baseRadiusM, marker._minPx, marker._maxPx)
      );
    });
  }
}

function buildObservedLayer(
  map: L.Map,
  features: FireFeature[],
  t: Translator,
): L.Layer {
  // Cluster mode removed from the UI — observed markers are always individual
  // dots now, matching the simplified frontend/index.html layout.
  const layer = L.layerGroup();

  const renderer = L.canvas({ padding: 0.5 });
  const gridM = gridSizeMeters(features);
  const baseM = dotRadiusMeters(OBSERVED_DOT_FRAC, gridM);
  const px = clampPxForFrac(OBSERVED_DOT_FRAC);

  for (const f of features) {
    const [lon, lat] = f.geometry.coordinates;
    const p = f.properties;
    const radiusM = clampedRadiusMeters(map, lat, baseM, px.min, px.max);
    const marker = L.circle([lat, lon], {
      radius: radiusM,
      renderer,
      fillColor: "#ff5722",
      color: "#fff",
      weight: 0.4,
      fillOpacity: 0.55,
    }) as AnchoredCircle;
    marker._baseRadiusM = baseM;
    marker._minPx = px.min;
    marker._maxPx = px.max;
    marker._anchorLat = lat;
    marker.bindPopup(
      `<div class="popup">
         <b style="color:#ff5722;">${esc(t("map.observed.title"))}</b><br>
         <small>${esc(t("map.observed.date"))}: ${esc(p.date ?? "—")}</small><br>
         <small>${esc(t("map.observed.detections"))}: ${esc(p.fire_count ?? "—")}</small><br>
         <small>${esc(t("map.observed.location"))}: ${lat.toFixed(3)}°, ${lon.toFixed(3)}°</small>
       </div>`
    );
    (layer as unknown as { addLayer: (m: L.Layer) => void }).addLayer(marker);
  }
  return layer as unknown as L.Layer;
}

function buildPredictedLayer(map: L.Map, features: FireFeature[], gridM: number, t: Translator): L.Layer {
  const layer = L.layerGroup();
  const renderer = L.canvas({ padding: 0.5 });

  for (const f of features) {
    const [lon, lat] = f.geometry.coordinates;
    const p = f.properties;
    const rawUrgency = p.urgency_level ?? "NONE";
    const urgency: UrgencyLevel =
      Object.prototype.hasOwnProperty.call(URGENCY_COLORS, rawUrgency)
        ? rawUrgency as UrgencyLevel
        : "NONE";
    const color = URGENCY_COLORS[urgency] ?? URGENCY_COLORS.NONE;
    const frac = URGENCY_DOT_FRAC[urgency] ?? URGENCY_DOT_FRAC.NONE;
    const baseM = dotRadiusMeters(frac, gridM);
    const px = clampPxForFrac(frac);
    const radiusM = clampedRadiusMeters(map, lat, baseM, px.min, px.max);

    const marker = L.circle([lat, lon], {
      radius: radiusM,
      renderer,
      fillColor: color,
      color: urgency === "LOW" ? "#d1fae5" : "#fff",
      weight: urgency === "LOW" ? 0.35 : 0.4,
      fillOpacity: urgency === "LOW" ? 0.62 : 0.68,
    }) as AnchoredCircle;
    marker._baseRadiusM = baseM;
    marker._minPx = px.min;
    marker._maxPx = px.max;
    marker._anchorLat = lat;

    // The binary model reports a calibrated probability of ≥1 hotspot within
    // the next few days — NEVER an exact fire date. We deliberately do not
    // surface predicted_fire_date/days_until_fire here (they are internal
    // risk-window-end / rank-bucket fields, not per-cell date forecasts).
    const probability = readProbability(p);
    const histCount = p.historical_fire_count_30d;
    const fireDaysPerYear =
      p.fire_days_per_year == null ? null : Number(p.fire_days_per_year);
    const seasonalContext =
      probability !== null &&
      probability < 0.01 &&
      fireDaysPerYear !== null &&
      Number.isFinite(fireDaysPerYear) &&
      fireDaysPerYear > 10
        ? thaiFireSeasonForDate(p.base_date)
        : null;

    const urgencyMap: Record<UrgencyLevel, string> = {
      CRITICAL: t("map.urgency.critical"),
      HIGH:     t("map.urgency.high"),
      MEDIUM:   t("map.urgency.medium"),
      LOW:      t("map.urgency.low"),
      NONE:     t("map.urgency.none"),
    };
    const urgencyLabel = urgencyMap[urgency];

    // Interpret the persisted operating tier, not arbitrary round-number
    // probability bands. Statements such as "7 of 10 cells will burn" were too
    // strong under the documented seasonal prevalence drift.
    const probInterpretation =
      probability === null
        ? ""
        : urgency === "HIGH"
          ? t("map.prob.interp.high")
          : urgency === "MEDIUM"
            ? t("map.prob.interp.watch")
            : urgency === "LOW"
              ? t("map.prob.interp.background")
              : t("map.prob.interp.unavailable");

    const probPct = probabilityPercent(probability);
    const barPct = probPct ?? 0;

    let html =
      `<div class="popup" style="min-width:240px;">
         <div style="font-weight:600; color:${color}; font-size:14px; margin-bottom:4px;">
           ${urgencyLabel}
         </div>`;
    if (probPct !== null) {
      html +=
        `<div style="margin:6px 0;">
           <div style="font-size:11px; color:#9aa0aa;">${t("map.prob.title")}</div>
           <div style="font-size:26px; font-weight:700; line-height:1;">${formatProbabilityPercent(probability)}</div>
           <div style="margin-top:4px; height:6px; background:#22272e; border-radius:3px; overflow:hidden;">
             <div style="height:100%; width:${barPct}%; background:${color};"></div>
           </div>
           <div style="font-size:10px; color:#6c707a; margin-top:4px;">${probInterpretation}</div>
         </div>`;
    }
    // Honest horizon statement — the ONLY temporal claim the model supports.
    html += `<small>${t("map.popup.risk_window")}</small><br>`;
    if (histCount != null)
      html += `<small>${t("map.popup.fires_30d")}: <b>${esc(histCount)}</b></small><br>`;
    if (fireDaysPerYear !== null && Number.isFinite(fireDaysPerYear))
      html += `<small><b>${fireDaysPerYear.toFixed(1)}</b> ${t("map.popup.fires_per_year")}</small><br>`;
    if (seasonalContext !== null) {
      const seasonKey =
        seasonalContext === "in-season"
          ? "map.popup.season_context.in_season"
          : "map.popup.season_context.off_season";
      html += `<small style="display:block;margin:3px 0;color:#9aa0aa;">${t(seasonKey)}</small>`;
    }
    if (p.nearest_urban_area && p.nearest_urban_distance_km != null) {
      const distance = Number(p.nearest_urban_distance_km);
      const distanceText = Number.isFinite(distance)
        ? ` (${distance.toFixed(1)} km)`
        : "";
      html += `<small>${t("map.popup.nearest_city")}: ${esc(p.nearest_urban_area)}${distanceText}</small><br>`;
    }
    html += `<small>${t("map.popup.coords")}: ${lat.toFixed(3)}°, ${lon.toFixed(3)}°</small>`;
    const weatherSlotId = `wx-${lat.toFixed(4)}-${lon.toFixed(4)}-${(p.base_date || "").replace(/-/g, "")}`;
    html +=
      `<hr style="border:0;border-top:1px dashed var(--border,#24252a);margin:6px 0">
       <div id="${weatherSlotId}" class="popup-weather-slot">
         <small style="color:#9aa0aa">${t("map.weather.loading")}</small>
       </div>`;
    html += `</div>`;
    marker.bindPopup(html);

    const renderWeatherFailure = (slot: HTMLElement, reason: string) => {
      slot.dataset.loaded = "false";
      slot.innerHTML =
        `<small style="display:block;color:#9aa0aa;margin-bottom:5px">` +
        `${fmtTr(t("map.weather.unavailable"), { reason: esc(reason) })}</small>` +
        `<button type="button" data-weather-retry ` +
        `style="padding:3px 8px;border:1px solid #3a3b42;border-radius:5px;` +
        `background:#17181d;color:#d7d9df;cursor:pointer;font-size:10px">` +
        `↻ ${esc(t("analytics.error.retry"))}</button>`;
      slot
        .querySelector<HTMLButtonElement>("[data-weather-retry]")
        ?.addEventListener("click", loadWeather, { once: true });
    };

    // Fetch + inject weather when popup opens (lazy). Failed requests remain
    // retryable; the previous code marked failures as permanently "loaded".
    const loadWeather = () => {
      const slot = document.getElementById(weatherSlotId);
      if (
        !slot ||
        slot.dataset.loaded === "true" ||
        slot.dataset.loading === "true"
      ) {
        return;
      }
      slot.dataset.loading = "true";
      slot.innerHTML =
        `<small style="color:#9aa0aa">${t("map.weather.loading")}</small>`;
      const targetDate = p.base_date || undefined;
      fetchCellWeather(lat, lon, targetDate)
        .then((w) => {
          if (!w.available) {
            renderWeatherFailure(
              slot,
              w.reason || t("map.weather.all_sources_failed"),
            );
            return;
          }
          slot.dataset.loaded = "true";
          const fwi = w.fire_weather_index ?? { level: "—", color: "#9aa0aa", emoji: "❔" };
          const fwiLevelKey =
            fwi.level_key ?? fireWeatherLevelKey(fwi.level);
          const fwiLevel = t(`map.weather.fwi.level.${fwiLevelKey}`);
          const fwiColor =
            typeof fwi.color === "string" &&
            /^#[0-9a-f]{6}$/i.test(fwi.color)
              ? fwi.color
              : "#9aa0aa";
          const tmin =
            typeof w.temp_min_c === "number" && isFinite(w.temp_min_c)
              ? w.temp_min_c
              : null;
          const tmax =
            typeof w.temp_max_c === "number" && isFinite(w.temp_max_c)
              ? w.temp_max_c
              : null;
          const p7 =
            typeof w.precip_7d_mm === "number" && isFinite(w.precip_7d_mm)
              ? w.precip_7d_mm
              : null;
          const wind =
            typeof w.wind_max_kmh === "number" && isFinite(w.wind_max_kmh)
              ? w.wind_max_kmh
              : null;
          const tempPosition = (value: number) =>
            Math.max(2, Math.min(158, ((value - 15) / 30) * 160));
          const tempBar =
            tmin !== null && tmax !== null
              ? `<svg width="160" height="14" style="border-radius:7px;display:block;margin-top:3px">
               <defs><linearGradient id="tg-${weatherSlotId}" x1="0" x2="1">
                 <stop offset="0%" stop-color="#3b82f6"/>
                 <stop offset="50%" stop-color="#eab308"/>
                 <stop offset="100%" stop-color="#ef4444"/>
               </linearGradient></defs>
               <rect width="160" height="14" fill="url(#tg-${weatherSlotId})" rx="7"/>
               <line x1="${tempPosition(tmin)}" y1="0" x2="${tempPosition(tmin)}" y2="14" stroke="#fff" stroke-width="2"/>
               <line x1="${tempPosition(tmax)}" y1="0" x2="${tempPosition(tmax)}" y2="14" stroke="#fff" stroke-width="2"/>
             </svg>`
              : "";
          slot.innerHTML =
            `<div style="font-size:11px;color:#9aa0aa;margin-bottom:2px">${fmtTr(t("map.weather.title"), { date: esc(w.date) })}</div>
             ${w.source ? `<div style="font-size:9px;color:#6c707a;margin-bottom:3px">${fmtTr(t("map.weather.source"), { source: esc(w.source) })}</div>` : ""}
             <div style="display:flex;align-items:baseline;gap:6px">
               <small>${t("map.weather.temp")}</small>
               <b>${tmin !== null && tmax !== null ? `${tmin.toFixed(0)}°C – ${tmax.toFixed(0)}°C` : "—"}</b>
             </div>
             ${tempBar}
             <div style="margin-top:5px"><small>${fmtTr(t("map.weather.rain7d"), { value: p7 !== null ? p7.toFixed(1) : "—" })}</small></div>
             ${wind !== null ? `<div><small>${fmtTr(t("map.weather.wind"), { value: wind.toFixed(1) })}</small></div>` : ""}
             <div style="margin-top:6px;display:inline-flex;align-items:center;gap:4px;padding:2px 8px;border-radius:999px;background:${fwiColor}22;color:${fwiColor};border:1px solid ${fwiColor}66;font-size:11px;font-weight:700">
               ${fmtTr(t("map.weather.fwi"), { emoji: esc(fwi.emoji), level: esc(fwiLevel) })}
             </div>`;
        })
        .catch(() => {
          renderWeatherFailure(slot, t("map.weather.all_sources_failed"));
        })
        .finally(() => {
          slot.dataset.loading = "false";
        });
    };
    marker.on("popupopen", loadWeather);

    layer.addLayer(marker);
  }
  return layer;
}

// Live GISTDA VIIRS hotspots — distinct cyan dots so they don't visually
// merge with the urgency-coloured prediction layer.
function buildLiveFireLayer(
  map: L.Map,
  features: GistdaFeature[],
  gridM: number,
  t: Translator,
): L.Layer {
  const layer = L.layerGroup();
  const renderer = L.canvas({ padding: 0.5 });
  const frac = 0.22;
  const baseM = dotRadiusMeters(frac, gridM);
  const px = clampPxForFrac(frac);

  for (const feat of features) {
    const a = feat.attributes ?? {};
    const lat = Number(a.latitude);
    const lon = Number(a.longitude);
    if (!isFinite(lat) || !isFinite(lon)) continue;

    const radiusM = clampedRadiusMeters(map, lat, baseM, px.min, px.max);
    const marker = L.circle([lat, lon], {
      radius: radiusM,
      renderer,
      fillColor: LIVE_FIRE_COLOR,
      color: "#fff",
      weight: 0.4,
      fillOpacity: 0.7,
    }) as AnchoredCircle;
    marker._baseRadiusM = baseM;
    marker._minPx = px.min;
    marker._maxPx = px.max;
    marker._anchorLat = lat;

    const explicitDetectedAt = (a.detected_at_thailand ?? "").trim();
    const legacyDate = String(a.date ?? "");
    const legacyIso =
      legacyDate.length === 8
        ? `${legacyDate.slice(0, 4)}-${legacyDate.slice(4, 6)}-${legacyDate.slice(6, 8)}` +
          `T${(a.time ?? "").trim() || "00:00"}:00+07:00`
        : null;
    const parsedDate = explicitDetectedAt
      ? new Date(explicitDetectedAt)
      : legacyIso
        ? new Date(legacyIso)
        : a.date
        ? new Date(a.date)
        : null;
    const detectedText =
      parsedDate && Number.isFinite(parsedDate.getTime())
        ? parsedDate.toLocaleString()
        : "—";
    const conf = String(a.confident ?? "").trim() || "—";
    const lu = (a.lu_name ?? "").trim() || "—";
    const prov = (a.pv_tn ?? "").trim();
    const dist = (a.ap_tn ?? "").trim();
    const location = [prov, dist].filter(Boolean).join(" · ") || "—";
    const sat = (a.satellite ?? "").trim() || "VIIRS";

    marker.bindPopup(
      `<div class="popup">
         <b style="color:${LIVE_FIRE_COLOR};">${esc(fmtTr(t("map.live.title"), { satellite: sat }))}</b><br>
         <div class="popup-block">
           <b>${esc(detectedText)}</b><br>
           <small>${esc(location)}</small>
         </div>
         <small>${esc(t("map.live.land_use"))}: ${esc(lu)}</small><br>
         <small>${esc(t("map.live.confidence"))}: ${esc(conf)}</small><br>
         <small>${lat.toFixed(3)}°N, ${lon.toFixed(3)}°E</small>
       </div>`
    );
    layer.addLayer(marker);
  }
  return layer;
}
