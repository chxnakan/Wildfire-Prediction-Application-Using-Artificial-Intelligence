import L from "leaflet";
import type { FireFeature, UrgencyThresholds } from "../types";
import { URGENCY_COLORS } from "../constants";
import {
  heatmapColor,
  heatmapProbability,
  heatmapRiskIntensity,
} from "./heatmapScale";

// Each pixel's color comes from the NEAREST cell's calibrated probability
// (averaging produced tier-mismatches around boundaries). Per-pixel alpha
// combines distance falloff with ABSOLUTE risk, so an off-season snapshot
// cannot make a tiny probability look intense merely because it is the
// largest value on that day.

interface Pt {
  lat: number;
  lon: number;
  value: number;
  color: string;
}

interface RGB { r: number; g: number; b: number }

function hexToRgb(hex: string): RGB {
  const h = hex.replace("#", "");
  return {
    r: parseInt(h.slice(0, 2), 16),
    g: parseInt(h.slice(2, 4), 16),
    b: parseInt(h.slice(4, 6), 16),
  };
}

const STRIDE = 4;
const MAX_ALPHA = 200;

export function createHeatmapLayer(
  features: FireFeature[],
  thresholds: UrgencyThresholds | null,
  radius: number
): L.Layer | null {
  if (!features || features.length === 0) return null;
  const points: Pt[] = features
    .map((f) => ({
      lat: f.geometry.coordinates[1],
      lon: f.geometry.coordinates[0],
      value: heatmapProbability(f.properties) ?? NaN,
      // Persisted tiers are the source of truth for map colour. This matters
      // for archived rows: current metadata thresholds must not be reapplied
      // to a historical probability and silently change its tier colour.
      color:
        f.properties.urgency_level &&
        f.properties.urgency_level in URGENCY_COLORS
          ? URGENCY_COLORS[f.properties.urgency_level]
          : heatmapColor(heatmapProbability(f.properties) ?? 0, thresholds),
    }))
    .filter((p) => isFinite(p.value));
  if (points.length === 0) return null;

  const cutoff = radius;
  const cutoffSq = cutoff * cutoff;

  // Inline GridLayer subclass — has access to `points`, `t`, `radius` via closure.
  const InterpolationLayer = L.GridLayer.extend({
    createTile(this: L.GridLayer, coords: L.Coords): HTMLCanvasElement {
      const tile = document.createElement("canvas");
      const size = this.getTileSize();
      tile.width = size.x;
      tile.height = size.y;
      const ctx = tile.getContext("2d")!;

      // _tileCoordsToBounds is internal but stable since Leaflet 1.0; cast through any.
      const tileBounds = (this as unknown as { _tileCoordsToBounds(c: L.Coords): L.LatLngBounds })
        ._tileCoordsToBounds(coords);
      const nw = tileBounds.getNorthWest();
      const se = tileBounds.getSouthEast();
      const lonSpan = se.lng - nw.lng;
      const latSpan = nw.lat - se.lat;
      if (lonSpan <= 0 || latSpan <= 0) return tile;

      const bufferLng = (radius / size.x) * lonSpan;
      const bufferLat = (radius / size.y) * latSpan;

      const localPoints: { px: number; py: number; value: number; color: string }[] = [];
      for (let i = 0; i < points.length; i++) {
        const p = points[i];
        if (p.lat < se.lat - bufferLat || p.lat > nw.lat + bufferLat) continue;
        if (p.lon < nw.lng - bufferLng || p.lon > se.lng + bufferLng) continue;
        const px = ((p.lon - nw.lng) / lonSpan) * size.x;
        const py = ((nw.lat - p.lat) / latSpan) * size.y;
        localPoints.push({ px, py, value: p.value, color: p.color });
      }
      if (localPoints.length === 0) return tile;

      const W = Math.ceil(size.x / STRIDE);
      const H = Math.ceil(size.y / STRIDE);
      const lowRes = ctx.createImageData(W, H);
      const data = lowRes.data;

      for (let py = 0; py < H; py++) {
        for (let px = 0; px < W; px++) {
          const x = (px + 0.5) * STRIDE;
          const y = (py + 0.5) * STRIDE;

          let nearestValue: number | null = null;
          let nearestColor: string | null = null;
          let nearestDistSq = Infinity;
          let blobAlpha = 0;

          for (let i = 0; i < localPoints.length; i++) {
            const lp = localPoints[i];
            const dx = lp.px - x;
            const dy = lp.py - y;
            const d2 = dx * dx + dy * dy;
            if (d2 > cutoffSq) continue;
            if (d2 < nearestDistSq) {
              nearestDistSq = d2;
              nearestValue = lp.value;
              nearestColor = lp.color;
            }
            const f = 1 - Math.sqrt(d2) / cutoff;
            const wAlpha = f * f;
            if (wAlpha > blobAlpha) blobAlpha = wAlpha;
          }

          const idx = (py * W + px) * 4;
          if (nearestValue === null) {
            data[idx + 3] = 0;
            continue;
          }
          const color = hexToRgb(
            nearestColor ?? heatmapColor(nearestValue, thresholds),
          );
          const riskIntensity = heatmapRiskIntensity(nearestValue, thresholds);
          data[idx] = color.r;
          data[idx + 1] = color.g;
          data[idx + 2] = color.b;
          data[idx + 3] = Math.round(
            Math.min(1, blobAlpha * 1.4) * MAX_ALPHA * riskIntensity,
          );
        }
      }

      const tmp = document.createElement("canvas");
      tmp.width = W;
      tmp.height = H;
      tmp.getContext("2d")!.putImageData(lowRes, 0, 0);
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = "high";
      ctx.drawImage(tmp, 0, 0, size.x, size.y);

      return tile;
    },
  });

  // L.GridLayer.extend() loses the constructor type — cast to a permissive
  // factory shape so we can pass the standard GridLayer options.
  type Ctor = new (opts: L.GridLayerOptions) => L.GridLayer;
  return new (InterpolationLayer as unknown as Ctor)({
    opacity: 0.85,
    keepBuffer: 2,
  });
}
