import type { FireFeature, PredictionProperties } from "../types";
import { dateAdd } from "./dates";
import { probabilityPercent, readProbability } from "./probability";

const DEFAULT_HORIZON_DAYS = 3;

export function normaliseHorizonDays(value: unknown): number {
  const days = Number(value);
  return Number.isInteger(days) && days >= 1 && days <= 30
    ? days
    : DEFAULT_HORIZON_DAYS;
}

const HEADERS = [
  "lat",
  "lon",
  "risk_probability_pct",
  "risk_tier",
  "base_date",
  "risk_window_start",
  "risk_window_end",
  "historical_fire_count_30d",
  "fire_days_per_year",
  "tree_cover_pct_2000",
  "tree_loss_pct_recent",
  "nearest_urban_area",
  "nearest_urban_distance_km",
  "province",
] as const;

const escape = (value: unknown): string => {
  if (value == null) return "";
  const text = String(value);
  return /[",\r\n]/.test(text)
    ? `"${text.replace(/"/g, '""')}"`
    : text;
};

function operatorTier(level: PredictionProperties["urgency_level"]): string {
  if (level === "HIGH") return "HIGH";
  if (level === "MEDIUM") return "WATCH";
  if (level === "LOW") return "BACKGROUND";
  return level ?? "";
}

/**
 * Build the operator export from the model's supported output contract:
 * calibrated horizon probability, absolute tier, and one shared risk window.
 *
 * Deliberately excludes the legacy `predicted_fire_date` and
 * `days_until_fire` pseudo-day fields. The binary model cannot predict an
 * exact ignition date, so exporting those fields would give downstream users
 * a stronger claim than the model actually makes.
 */
export function buildCellsCsv(
  cells: FireFeature[],
  horizonDays: unknown = DEFAULT_HORIZON_DAYS,
): string {
  const rows = [HEADERS.map(escape).join(",")];
  const validatedHorizonDays = normaliseHorizonDays(horizonDays);

  for (const feature of cells) {
    const [lon, lat] = feature.geometry.coordinates;
    const props = feature.properties;
    const probability = probabilityPercent(readProbability(props));
    const baseDate = props.base_date ?? "";
    const row = [
      lat.toFixed(4),
      lon.toFixed(4),
      probability?.toFixed(2) ?? "",
      operatorTier(props.urgency_level),
      baseDate,
      baseDate ? dateAdd(baseDate, 1) : "",
      baseDate ? dateAdd(baseDate, validatedHorizonDays) : "",
      props.historical_fire_count_30d,
      props.fire_days_per_year,
      props.tree_cover_pct_2000,
      props.tree_loss_pct_recent,
      props.nearest_urban_area,
      props.nearest_urban_distance_km != null
        ? Number(props.nearest_urban_distance_km).toFixed(1)
        : "",
      props.province,
    ];
    rows.push(row.map(escape).join(","));
  }

  return rows.join("\r\n");
}

export function exportCellsCsv(
  cells: FireFeature[],
  baseDate: string,
  province: string,
  tier: string,
  horizonDays?: unknown,
): boolean {
  if (!cells.length) return false;

  const provinceSlug = province === "all"
    ? "all"
    : province.replace(/\s+/g, "_");
  const tierSlug = tier === "all" ? "all" : tier.toLowerCase();
  const filename =
    `wildfire_risk_${baseDate}_${provinceSlug}_${tierSlug}.csv`;

  // UTF-8 BOM keeps Thai province names readable when opened in Excel.
  const blob = new Blob(["\uFEFF", buildCellsCsv(cells, horizonDays)], {
    type: "text/csv;charset=utf-8",
  });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.style.display = "none";
  document.body.appendChild(anchor);
  anchor.click();
  document.body.removeChild(anchor);
  setTimeout(() => URL.revokeObjectURL(url), 100);
  return true;
}
