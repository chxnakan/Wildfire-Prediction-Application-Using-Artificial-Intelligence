// Shared risk-semantics helpers for the operator UI.
//
// The binary model reports a calibrated probability of ≥1 FIRMS hotspot in a
// cell within the next few days — never an exact fire date. These pure helpers
// keep the tier vocabulary (High / Watch / Background) and its mapping onto the
// persisted absolute urgency levels (HIGH / MEDIUM / LOW) in ONE place so the
// filter, the map and any label can't drift. Unit-tested in risk.test.ts.

import type { UrgencyThresholds } from "../types";
import { readProbability } from "./probability";

export type RiskTier = "High" | "Watch" | "Background";
export type TierFilter = "Alerts" | "all" | RiskTier;
export type TierMode = "absolute" | "relative" | "days";

/** Complete retained snapshot: alerts plus clearly-labelled context. */
export const DEFAULT_TIER_FILTER: TierFilter = "all";

// User-facing tier → persisted GeoJSON urgency_level.
export const TIER_TO_URGENCY: Record<RiskTier, string> = {
  High: "HIGH",
  Watch: "MEDIUM",
  Background: "LOW",
};

// Persisted urgency_level → user-facing tier (inverse of the above).
export function urgencyToTier(urgency: string | null | undefined): RiskTier | null {
  switch (urgency) {
    case "CRITICAL":
    case "HIGH": return "High";
    case "MEDIUM": return "Watch";
    case "LOW": return "Background";
    default: return null; // OBSERVED / NONE / unknown → not a predicted tier
  }
}

// True when a feature belongs to the selected tier filter.
export function matchesTierFilter(
  urgency: string | null | undefined,
  filter: TierFilter,
): boolean {
  const known =
    urgency === "CRITICAL" || urgency === "HIGH" || urgency === "MEDIUM" ||
    urgency === "LOW";
  if (filter === "all") return known;
  if (filter === "Alerts") return isActionableUrgency(urgency);
  // CRITICAL is part of the precision-oriented High operator view. The current
  // model normally disables it, but legacy/current artifacts can still carry
  // the tier and must never disappear from the alert filter.
  if (filter === "High") return urgency === "CRITICAL" || urgency === "HIGH";
  return urgency === TIER_TO_URGENCY[filter];
}

/** CRITICAL/HIGH and MEDIUM/WATCH are operator alert tiers. */
export function isActionableUrgency(
  urgency: string | null | undefined,
): boolean {
  return urgency === "CRITICAL" || urgency === "HIGH" || urgency === "MEDIUM";
}

/**
 * Resolve threshold semantics without guessing from a particular snapshot.
 *
 * Explicit metadata wins. If an older artifact omitted the source string,
 * valid ordered 0..1 cutoffs are still absolute probabilities. The previous
 * UI classified that fallback as "relative", which incorrectly told users
 * fixed probability alerts were percentile ranks.
 */
export function resolveTierMode(
  thresholds: UrgencyThresholds | null,
  thresholdSource: string | null | undefined,
): TierMode {
  const source = (thresholdSource ?? "").trim().toLowerCase();
  if (source.startsWith("priority ranking")) return "relative";
  if (source.startsWith("absolute calibrated probability")) return "absolute";

  const high = Number(thresholds?.HIGH);
  const medium = Number(thresholds?.MEDIUM);
  const hasProbabilityCutoffs =
    Number.isFinite(high) &&
    Number.isFinite(medium) &&
    high >= 0 &&
    high <= 1 &&
    medium >= 0 &&
    medium <= 1 &&
    high >= medium;
  return hasProbabilityCutoffs ? "absolute" : "days";
}

// Backwards-compatible export for older imports. There is one canonical
// probability decoder; keeping a second, linear inverse here previously made
// legacy snapshots disagree with the map popup and heatmap.
export { readProbability };
