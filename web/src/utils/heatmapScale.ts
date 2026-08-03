import { URGENCY_COLORS } from "../constants";
import type {
  PredictionProperties,
  UrgencyThresholds,
} from "../types";
import { rawPredToProb, readProbability } from "./probability";

interface ProbabilityCutoffs {
  critical: number;
  high: number;
  medium: number;
}

const DEFAULT_CUTOFFS: ProbabilityCutoffs = {
  critical: 0.7,
  high: 0.5,
  medium: 0.35,
};

function clampProbability(value: number | null): number {
  return value === null ? 0 : Math.max(0, Math.min(1, value));
}

/**
 * Normalise both current probability cutoffs and legacy pseudo-day cutoffs
 * into one probability scale. Current binary snapshots use HIGH/MEDIUM in
 * [0, 1] and may use 1.01 as the "CRITICAL unused" sentinel.
 */
export function probabilityCutoffs(
  thresholds: UrgencyThresholds | null,
): ProbabilityCutoffs {
  if (!thresholds) return DEFAULT_CUTOFFS;

  const isProbabilityScale =
    thresholds.HIGH >= 0 &&
    thresholds.HIGH <= 1 &&
    thresholds.MEDIUM >= 0 &&
    thresholds.MEDIUM <= 1 &&
    thresholds.HIGH >= thresholds.MEDIUM;

  if (isProbabilityScale) {
    const critical =
      Number.isFinite(thresholds.CRITICAL) &&
      thresholds.CRITICAL >= thresholds.HIGH
        ? thresholds.CRITICAL
        : thresholds.HIGH;
    return {
      // Preserve >1 "unused" sentinels. Clamping 1.01 to 1 made an exact
      // 100% HIGH cell appear CRITICAL even though the model defines no
      // CRITICAL operating point.
      critical,
      high: thresholds.HIGH,
      medium: thresholds.MEDIUM,
    };
  }

  return {
    critical: clampProbability(rawPredToProb(thresholds.CRITICAL)),
    high: clampProbability(rawPredToProb(thresholds.HIGH)),
    medium: clampProbability(rawPredToProb(thresholds.MEDIUM)),
  };
}

export function heatmapProbability(
  properties: Pick<PredictionProperties, "probability" | "raw_prediction">,
): number | null {
  return readProbability(properties);
}

/**
 * Keep heatmap colors aligned with the operator-facing fixed tiers. Opacity,
 * rather than a relative per-snapshot normalisation, communicates variation
 * inside a tier.
 */
export function heatmapColor(
  probability: number,
  thresholds: UrgencyThresholds | null,
): string {
  const cutoffs = probabilityCutoffs(thresholds);
  if (probability >= cutoffs.critical) return URGENCY_COLORS.CRITICAL;
  if (probability >= cutoffs.high) return URGENCY_COLORS.HIGH;
  if (probability >= cutoffs.medium) return URGENCY_COLORS.MEDIUM;
  return URGENCY_COLORS.LOW;
}

/**
 * Absolute-risk opacity: a near-zero cell is nearly transparent even when it
 * is the highest value in an off-season snapshot. A cell at the WATCH/MEDIUM
 * cutoff reaches full intensity.
 */
export function heatmapRiskIntensity(
  probability: number,
  thresholds: UrgencyThresholds | null,
): number {
  const reference = Math.max(probabilityCutoffs(thresholds).medium, 0.01);
  const ratio = Math.max(0, Math.min(1, probability / reference));
  if (probability < reference) {
    // Background cells remain inspectable as pins but never get an
    // alert-like glow. This avoids making a near-WATCH background cell look
    // like a detected fire or an operational alert.
    return (ratio ** 1.5) * 0.36;
  }
  const aboveWatch = (probability - reference) / Math.max(1 - reference, 0.01);
  return 0.45 + Math.max(0, Math.min(1, aboveWatch)) * 0.55;
}
