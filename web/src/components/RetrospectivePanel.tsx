import { useMemo } from "react";
import type { FireFeature } from "../types";
import { fmtTr, useLang } from "../utils/i18n";
import { isActionableUrgency } from "../utils/risk";

interface Props {
  allFeatures: FireFeature[];  // ALL features from GeoJSON (not just current snapshot)
}

// Retrospective validation panel — answers "do past predictions actually
// match reality?". The append-mode GeoJSON tags every past prediction with
// validation_status:
//   "hit"    — the same cell recorded FIRMS during base_date+1 … +3
//   "miss"   — no FIRMS hotspot in that cell during the complete window
//   "future" — prediction window hasn't closed yet
//
// The operator headline is restricted to actionable HIGH/WATCH tiers.
// Including Background rows would answer a different question and would make
// the "alert track record" look like a model-wide precision estimate.
const MIN_RATE_SAMPLE = 20;
interface BaseDateStats {
  baseDate: string;
  hits: number;
  misses: number;
  future: number;
  total: number;
  hitRate: number;             // hits / decided (excludes future)
  byTier: Record<string, { hit: number; miss: number; future: number }>;
}

function aggregate(features: FireFeature[]): BaseDateStats[] {
  const map = new Map<string, BaseDateStats>();
  for (const f of features) {
    const p = f.properties;
    const bd = p.base_date;
    const st = p.validation_status;
    if (!bd || !st || !isActionableUrgency(p.urgency_level)) continue;
    let entry = map.get(bd);
    if (!entry) {
      entry = {
        baseDate: bd,
        hits: 0,
        misses: 0,
        future: 0,
        total: 0,
        hitRate: 0,
        byTier: {},
      };
      map.set(bd, entry);
    }
    entry.total += 1;
    if (st === "hit") entry.hits += 1;
    else if (st === "miss") entry.misses += 1;
    else if (st === "future") entry.future += 1;

    const tier = (p.urgency_level ?? "NONE") as string;
    if (!entry.byTier[tier]) entry.byTier[tier] = { hit: 0, miss: 0, future: 0 };
    if (st in entry.byTier[tier]) {
      entry.byTier[tier][st as "hit" | "miss" | "future"] += 1;
    }
  }
  return Array.from(map.values())
    .map((e) => ({
      ...e,
      hitRate: e.hits + e.misses > 0 ? e.hits / (e.hits + e.misses) : 0,
    }))
    .sort((a, b) => b.baseDate.localeCompare(a.baseDate));
}

const TIER_COLORS: Record<string, string> = {
  CRITICAL: "#ef4444",
  HIGH: "#f97316",
  MEDIUM: "#eab308",
  LOW: "#22c55e",
};

export default function RetrospectivePanel({ allFeatures }: Props) {
  const { t } = useLang();
  const stats = useMemo(() => aggregate(allFeatures), [allFeatures]);

  // Overall summary across actionable decided rows (excluding future).
  const overall = useMemo(() => {
    let hit = 0, miss = 0, future = 0;
    for (const s of stats) {
      hit += s.hits;
      miss += s.misses;
      future += s.future;
    }
    const decided = hit + miss;
    return { hit, miss, future, decided, rate: decided ? hit / decided : 0 };
  }, [stats]);

  if (stats.length === 0) {
    return (
      <div className="section">
        <h3>🎯 {t("retrospective.title")}</h3>
        <p style={{ fontSize: 11, color: "var(--text-3)" }}>
          {t("retrospective.empty")}
        </p>
      </div>
    );
  }

  return (
    <div className="section">
      <h3>🎯 {t("retrospective.title")}</h3>
      <p style={{ fontSize: 11, color: "var(--text-3)", marginBottom: 8 }}>
        {t("retrospective.subtitle")}
      </p>

      {/* Overall summary */}
      <div
        style={{
          padding: "10px 12px",
          background: "var(--surface-2)",
          borderRadius: 6,
          marginBottom: 10,
          border: "1px solid var(--border)",
        }}
      >
        <div style={{ fontSize: 10, fontWeight: 600, color: "var(--text-3)", textTransform: "uppercase", letterSpacing: "0.06em" }}>
          {t("retrospective.overall")}
        </div>
        <div style={{ display: "flex", alignItems: "baseline", gap: 8, marginTop: 4 }}>
          <span style={{ fontSize: 24, fontWeight: 700, color: overall.decided >= MIN_RATE_SAMPLE ? "#60a5fa" : "var(--text-3)" }}>
            {overall.decided >= MIN_RATE_SAMPLE
              ? `${(overall.rate * 100).toFixed(1)}%`
              : t("retrospective.small_sample")}
          </span>
          <span style={{ fontSize: 11, color: "var(--text-2)" }}>
            {fmtTr(t("retrospective.summary"), {
              hits: overall.hit,
              decided: overall.decided,
              pending: overall.future,
            })}
          </span>
        </div>
      </div>

      {/* Per base_date table */}
      <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
        {stats.slice(0, 8).map((s) => {
          const decided = s.hits + s.misses;
          const isFutureOnly = decided === 0 && s.future > 0;
          const canShowRate = decided >= MIN_RATE_SAMPLE;
          const color = canShowRate ? "#60a5fa" : "var(--text-3)";
          return (
            <div
              key={s.baseDate}
              style={{
                padding: "8px 10px",
                background: "var(--surface)",
                borderRadius: 4,
                border: "1px solid var(--border)",
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                fontSize: 12,
                gap: 8,
              }}
            >
              <div style={{ minWidth: 78 }}>
                <div style={{ fontWeight: 600, color: "var(--text)" }}>{s.baseDate}</div>
                <div style={{ fontSize: 10, color: "var(--text-3)", marginTop: 2 }}>
                  {fmtTr(t("retrospective.cells"), { count: s.total })}
                </div>
              </div>
              <div style={{ display: "flex", gap: 4, flex: 1 }}>
                {(["CRITICAL", "HIGH", "MEDIUM"] as const).map((tier) => {
                  const tierStats = s.byTier[tier] ?? {
                    hit: 0,
                    miss: 0,
                    future: 0,
                  };
                  const td = tierStats.hit + tierStats.miss;
                  if (td === 0 && tierStats.future === 0) return null;
                  return (
                    <div
                      key={tier}
                      title={fmtTr(t("retrospective.tier_tooltip"), {
                        tier: tier === "MEDIUM" ? "WATCH" : tier,
                        hit: tierStats.hit,
                        miss: tierStats.miss,
                        future: tierStats.future,
                      })}
                      style={{
                        padding: "2px 6px",
                        borderRadius: 3,
                        background: TIER_COLORS[tier] + "22",
                        border: `1px solid ${TIER_COLORS[tier]}66`,
                        fontSize: 10,
                        color: TIER_COLORS[tier],
                        fontWeight: 600,
                        whiteSpace: "nowrap",
                      }}
                    >
                      {tier === "MEDIUM" ? "W" : tier.charAt(0)} {td > 0 ? `${tierStats.hit}/${td}` : `+${tierStats.future}`}
                    </div>
                  );
                })}
              </div>
              <div style={{ textAlign: "right", minWidth: 56 }}>
                {isFutureOnly ? (
                  <span style={{ color: "var(--text-3)", fontSize: 11 }}>{t("retrospective.pending")}</span>
                ) : !canShowRate ? (
                  <>
                    <div style={{ fontSize: 11, fontWeight: 700, color }}>
                      {s.hits}/{decided}
                    </div>
                    <div style={{ fontSize: 9, color: "var(--text-3)" }}>
                      {fmtTr(t("retrospective.count_small"), { min: MIN_RATE_SAMPLE })}
                    </div>
                  </>
                ) : (
                  <>
                    <div style={{ fontSize: 16, fontWeight: 700, color }}>
                      {(s.hitRate * 100).toFixed(0)}%
                    </div>
                    <div style={{ fontSize: 9, color: "var(--text-3)" }}>
                      {t("retrospective.hit_rate")}
                    </div>
                  </>
                )}
              </div>
            </div>
          );
        })}
      </div>

      <p style={{ fontSize: 10, color: "var(--text-3)", marginTop: 8, lineHeight: 1.45 }}>
        {fmtTr(t("retrospective.methodology"), { min: MIN_RATE_SAMPLE })}
      </p>
    </div>
  );
}
