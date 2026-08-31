import { useEffect, useMemo, useState } from "react";
import type { FireFeature, GeoJsonMetadata } from "../types";
import { downloadCsv } from "../utils/csvExport";
import { fmtTr, useLang } from "../utils/i18n";
import {
  ALL_PRIORITY_PROVINCES,
  buildPriorityCsvRows,
  buildPrioritySnapshot,
  filterPriorityRows,
  getPriorityFreshness,
  isPrioritySnapshotCurrent,
  type PriorityAreaRow,
} from "../utils/priorityAreas";
import "./PriorityAreas.css";

interface Props {
  features: FireFeature[];
  metadata?: GeoJsonMetadata | null;
  initialProvince?: string;
  initialLimit?: 10 | 20;
  onSelectCell?: (feature: FireFeature) => void;
}

function fileSafe(value: string): string {
  return value
    .normalize("NFKC")
    .replace(/[^\p{L}\p{N}._-]+/gu, "_")
    .replace(/^_+|_+$/g, "") || "all";
}

function tierLabel(row: PriorityAreaRow): string {
  const value = row.priorityTier ?? row.absoluteTier;
  if (value === "RELATIVE_TOP_10") return "RELATIVE TOP 10%";
  if (value === "RELATIVE_TOP_20") return "RELATIVE TOP 20%";
  if (value === "UNRANKED") return "NO RELATIVE BAND";
  if (value === "WATCH") return "LEGACY RELATIVE TOP 10%";
  if (value === "EXTENDED_WATCH") return "LEGACY RELATIVE TOP 20%";
  if (value === "MEDIUM") return "WATCH";
  if (value === "LOW" || value === "NONE") return "BACKGROUND";
  return value;
}

function tierClass(row: PriorityAreaRow): string {
  return (row.priorityTier ?? row.absoluteTier).toLowerCase().split("_").join("-");
}

function formatFireDays(value: number | null): string {
  return value === null ? "—" : value.toFixed(1);
}

function formatCity(row: PriorityAreaRow): string {
  if (!row.nearestCity) return "—";
  return row.nearestCityDistanceKm === null
    ? row.nearestCity
    : `${row.nearestCity} (${row.nearestCityDistanceKm.toFixed(1)} km)`;
}

export default function PriorityAreas({
  features,
  metadata = null,
  initialProvince = ALL_PRIORITY_PROVINCES,
  initialLimit = 10,
  onSelectCell,
}: Props) {
  const { lang, t } = useLang();
  const tr = (key: string, fallback: string, vars?: Record<string, unknown>) =>
    vars ? fmtTr(t(key, fallback), vars) : t(key, fallback);
  const snapshot = useMemo(
    () => buildPrioritySnapshot(features, metadata),
    [features, metadata],
  );
  const [province, setProvince] = useState(initialProvince);
  const [limit, setLimit] = useState<10 | 20>(initialLimit);

  useEffect(() => {
    if (
      province !== ALL_PRIORITY_PROVINCES &&
      !snapshot.provinces.includes(province)
    ) {
      setProvince(ALL_PRIORITY_PROVINCES);
    }
  }, [province, snapshot.provinces]);

  const visibleRows = useMemo(
    () => filterPriorityRows(snapshot.rows, province, limit),
    [snapshot.rows, province, limit],
  );
  const rankCounts = useMemo(() => {
    const counts = new Map<number, number>();
    for (const row of snapshot.rows) {
      counts.set(row.rank, (counts.get(row.rank) ?? 0) + 1);
    }
    return counts;
  }, [snapshot.rows]);
  const provinceCandidateCount = province === ALL_PRIORITY_PROVINCES
    ? snapshot.rows.length
    : snapshot.rows.filter((row) => row.province === province).length;
  const freshness = getPriorityFreshness(snapshot);
  const isCurrentSnapshot = isPrioritySnapshotCurrent(snapshot.baseDate);

  const freshnessAge = (() => {
    if (freshness.ageHours === null) {
      return tr("priority.freshness.unavailable", "Freshness unavailable");
    }
    if (freshness.isAheadOfBrowserClock) {
      return tr(
        "priority.freshness.future",
        "Timestamp is ahead of this browser clock",
      );
    }
    if (freshness.ageHours < 1) {
      return tr("priority.freshness.less_than_hour", "less than 1 hour ago");
    }
    if (freshness.ageHours < 48) {
      return tr("priority.freshness.hours", "{count} hours ago", {
        count: Math.floor(freshness.ageHours),
      });
    }
    return tr("priority.freshness.days", "{count} days ago", {
      count: Math.floor(freshness.ageHours / 24),
    });
  })();

  const freshnessTimestamp = (() => {
    if (!freshness.timestamp) return "—";
    const parsed = new Date(freshness.timestamp);
    if (Number.isNaN(parsed.getTime())) return freshness.timestamp;
    return new Intl.DateTimeFormat(lang === "th" ? "th-TH" : "en-GB", {
      dateStyle: "medium",
      timeStyle: "short",
      timeZone: "Asia/Bangkok",
    }).format(parsed);
  })();

  const sourceLabel = freshness.timestampKind === "source_data"
    ? tr("priority.source.latest_input", "Latest source input")
    : freshness.timestampKind === "snapshot_generated"
      ? tr("priority.source.snapshot_built", "Snapshot generated")
      : tr("priority.source.timestamp", "Source timestamp");

  const exportRows = () => {
    if (!visibleRows.length) return;
    const provinceSlug = province === ALL_PRIORITY_PROVINCES ? "all" : province;
    downloadCsv(
      `firewatch_priority_${snapshot.baseDate ?? "unknown"}_${fileSafe(provinceSlug)}_top${limit}.csv`,
      buildPriorityCsvRows(visibleRows, snapshot.rankingMode, snapshot.baseDate),
    );
  };

  return (
    <section className="priority-areas" aria-labelledby="priority-areas-title">
      <header className="priority-areas__header">
        <div>
          <p className="priority-areas__eyebrow">
            {tr("priority.eyebrow", "DAILY OPERATIONS BRIEF")}
          </p>
          <h2 id="priority-areas-title">
            {isCurrentSnapshot
              ? tr("priority.title", "Today's Priority Areas")
              : tr("priority.title.latest", "Latest Available Priority Areas")}
          </h2>
          <p className="priority-areas__subtitle">
            {tr(
              "priority.subtitle",
              "Inspect the highest-ranked retained cells first; ranking supports prioritisation and does not guarantee a fire.",
            )}
          </p>
        </div>
        <button
          type="button"
          className="priority-areas__export"
          onClick={exportRows}
          disabled={!visibleRows.length}
        >
          {tr("priority.export", "📥 Export displayed CSV")}
        </button>
      </header>

      <div className="priority-areas__provenance" aria-label={tr("priority.provenance.aria", "Snapshot provenance")}>
        <div>
          <span>{tr("priority.source", "Source")}</span>
          <strong>{tr("priority.source.geojson", "FireWatch model GeoJSON")}</strong>
        </div>
        <div>
          <span>{tr("priority.base_date", "Base date")}</span>
          <strong>{snapshot.baseDate ?? "—"}</strong>
        </div>
        <div>
          <span>{sourceLabel}</span>
          <strong>{freshnessTimestamp}</strong>
          <small>{freshnessAge}</small>
        </div>
      </div>

      {!snapshot.metadataMatchesSnapshot && (
        <p className="priority-areas__notice priority-areas__notice--warning" role="status">
          {tr(
            "priority.metadata_mismatch",
            "Metadata base date is missing or does not match the loaded rows. Showing the latest feature snapshot; metadata scope and freshness are withheld.",
          )}
        </p>
      )}

      {snapshot.baseDate && !isCurrentSnapshot && (
        <p className="priority-areas__notice priority-areas__notice--warning" role="status">
          {tr(
            "priority.snapshot.not_current",
            "The latest trusted snapshot is dated {date}, not today in Thailand. Treat it as stale and verify the data pipeline before field use.",
            { date: snapshot.baseDate ?? "—" },
          )}
        </p>
      )}

      {snapshot.rows.length > 0 && (
        <>
          <div className="priority-areas__method">
            <strong>
              {snapshot.rankingMode === "persisted"
                ? tr("priority.method.persisted.title", "Persisted operational ranking")
                : tr("priority.method.fallback.title", "Probability-order display fallback")}
            </strong>
            <span>
              {snapshot.rankingMode === "persisted"
                ? tr(
                  "priority.method.persisted.body",
                  "Uses the rank stored with this snapshot. It is relative priority, not an absolute alert.",
                )
                : tr(
                  "priority.method.fallback.body",
                  "A complete persisted rank was unavailable, so loaded cells use probability competition ranks. Equal probabilities share a rank; coordinates only stabilise row display and never create priority.",
                )}
            </span>
          </div>

          <p className="priority-areas__scope">
            {snapshot.retainedMapCandidateCount !== null && snapshot.totalScoredCells !== null
              ? tr(
                "priority.scope.full",
                "The map artifact retains {retained} candidate cells from {total} model-scored cells; {loaded} candidates for this snapshot are loaded here.",
                {
                  retained: snapshot.retainedMapCandidateCount.toLocaleString(),
                  total: snapshot.totalScoredCells.toLocaleString(),
                  loaded: snapshot.loadedCandidateCount.toLocaleString(),
                },
              )
              : tr(
                "priority.scope.retained_only",
                "Ranking is limited to the {loaded} retained map candidates loaded for this snapshot, not every model-scored grid cell.",
                { loaded: snapshot.loadedCandidateCount.toLocaleString() },
              )}
          </p>

          <div className="priority-areas__toolbar">
            <label>
              <span>{tr("priority.filter.province", "Province")}</span>
              <select value={province} onChange={(event) => setProvince(event.target.value)}>
                <option value={ALL_PRIORITY_PROVINCES}>
                  {tr("priority.filter.all_provinces", "All provinces")}
                </option>
                {snapshot.provinces.map((name) => (
                  <option key={name} value={name}>{name}</option>
                ))}
              </select>
            </label>
            <label>
              <span>{tr("priority.filter.show", "Rows")}</span>
              <select
                value={limit}
                onChange={(event) => setLimit(Number(event.target.value) === 20 ? 20 : 10)}
              >
                <option value={10}>{tr("priority.filter.top10", "First 10 ranks (ties included)")}</option>
                <option value={20}>{tr("priority.filter.top20", "First 20 ranks (ties included)")}</option>
              </select>
            </label>
            <p>
              {tr("priority.filter.summary", "Showing {shown} of {count} retained candidates", {
                shown: visibleRows.length,
                count: provinceCandidateCount,
              })}
            </p>
          </div>
        </>
      )}

      {snapshot.rows.length === 0 ? (
        <div className="priority-areas__empty" role="status">
          <strong>{snapshot.metadataMatchesSnapshot && snapshot.baseDate
            ? tr("priority.empty.no_alerts.title", "No HIGH/WATCH alerts in the current snapshot")
            : tr("priority.empty.title", "No active prediction snapshot")}</strong>
          <span>
            {snapshot.metadataMatchesSnapshot && snapshot.baseDate
              ? tr(
                "priority.empty.no_alerts.body",
                "No actionable cells were emitted for snapshot {date}. This does not prove there is no fire risk; check data freshness before field use. Background context is intentionally not promoted into a priority list.",
                { date: snapshot.baseDate },
              )
              : tr(
                "priority.empty.body",
                "No dated predicted cells are available, so no priority list is shown.",
              )}
          </span>
        </div>
      ) : visibleRows.length === 0 ? (
        <div className="priority-areas__empty" role="status">
          <strong>{tr("priority.empty.province_title", "No retained candidates for this province")}</strong>
          <span>{tr("priority.empty.province_body", "Choose another province or show all provinces.")}</span>
        </div>
      ) : (
        <div className="priority-areas__table-wrap">
          <table className="priority-areas__table">
            <thead>
              <tr>
                <th>{tr("priority.table.rank", "Competition rank")}</th>
                <th>{tr("priority.table.province", "Province")}</th>
                <th>
                  {snapshot.rankingMode === "persisted"
                    ? tr("priority.table.tier", "Priority tier")
                    : tr("priority.table.alert_tier_header", "Alert tier")}
                </th>
                <th className="priority-areas__numeric">{tr("priority.table.probability", "3-day probability")}</th>
                <th className="priority-areas__numeric">{tr("priority.table.history", "Historical fire days/year")}</th>
                <th>{tr("priority.table.city", "Nearest city")}</th>
                {onSelectCell && <th>{tr("priority.table.action", "Action")}</th>}
              </tr>
            </thead>
            <tbody>
              {visibleRows.map((row) => (
                <tr
                  key={`${snapshot.baseDate}-${row.latitude}-${row.longitude}`}
                >
                  <td className="priority-areas__rank">
                    #{row.rank}
                    {(rankCounts.get(row.rank) ?? 0) > 1 && (
                      <small>{tr("priority.table.tied", "Tied score")}</small>
                    )}
                  </td>
                  <td>{row.province}</td>
                  <td>
                    <span className={`priority-areas__tier priority-areas__tier--${tierClass(row)}`}>
                      {tierLabel(row)}
                    </span>
                    {row.priorityTier && (
                      <small>
                        {tr("priority.table.alert_tier", "Alert: {tier}", {
                          tier: tierLabel({ ...row, priorityTier: null }),
                        })}
                      </small>
                    )}
                  </td>
                  <td className="priority-areas__numeric priority-areas__probability">
                    {row.probabilityLabel}
                  </td>
                  <td className="priority-areas__numeric">{formatFireDays(row.historicalFireDaysPerYear)}</td>
                  <td>{formatCity(row)}</td>
                  {onSelectCell && (
                    <td>
                      <button
                        type="button"
                        className="priority-areas__view"
                        onClick={() => onSelectCell(row.feature)}
                        aria-label={tr("priority.table.view_map_for", "View rank {rank} on map", {
                          rank: row.rank,
                        })}
                      >
                        {tr("priority.table.view_map", "View on map")}
                      </button>
                    </td>
                  )}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
