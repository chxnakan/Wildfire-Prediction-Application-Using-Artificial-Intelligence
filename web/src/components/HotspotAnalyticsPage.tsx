import { useEffect, useMemo, useRef, useState } from "react";
import {
  Bar, BarChart, CartesianGrid, Cell, Pie, PieChart,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import {
  fetchHotspotAnalytics,
  type HotspotAnalyticsData,
} from "../api";
import type { GistdaFeature, LiveFireMeta } from "../types";
import { ageDays, parseGistdaTime } from "../utils/gistdaTime";
import { fmtTr, useLang } from "../utils/i18n";

interface Props {
  liveFires: GistdaFeature[];
  liveFireMeta: LiveFireMeta;
}

// ── Thai province → region mapping ──────────────────────────────────────────
const PROVINCE_TO_REGION: Record<string, string> = {
  // North
  "เชียงใหม่":"เหนือ","เชียงราย":"เหนือ","แม่ฮ่องสอน":"เหนือ","ลำปาง":"เหนือ",
  "ลำพูน":"เหนือ","พะเยา":"เหนือ","แพร่":"เหนือ","น่าน":"เหนือ","อุตรดิตถ์":"เหนือ",
  "ตาก":"เหนือ","สุโขทัย":"เหนือ","พิษณุโลก":"เหนือ","พิจิตร":"เหนือ",
  "กำแพงเพชร":"เหนือ","เพชรบูรณ์":"เหนือ","นครสวรรค์":"เหนือ",
  "อุทัยธานี":"เหนือ",
  // Northeast
  "นครราชสีมา":"ตะวันออกเฉียงเหนือ","บุรีรัมย์":"ตะวันออกเฉียงเหนือ",
  "สุรินทร์":"ตะวันออกเฉียงเหนือ","ศรีสะเกษ":"ตะวันออกเฉียงเหนือ",
  "อุบลราชธานี":"ตะวันออกเฉียงเหนือ","ยโสธร":"ตะวันออกเฉียงเหนือ",
  "อำนาจเจริญ":"ตะวันออกเฉียงเหนือ","ร้อยเอ็ด":"ตะวันออกเฉียงเหนือ",
  "มหาสารคาม":"ตะวันออกเฉียงเหนือ","กาฬสินธุ์":"ตะวันออกเฉียงเหนือ",
  "ขอนแก่น":"ตะวันออกเฉียงเหนือ","ชัยภูมิ":"ตะวันออกเฉียงเหนือ",
  "หนองบัวลำภู":"ตะวันออกเฉียงเหนือ","เลย":"ตะวันออกเฉียงเหนือ",
  "หนองคาย":"ตะวันออกเฉียงเหนือ","อุดรธานี":"ตะวันออกเฉียงเหนือ",
  "สกลนคร":"ตะวันออกเฉียงเหนือ","นครพนม":"ตะวันออกเฉียงเหนือ",
  "มุกดาหาร":"ตะวันออกเฉียงเหนือ","บึงกาฬ":"ตะวันออกเฉียงเหนือ",
  // Central
  "กรุงเทพมหานคร":"กลาง","นนทบุรี":"กลาง","ปทุมธานี":"กลาง",
  "พระนครศรีอยุธยา":"กลาง","อ่างทอง":"กลาง","สิงห์บุรี":"กลาง",
  "ชัยนาท":"กลาง","ลพบุรี":"กลาง","สระบุรี":"กลาง","นครนายก":"กลาง",
  "ปราจีนบุรี":"กลาง","ฉะเชิงเทรา":"กลาง","สมุทรปราการ":"กลาง",
  "สมุทรสาคร":"กลาง","สมุทรสงคราม":"กลาง","สุพรรณบุรี":"กลาง",
  "นครปฐม":"กลาง",
  // West
  "กาญจนบุรี":"ตะวันตก","ราชบุรี":"ตะวันตก","เพชรบุรี":"ตะวันตก",
  "ประจวบคีรีขันธ์":"ตะวันตก",
  // East
  "ระยอง":"ตะวันออก","ชลบุรี":"ตะวันออก","จันทบุรี":"ตะวันออก",
  "ตราด":"ตะวันออก","สระแก้ว":"ตะวันออก",
  // South
  "ชุมพร":"ใต้","ระนอง":"ใต้","สุราษฎร์ธานี":"ใต้","พังงา":"ใต้",
  "ภูเก็ต":"ใต้","กระบี่":"ใต้","ตรัง":"ใต้","พัทลุง":"ใต้",
  "สตูล":"ใต้","สงขลา":"ใต้","ปัตตานี":"ใต้","ยะลา":"ใต้",
  "นราธิวาส":"ใต้","นครศรีธรรมราช":"ใต้",
};

const LAND_USE_EN: Record<string, string> = {
  "ป่าอนุรักษ์": "Protected Forest",
  "เขต สปก.": "ALRO Land",
  "ป่าสงวนแห่งชาติ": "National Forest",
  "พื้นที่ริมทางหลวง": "Highway ROW",
  "พื้นที่เกษตร": "Agricultural",
  "ชุมชนและอื่น ๆ": "Communities",
};

const REGION_EN: Record<string, string> = {
  "เหนือ": "North", "ตะวันออกเฉียงเหนือ": "Northeast",
  "กลาง": "Central", "ตะวันตก": "West",
  "ตะวันออก": "East", "ใต้": "South",
};

const LAND_COLORS = ["#22c55e","#f59e0b","#3b82f6","#ec4899","#f97316","#8b5cf6","#6b7280"];
const REGION_COLORS: Record<string, string> = {
  "เหนือ": "#f97316", "ตะวันออกเฉียงเหนือ": "#ef4444",
  "กลาง": "#eab308", "ตะวันตก": "#06b6d4",
  "ตะวันออก": "#8b5cf6", "ใต้": "#22c55e",
};
const COUNTRY_COLOR = "#f97316";

export default function HotspotAnalyticsPage({
  liveFires,
  liveFireMeta,
}: Props) {
  const { lang, t } = useLang();
  const th = lang === "th";
  const [analyticsData, setAnalyticsData] = useState<HotspotAnalyticsData | null>(null);
  const [loading, setLoading] = useState(true);
  const [analyticsError, setAnalyticsError] = useState(false);
  const [retryKey, setRetryKey] = useState(0);
  const chartRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setAnalyticsError(false);
    fetchHotspotAnalytics(controller.signal)
      .then((d) => {
        if (!controller.signal.aborted) setAnalyticsData(d);
      })
      .catch((error: unknown) => {
        if (controller.signal.aborted) return;
        console.error(error);
        setAnalyticsError(true);
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, [retryKey]);

  // ── Derived stats from live GISTDA features ──────────────────────────────

  const landUseData = useMemo(() => {
    const counts: Record<string, number> = {};
    for (const f of liveFires) {
      const rawLandUse = String(f.attributes.lu_name ?? "").trim();
      const lu = rawLandUse || (th ? "ไม่ระบุ" : "Unknown");
      counts[lu] = (counts[lu] ?? 0) + 1;
    }
    return Object.entries(counts)
      .sort((a, b) => b[1] - a[1])
      .map(([name, value], i) => ({
        name: th ? name : (LAND_USE_EN[name] ?? name),
        nameTh: name,
        value,
        fill: LAND_COLORS[i % LAND_COLORS.length],
      }));
  }, [liveFires, th]);

  const regionData = useMemo(() => {
    const counts: Record<string, number> = {};
    for (const f of liveFires) {
      const pv = String(f.attributes.pv_tn ?? "").trim();
      const region = PROVINCE_TO_REGION[pv] ?? (th ? "ไม่ระบุ" : "Other");
      counts[region] = (counts[region] ?? 0) + 1;
    }
    const order = ["เหนือ","ตะวันออกเฉียงเหนือ","กลาง","ตะวันตก","ตะวันออก","ใต้"];
    return order
      .filter((r) => counts[r] !== undefined)
      .map((r) => ({
        name: th ? r : (REGION_EN[r] ?? r),
        count: counts[r] ?? 0,
        fill: REGION_COLORS[r] ?? "#6b7280",
      }))
      .concat(
        Object.entries(counts)
          .filter(([r]) => !order.includes(r))
          .map(([r, n]) => ({ name: r, count: n, fill: "#6b7280" }))
      )
      .sort((a, b) => b.count - a.count);
  }, [liveFires, th]);

  const provinceData = useMemo(() => {
    const counts: Record<string, number> = {};
    for (const f of liveFires) {
      const pv = String(f.attributes.pv_tn ?? "").trim();
      if (!pv) continue;
      counts[pv] = (counts[pv] ?? 0) + 1;
    }
    return Object.entries(counts)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 10)
      .map(([name, count]) => ({ name, count }));
  }, [liveFires]);

  const timeSummary = useMemo(() => {
    const bins = [
      { label: "<1h",  maxH: 1,   count: 0, fill: "#ef4444" },
      { label: "1–3h", maxH: 3,   count: 0, fill: "#f97316" },
      { label: "3–6h", maxH: 6,   count: 0, fill: "#f59e0b" },
      { label: "6–12h",maxH: 12,  count: 0, fill: "#facc15" },
      { label: "12–24h",maxH: 24, count: 0, fill: "#a3e635" },
      { label: ">24h", maxH: Infinity, count: 0, fill: "#d1d5db" },
    ];
    let timestampUnavailable = 0;
    for (const f of liveFires) {
      const parsed = parseGistdaTime(
        f.attributes.date as number | string | null | undefined,
        f.attributes.time as number | string | null | undefined,
      );
      const age = ageDays(parsed?.ms ?? null);
      if (!parsed || parsed.precision !== "minute" || age == null) {
        timestampUnavailable += 1;
        continue;
      }
      const h = age * 24;
      for (let i = 0; i < bins.length; i++) {
        const prev = i === 0 ? 0 : bins[i-1].maxH;
        if (h <= bins[i].maxH && (i === 0 ? h >= prev : h > prev)) {
          bins[i].count++;
          break;
        }
      }
    }
    return { bins, timestampUnavailable };
  }, [liveFires]);
  const timeData = timeSummary.bins;

  const transboundaryData = useMemo(() => {
    if (!analyticsData?.transboundary) return [];
    return analyticsData.transboundary.map((d) => ({
      ...d,
      fill: d.country === "Thailand" ? "#22c55e" : COUNTRY_COLOR,
    }));
  }, [analyticsData]);
  const transboundaryMeta = analyticsData?.transboundary_metadata;
  const transboundaryUnavailable =
    analyticsError || transboundaryMeta?.available === false;

  const historicalData = useMemo(() => {
    return (analyticsData?.historical_burned_mrai ?? []).map((d) => ({
      label: th ? `พ.ศ. ${d.year}` : `${d.year_ce}`,
      burned: d.burned_mrai,
    }));
  }, [analyticsData, th]);

  const historicalYears =
    (analyticsData?.historical_burned_mrai ?? [])
      .map((d) => d.year_ce)
      .filter((year) => Number.isFinite(year));
  const historicalRange = historicalYears.length > 0
    ? `${Math.min(...historicalYears)}–${Math.max(...historicalYears)}`
    : "—";

  const hasCachedLiveData = liveFires.length > 0;
  const hasUsableLiveState =
    liveFireMeta.status === "ok" || hasCachedLiveData;
  const feedState =
    liveFireMeta.status === "loading"
      ? hasCachedLiveData
        ? "refreshing"
        : "loading"
      : liveFireMeta.status === "error"
        ? hasCachedLiveData
          ? "stale"
          : "unavailable"
        : liveFireMeta.isStale
          ? "stale"
          : liveFireMeta.status === "ok"
            ? "available"
            : "idle";
  const feedMessageKey =
    feedState === "available" && liveFires.length === 0
      ? "analytics.feed.empty_confirmed"
      : `analytics.feed.${feedState}`;

  const CustomTooltip = ({ active, payload, label }: {
    active?: boolean;
    payload?: {
      value: number;
      fill?: string;
      dataKey?: string | number;
    }[];
    label?: string;
  }) => {
    if (!active || !payload?.length) return null;
    return (
      <div style={{
        background: "var(--surface)", border: "1px solid var(--border)",
        borderRadius: 6, padding: "8px 12px", fontSize: 12,
      }}>
        <div style={{ fontWeight: 700, marginBottom: 4 }}>{label}</div>
        {payload.map((p, i) => (
          <div key={i} style={{ color: p.fill ?? "var(--accent)" }}>
            {typeof p.value === "number"
              ? p.dataKey === "burned"
                ? `${p.value.toFixed(2)} M rai`
                : fmtTr(t("analytics.tooltip.points"), { count: p.value })
              : p.value}
          </div>
        ))}
      </div>
    );
  };

  return (
    <div className="notify-page" ref={chartRef}>
      <header className="notify-page-header" style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 16, flexWrap: "wrap" }}>
        <div>
          <h1>
            {th ? "📈 สถิติจุดความร้อน" : "📈 Hotspot Analytics"}
          </h1>
          <p className="notify-page-subtitle">
            {t("analytics.subtitle")}
          </p>
        </div>
        <div style={{ fontSize: 11, color: "var(--text-3)", textAlign: "right", lineHeight: 1.6 }}>
          {analyticsData && <div>as of {analyticsData.as_of}</div>}
          <div style={{ color: "var(--text-3)" }}>
            {th ? "แหล่งข้อมูล: GISTDA · NASA FIRMS" : "Source: GISTDA · NASA FIRMS"}
          </div>
        </div>
      </header>

      {analyticsError && (
        <div className="analytics-error" role="alert">
          <div>
            <strong>{t("analytics.error.title")}</strong>
            <span>{t("analytics.error.hint")}</span>
          </div>
          <button type="button" onClick={() => setRetryKey((key) => key + 1)}>
            {t("analytics.error.retry")}
          </button>
        </div>
      )}

      <div
        className={`analytics-feed-state ${feedState}`}
        role={feedState === "unavailable" ? "alert" : "status"}
      >
        <strong>{t(`analytics.feed.status.${feedState}`)}</strong>
        <span>
          {fmtTr(t(feedMessageKey), { count: liveFires.length })}
        </span>
      </div>

      {/* KPI strip */}
      <div className="alert-counts" style={{ gridTemplateColumns: "repeat(3, 1fr)", marginBottom: 16 }}>
        <div className="alert-count-card" style={{ borderLeftColor: "#06b6d4" }}>
          <div className="alert-count-label" style={{ color: "#06b6d4" }}>
            {t("analytics.kpi.hotspots")}
          </div>
          <div className="alert-count-value">
            {hasUsableLiveState ? liveFires.length : "—"}
          </div>
        </div>
        <div className="alert-count-card" style={{ borderLeftColor: "#f97316" }}>
          <div className="alert-count-label" style={{ color: "#f97316" }}>
            {t("analytics.kpi.feed_status")}
          </div>
          <div className="alert-count-value" style={{ fontSize: 18 }}>
            {t(`analytics.feed.status.${feedState}`)}
          </div>
        </div>
        <div className="alert-count-card" style={{ borderLeftColor: "#22c55e" }}>
          <div className="alert-count-label" style={{ color: "#22c55e" }}>
            {t("analytics.kpi.land_use")}
          </div>
          <div className="alert-count-value">
            {hasUsableLiveState ? landUseData.length : "—"}
          </div>
        </div>
      </div>

      {/* Row 1: Land use donut + Time-based bar */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 16 }}>

        {/* Land use donut */}
        <div className="report-section">
          <h3 className="report-section-title">
            {th ? "จุดความร้อนตามพื้นที่รับผิดชอบ (จุด)" : "Hotspots by Land Responsibility"}
          </h3>
          {liveFires.length === 0 ? (
            <div className="empty-state" style={{ minHeight: 180 }}>
              <div className="empty-icon">🔥</div>
              <div className="empty-hint">
                {hasUsableLiveState
                  ? t("analytics.empty.confirmed")
                  : t("analytics.empty.unavailable")}
              </div>
            </div>
          ) : (
            <div style={{ display: "flex", gap: 16, alignItems: "center", flexWrap: "wrap" }}>
              <div style={{ flex: "0 0 auto", width: 180, height: 180 }}>
                <PieChart width={180} height={180}>
                  <Pie data={landUseData} dataKey="value" cx={85} cy={85}
                    innerRadius={50} outerRadius={78} strokeWidth={1} stroke="var(--surface)">
                    {landUseData.map((d, i) => <Cell key={i} fill={d.fill} />)}
                  </Pie>
                  <text x={89} y={81} textAnchor="middle" fill="var(--text)" fontSize={22} fontWeight={700}>
                    {liveFires.length}
                  </text>
                  <text x={89} y={100} textAnchor="middle" fill="var(--text-3)" fontSize={10}>
                    {th ? "จุด" : "pts"}
                  </text>
                </PieChart>
              </div>
              <div style={{ flex: 1, minWidth: 120 }}>
                {landUseData.map((d) => (
                  <div key={d.nameTh} style={{
                    display: "flex", alignItems: "center", gap: 8,
                    marginBottom: 5, fontSize: 12,
                  }}>
                    <span style={{
                      width: 10, height: 10, borderRadius: 2,
                      background: d.fill, flexShrink: 0,
                    }} />
                    <span style={{ flex: 1, color: "var(--text-2)", fontSize: 11 }}>{d.name}</span>
                    <span style={{ fontWeight: 700, color: "var(--text)", minWidth: 28, textAlign: "right" }}>
                      {d.value}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Time-based VIIRS */}
        <div className="report-section">
          <h3 className="report-section-title">
            {th ? "เวลานับตั้งแต่ตรวจพบ · ชั่วโมง" : "Time Since Detection · Hours"}
          </h3>
          {liveFires.length === 0 ? (
            <div className="empty-state" style={{ minHeight: 180 }}>
              <div className="empty-hint">
                {hasUsableLiveState
                  ? t("analytics.empty.confirmed")
                  : t("analytics.empty.unavailable")}
              </div>
            </div>
          ) : (
            <>
              <ResponsiveContainer width="100%" height={180}>
                <BarChart data={timeData} margin={{ top: 4, right: 8, left: -20, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
                  <XAxis dataKey="label" tick={{ fontSize: 10, fill: "var(--text-3)" }} />
                  <YAxis tick={{ fontSize: 10, fill: "var(--text-3)" }} />
                  <Tooltip content={<CustomTooltip />} />
                  <Bar dataKey="count" radius={[3, 3, 0, 0]}>
                    {timeData.map((d, i) => <Cell key={i} fill={d.fill} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
              {timeSummary.timestampUnavailable > 0 && (
                <p style={{ fontSize: 10, color: "#eab308", marginTop: 4, textAlign: "center" }}>
                  {fmtTr(t("analytics.time.excluded"), {
                    count: timeSummary.timestampUnavailable,
                  })}
                </p>
              )}
            </>
          )}
        </div>
      </div>

      {/* Row 2: Region bar + Province horizontal bar */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 16 }}>

        {/* By region */}
        <div className="report-section">
          <h3 className="report-section-title">
            {th ? "จุดความร้อนจำแนกตามภาค (จุด)" : "Hotspots by Region"}
          </h3>
          {regionData.length === 0 ? (
            <div className="empty-state" style={{ minHeight: 200 }}>
              <div className="empty-hint">
                {hasUsableLiveState
                  ? t("analytics.empty.confirmed")
                  : t("analytics.empty.unavailable")}
              </div>
            </div>
          ) : (
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={regionData} margin={{ top: 4, right: 8, left: -20, bottom: 24 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
                <XAxis dataKey="name" tick={{ fontSize: 10, fill: "var(--text-3)" }}
                  angle={-25} textAnchor="end" interval={0} />
                <YAxis tick={{ fontSize: 10, fill: "var(--text-3)" }} />
                <Tooltip content={<CustomTooltip />} />
                <Bar dataKey="count" radius={[3, 3, 0, 0]}>
                  {regionData.map((d, i) => <Cell key={i} fill={d.fill} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          )}
        </div>

        {/* Top provinces */}
        <div className="report-section">
          <h3 className="report-section-title">
            {th ? "จุดความร้อนจำแนกตามจังหวัด (จุด)" : "Hotspots by Province (top 10)"}
          </h3>
          {provinceData.length === 0 ? (
            <div className="empty-state" style={{ minHeight: 200 }}>
              <div className="empty-hint">
                {hasUsableLiveState
                  ? t("analytics.empty.confirmed")
                  : t("analytics.empty.unavailable")}
              </div>
            </div>
          ) : (
            <ResponsiveContainer width="100%" height={Math.max(200, provinceData.length * 28)}>
              <BarChart data={provinceData} layout="vertical"
                margin={{ top: 4, right: 40, left: 4, bottom: 4 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" horizontal={false} />
                <XAxis type="number" tick={{ fontSize: 10, fill: "var(--text-3)" }} />
                <YAxis type="category" dataKey="name" width={110}
                  tick={{ fontSize: 10, fill: "var(--text-2)" }} />
                <Tooltip content={<CustomTooltip />} />
                <Bar dataKey="count" fill="#f97316" radius={[0, 3, 3, 0]}>
                  {provinceData.map((_, i) => (
                    <Cell key={i} fill={i === 0 ? "#ef4444" : i === 1 ? "#f97316" : "#f59e0b"} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          )}
        </div>
      </div>

      {/* Row 3: Transboundary + Historical */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 16 }}>

        {/* Transboundary */}
        <div className="report-section">
          <h3 className="report-section-title">
            {th ? "จุดความร้อนประเทศเพื่อนบ้าน (จุด)" : "Transboundary Hotspots (FIRMS last 24h)"}
          </h3>
          {loading ? (
            <div className="empty-state" style={{ minHeight: 200 }}>
              <div className="empty-hint">{th ? "กำลังโหลด…" : "Loading…"}</div>
            </div>
          ) : transboundaryData.length === 0 ? (
            <div className="empty-state" style={{ minHeight: 200 }}>
              <div className="empty-hint">
                {transboundaryUnavailable
                  ? t("analytics.transboundary.unavailable")
                  : t("analytics.transboundary.empty")}
              </div>
            </div>
          ) : (
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={transboundaryData} layout="vertical"
                margin={{ top: 4, right: 50, left: 4, bottom: 4 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" horizontal={false} />
                <XAxis type="number" tick={{ fontSize: 10, fill: "var(--text-3)" }} />
                <YAxis type="category" dataKey="country" width={75}
                  tick={{ fontSize: 11, fill: "var(--text-2)" }} />
                <Tooltip content={<CustomTooltip />} />
                <Bar dataKey="count" radius={[0, 3, 3, 0]}>
                  {transboundaryData.map((d, i) => <Cell key={i} fill={d.fill} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          )}
          <p style={{ fontSize: 10, color: "var(--text-3)", marginTop: 4 }}>
            {transboundaryMeta?.available
              ? fmtTr(t("analytics.transboundary.method"), {
                  hours: transboundaryMeta.window_hours ?? 24,
                  assigned: transboundaryMeta.assigned_rows ?? "—",
                  source: transboundaryMeta.source_rows_in_window ?? "—",
                  unassigned: transboundaryMeta.unassigned_rows ?? "—",
                })
              : t("analytics.transboundary.method.unavailable")}
          </p>
        </div>

        {/* Historical burned area */}
        <div className="report-section">
          <h3 className="report-section-title">
            {t("analytics.history.title")}
          </h3>
          {historicalData.length > 0 && (
            <div style={{ fontSize: 13, fontWeight: 700, color: "#f97316", marginBottom: 8 }}>
              {fmtTr(t("analytics.history.range"), {
                range: historicalRange,
                years: historicalData.length,
              })}
            </div>
          )}
          {loading ? (
            <div className="empty-state" style={{ minHeight: 180 }}>
              <div className="empty-hint">{th ? "กำลังโหลด…" : "Loading…"}</div>
            </div>
          ) : analyticsError && historicalData.length === 0 ? (
            <div className="empty-state" style={{ minHeight: 180 }}>
              <div className="empty-hint">
                {t("analytics.history.unavailable")}
              </div>
            </div>
          ) : historicalData.length === 0 ? (
            <div className="empty-state" style={{ minHeight: 180 }}>
              <div className="empty-hint">
                {t("analytics.history.empty")}
              </div>
            </div>
          ) : (
            <ResponsiveContainer width="100%" height={180}>
              <BarChart data={historicalData} margin={{ top: 4, right: 8, left: -16, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
                <XAxis dataKey="label" tick={{ fontSize: 10, fill: "var(--text-3)" }} />
                <YAxis tick={{ fontSize: 10, fill: "var(--text-3)" }} unit="M" />
                <Tooltip content={<CustomTooltip />} />
                <Bar dataKey="burned" fill="#f97316" radius={[3, 3, 0, 0]}
                  name={th ? "ล้านไร่" : "M rai"} />
              </BarChart>
            </ResponsiveContainer>
          )}
          <p style={{ fontSize: 10, color: "var(--text-3)", marginTop: 4 }}>
            {t("analytics.history.source")}
          </p>
        </div>
      </div>

      {/* Attribution footer */}
      <footer className="report-footer">
        <div style={{ display: "flex", gap: 20, flexWrap: "wrap", justifyContent: "center", fontSize: 11, color: "var(--text-3)" }}>
          <a href="https://fire.gistda.or.th/" target="_blank" rel="noopener noreferrer"
            style={{ color: "#06b6d4", textDecoration: "none" }}>
            🛰 GISTDA · fire.gistda.or.th
          </a>
          <a href="https://firms.modaps.eosdis.nasa.gov/" target="_blank" rel="noopener noreferrer"
            style={{ color: "#f97316", textDecoration: "none" }}>
            🛰 NASA FIRMS · firms.modaps.eosdis.nasa.gov
          </a>
          <span>{t("analytics.footer.polling")}</span>
        </div>
      </footer>
    </div>
  );
}
