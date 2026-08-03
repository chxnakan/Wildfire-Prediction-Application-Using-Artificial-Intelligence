// Transparency / provenance panel — shows where the source observations and
// derived model inputs come from. Derived aggregates and reanalysis fields are
// named as such; the UI never claims every feature is a raw measurement.

interface Source {
  icon: string;
  name: string;
  description: string;
  url: string;
  license: string;
  contributes: string;
}

const SOURCES: Source[] = [
  {
    icon: "🔥",
    name: "GISTDA NRT VIIRS + MODIS",
    description: "จุดความร้อนใกล้เวลาจริงจากระบบติดตามของประเทศไทย ใช้เป็น live overlay แยกจากผลทำนาย",
    url: "https://fire.gistda.or.th/",
    license: "Public operational feed",
    contributes: "live hotspot map, live counts, land-use/province context (not a model probability)",
  },
  {
    icon: "🛰",
    name: "NASA FIRMS VIIRS NRT + Archive",
    description: "Hotspots จาก 3 ดาวเทียม (Suomi-NPP + NOAA-20 + NOAA-21) · archive ย้อนหลังถึงปี 2012",
    url: "https://firms.modaps.eosdis.nasa.gov/",
    license: "Open / US Government",
    contributes: "fire_count, FRP, brightness, confidence + spatial-neighbor features",
  },
  {
    icon: "🌤",
    name: "ECMWF ERA5 Reanalysis",
    description: "อุณหภูมิ ความชื้น ลม ฝน ระดับ 0.25° (Open-Meteo Archive + Copernicus CDS)",
    url: "https://open-meteo.com/en/docs/historical-weather-api",
    license: "CC-BY 4.0",
    contributes: "temp_max/min, precip_sum, wind_max, et0 — เปิดใช้เป็น feature เมื่อ coverage ≥ 20% (กำลัง backfill)",
  },
  {
    icon: "🌳",
    name: "Hansen GFC v1.11",
    description: "Tree cover baseline 2000 + recent loss",
    url: "https://glad.umd.edu/dataset/global-2010-tree-cover-30-m",
    license: "Open (University of Maryland)",
    contributes: "tree_cover_pct_2000, tree_loss_pct_recent",
  },
  {
    icon: "🇹🇭",
    name: "Thailand boundary GeoJSON",
    description: "ขอบเขต 77 จังหวัด จาก HDX",
    url: "https://data.humdata.org/dataset/cod-ab-tha",
    license: "Open",
    contributes: "is_in_thailand mask, province annotation",
  },
];

export default function DataSourcesPanel() {
  return (
    <div className="section">
      <h3>📡 Data Sources</h3>
      <p style={{ fontSize: 11, color: "var(--text-3)", marginBottom: 10 }}>
        ข้อมูลต้นทางมาจาก <b>แหล่งสาธารณะที่ตรวจสอบได้</b> แล้วระบบสร้าง
        temporal/spatial aggregates และ cell-day rows ตามวิธีใน Model Card;
        จำนวน live hotspot จะไม่ถูกแทนด้วยข้อมูลสมมติเมื่อ feed ล้มเหลว
      </p>

      <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
        {SOURCES.map((s) => (
          <a
            key={s.name}
            href={s.url}
            target="_blank"
            rel="noopener noreferrer"
            style={{
              display: "block",
              padding: "10px 12px",
              background: "var(--surface-2)",
              borderRadius: 6,
              border: "1px solid var(--border)",
              textDecoration: "none",
              color: "inherit",
              transition: "border-color 0.12s",
            }}
            onMouseEnter={(e) => (e.currentTarget.style.borderColor = "var(--accent)")}
            onMouseLeave={(e) => (e.currentTarget.style.borderColor = "var(--border)")}
            title={`เปิด ${s.url}`}
          >
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <span style={{ fontSize: 16 }}>{s.icon}</span>
              <b style={{ fontSize: 13, color: "var(--text)" }}>{s.name}</b>
              <span style={{ fontSize: 10, color: "var(--text-3)" }}>↗</span>
            </div>
            <div style={{ fontSize: 11, color: "var(--text-2)", marginTop: 4, lineHeight: 1.4 }}>
              {s.description}
            </div>
            <div style={{ fontSize: 10, color: "var(--text-3)", marginTop: 4 }}>
              📋 {s.contributes}
            </div>
            <div style={{ fontSize: 10, color: "var(--text-3)", marginTop: 2 }}>
              ⚖️ {s.license}
            </div>
          </a>
        ))}
      </div>
    </div>
  );
}
