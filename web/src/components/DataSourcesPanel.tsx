// Transparency / provenance panel — shows where the source observations and
// derived model inputs come from. Derived aggregates and reanalysis fields are
// named as such; the UI never claims every feature is a raw measurement.
import type { Lang } from "../utils/i18n";
import { useLang } from "../utils/i18n";

type LocalizedText = Record<Lang, string>;

interface Source {
  icon: string;
  name: string;
  description: LocalizedText;
  url: string;
  license: string;
  contributes: LocalizedText;
}

const SOURCES: Source[] = [
  {
    icon: "🔥",
    name: "GISTDA NRT VIIRS + MODIS",
    description: {
      en: "Near-real-time hotspots from Thailand's operational monitoring system, shown as a live overlay separate from predictions.",
      th: "จุดความร้อนใกล้เวลาจริงจากระบบติดตามของประเทศไทย ใช้เป็น live overlay แยกจากผลทำนาย",
    },
    url: "https://fire.gistda.or.th/",
    license: "Public operational feed",
    contributes: {
      en: "live hotspot map, counts, and land-use/province context (not a model probability)",
      th: "แผนที่และจำนวน hotspot สด พร้อมบริบทประเภทพื้นที่/จังหวัด (ไม่ใช่ probability ของโมเดล)",
    },
  },
  {
    icon: "🛰",
    name: "NASA FIRMS VIIRS NRT + Archive",
    description: {
      en: "VIIRS hotspot observations from Suomi-NPP, NOAA-20, and NOAA-21, including archive-derived history.",
      th: "ข้อมูล hotspot จาก VIIRS บน Suomi-NPP, NOAA-20 และ NOAA-21 รวมถึงข้อมูลย้อนหลังจาก archive",
    },
    url: "https://firms.modaps.eosdis.nasa.gov/",
    license: "Open / US Government",
    contributes: {
      en: "fire count, FRP, brightness, confidence, and derived spatial-neighbour features",
      th: "จำนวนไฟ, FRP, brightness, confidence และ feature เพื่อนบ้านเชิงพื้นที่ที่ระบบสร้างต่อ",
    },
  },
  {
    icon: "🌤",
    name: "ECMWF ERA5 Reanalysis",
    description: {
      en: "Temperature, humidity, wind, and rain reanalysis fields at 0.25° (Open-Meteo Archive + Copernicus CDS).",
      th: "ข้อมูล reanalysis อุณหภูมิ ความชื้น ลม และฝน ที่ความละเอียด 0.25° (Open-Meteo Archive + Copernicus CDS)",
    },
    url: "https://open-meteo.com/en/docs/historical-weather-api",
    license: "CC-BY 4.0",
    contributes: {
      en: "temp_max/min, precip_sum, wind_max, et0; enabled as model features only when recorded coverage meets the pipeline rule",
      th: "temp_max/min, precip_sum, wind_max, et0; ใช้เป็น feature โมเดลเมื่อ coverage ที่บันทึกไว้ผ่านกฎของ pipeline เท่านั้น",
    },
  },
  {
    icon: "🌳",
    name: "Hansen GFC v1.11",
    description: {
      en: "Tree-cover baseline and derived recent-loss context.",
      th: "บริบทพื้นที่ปกคลุมด้วยต้นไม้จาก baseline และการสูญเสียในช่วงล่าสุดที่ระบบคำนวณ",
    },
    // Version-matched official download notes. The previous link pointed to
    // a separate 2010 tree-cover product even though this application uses
    // Hansen GFC's `treecover2000` and `lossyear` layers.
    url: "https://storage.googleapis.com/earthenginepartners-hansen/GFC-2023-v1.11/download.html",
    license: "CC BY 4.0 · Hansen/UMD/Google/USGS/NASA",
    contributes: {
      en: "tree_cover_pct_2000 and tree_loss_pct_recent",
      th: "tree_cover_pct_2000 และ tree_loss_pct_recent",
    },
  },
  {
    icon: "🇹🇭",
    name: "Thailand boundary GeoJSON",
    description: {
      en: "Thailand administrative boundaries used to annotate and scope map rows.",
      th: "ขอบเขตการปกครองของไทย ใช้ระบุจังหวัดและจำกัดขอบเขตแถวบนแผนที่",
    },
    url: "https://data.humdata.org/dataset/cod-ab-tha",
    license: "Open",
    contributes: {
      en: "Thailand-scope mask and province annotation",
      th: "mask ขอบเขตประเทศไทยและการระบุจังหวัด",
    },
  },
];

export default function DataSourcesPanel() {
  const { lang } = useLang();
  return (
    <div className="section">
      <h3>📡 {lang === "th" ? "แหล่งข้อมูล" : "Data sources"}</h3>
      <p style={{ fontSize: 11, color: "var(--text-3)", marginBottom: 10 }}>
        {lang === "th"
          ? "ข้อมูลต้นทางมาจากแหล่งสาธารณะที่ตรวจสอบได้ แล้วระบบสร้าง temporal/spatial aggregates ตาม Model Card; เมื่อ feed สดล้มเหลว ระบบจะไม่แทนจำนวน hotspot ด้วยข้อมูลสมมติ"
          : "Inputs come from traceable public sources, then the pipeline derives temporal/spatial aggregates documented in the Model Card. A failed live feed is never replaced with fabricated hotspot counts."}
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
            title={`${lang === "th" ? "เปิดแหล่งข้อมูล" : "Open source"}: ${s.url}`}
          >
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <span style={{ fontSize: 16 }}>{s.icon}</span>
              <b style={{ fontSize: 13, color: "var(--text)" }}>{s.name}</b>
              <span style={{ fontSize: 10, color: "var(--text-3)" }}>↗</span>
            </div>
            <div style={{ fontSize: 11, color: "var(--text-2)", marginTop: 4, lineHeight: 1.4 }}>
              {s.description[lang]}
            </div>
            <div style={{ fontSize: 10, color: "var(--text-3)", marginTop: 4 }}>
              📋 {s.contributes[lang]}
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
