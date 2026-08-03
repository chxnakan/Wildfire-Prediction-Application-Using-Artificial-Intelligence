#!/usr/bin/env python3
"""Export the promoted wildfire model's feature contract as a categorized CSV."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
METADATA = ROOT / "outputs/metadata/dataset_info.json"
FEATURE_TABLE = ROOT / "outputs/features/full_features.parquet"
OUTPUT = ROOT / "outputs/metadata/model_features_categorized.csv"

CATEGORIES = {
    "A": ("ประวัติไฟและพลังงานไฟของเซลล์", "Cell fire and FRP history"),
    "B": ("กิจกรรมไฟเชิงพื้นที่", "Spatial fire activity"),
    "C": ("ปฏิทิน ประวัติซ้ำ และภูมิศาสตร์", "Calendar, recurrence, and geography"),
    "D": ("สภาพอากาศ", "Weather"),
}


def window_days(name: str) -> int | None:
    match = re.search(r"_(\d+)d(?:$|_)", name)
    return int(match.group(1)) if match else None


def lag_days(name: str) -> int | None:
    match = re.search(r"_lag_(\d+)$", name)
    return int(match.group(1)) if match else None


def metric_label(name: str) -> str:
    if "fire" in name:
        return "จำนวนจุดความร้อน"
    if "frp" in name:
        return "พลังงานการแผ่รังสีของไฟ (FRP)"
    return name


def classify(feature: str) -> tuple[str, str, str, str, str]:
    """Return category code, subcategory TH, source, temporal scope, description TH."""
    lag = lag_days(feature)
    window = window_days(feature)

    # D. Weather
    weather_names = {
        "temp_max": "อุณหภูมิสูงสุด",
        "temp_min": "อุณหภูมิต่ำสุด",
        "precip_sum": "ปริมาณฝนรวม",
        "wind_max": "ความเร็วลมสูงสุด",
        "et0": "การคายระเหยอ้างอิง (ET₀)",
    }
    for prefix, label in weather_names.items():
        if feature.startswith(prefix + "_"):
            source = "ERA5 ผ่าน Open-Meteo Archive API"
            if feature.endswith("_today"):
                return "D", "สภาพอากาศวันฐาน", source, "วันฐาน", f"{label}ของวันฐาน"
            if lag is not None:
                return (
                    "D", "สภาพอากาศย้อนหลังแบบ lag", source, f"ย้อนหลัง {lag} วัน",
                    f"{label}ย้อนหลัง {lag} วัน",
                )
            if "_mean_" in feature and window is not None:
                return (
                    "D", "ค่าเฉลี่ยสภาพอากาศย้อนหลัง", source,
                    f"ค่าเฉลี่ยอดีต {window} วัน ไม่รวมวันฐาน",
                    f"ค่าเฉลี่ย{label}ในอดีต {window} วัน",
                )

    # B. Spatial fire activity
    if feature.startswith(("neighbor_", "wide_neighbor_")):
        wide = feature.startswith("wide_neighbor_")
        area = "วงนอกกริด 5×5 จำนวน 16 เซลล์" if wide else "วงในกริด 3×3 จำนวน 8 เซลล์"
        prefix = "wide_neighbor_" if wide else "neighbor_"
        remainder = feature.removeprefix(prefix)
        source = "NASA FIRMS VIIRS + ตำแหน่งกริด 0.1°"
        if remainder in {"fire_today", "frp_today"}:
            metric = metric_label(remainder)
            return "B", f"กิจกรรมเพื่อนบ้าน{'วงนอก' if wide else 'วงใน'}วันฐาน", source, "วันฐาน", f"{metric}รวมใน{area}ของวันฐาน"
        if "_lag_" in remainder and lag is not None:
            metric = metric_label(remainder)
            return (
                "B", f"กิจกรรมเพื่อนบ้าน{'วงนอก' if wide else 'วงใน'}แบบ lag",
                source, f"ย้อนหลัง {lag} วัน", f"{metric}ใน{area}ย้อนหลัง {lag} วัน",
            )
        if "_sum_" in remainder and window is not None:
            metric = metric_label(remainder)
            return (
                "B", f"กิจกรรมเพื่อนบ้าน{'วงนอก' if wide else 'วงใน'}แบบ rolling",
                source, f"ผลรวมอดีต {window} วัน ไม่รวมวันฐาน",
                f"ผลรวม{metric}ใน{area}ตลอดอดีต {window} วัน",
            )
        if "_velocity_" in remainder and window is not None:
            metric = metric_label(remainder)
            return (
                "B", f"แนวโน้มเพื่อนบ้าน{'วงนอก' if wide else 'วงใน'}",
                source, "เปรียบเทียบ lag 1 กับ lag 3 วัน",
                f"การเปลี่ยนแปลง{metric}ใน{area}ระหว่างย้อนหลัง 1 และ 3 วัน",
            )

    # C. Calendar, recurrence, and geography
    if feature in {"month_sin", "month_cos", "doy_sin", "doy_cos", "week_sin", "week_cos"}:
        unit = "เดือน" if feature.startswith("month_") else ("วันในปี" if feature.startswith("doy_") else "สัปดาห์ในปี")
        trig = "ไซน์" if feature.endswith("_sin") else "โคไซน์"
        return "C", "ปฏิทินแบบวงกลม", "วันที่", "วันฐาน", f"การเข้ารหัส{unit}แบบ{trig}เพื่อรักษาความต่อเนื่องของรอบปี"
    calendar_flags = {
        "is_burn_season": "ระบุว่าวันฐานอยู่ในฤดูเผาหรือไม่",
        "is_dry_season": "ระบุว่าวันฐานอยู่ในฤดูแล้งหรือไม่",
        "days_from_burn_peak": "จำนวนวันห่างจากช่วงจุดสูงสุดของฤดูเผา",
        "day_of_week": "ลำดับวันในสัปดาห์ของวันฐาน",
        "is_weekend": "ระบุว่าวันฐานเป็นวันหยุดสุดสัปดาห์หรือไม่",
    }
    if feature in calendar_flags:
        return "C", "ฤดูกาลและปฏิทิน", "วันที่", "วันฐาน", calendar_flags[feature]
    recurrence = {
        "fire_days_per_year_so_far": ("สะสมตั้งแต่ต้นปีถึงก่อนวันฐาน", "จำนวนวันที่พบไฟในเซลล์ตั้งแต่ต้นปีถึงวันฐาน"),
        "days_since_last_fire": ("อดีตถึงวันฐาน", "จำนวนวันนับจากการพบไฟครั้งล่าสุดในเซลล์"),
        "season_fire_count_so_far": ("สะสมในฤดูเผาปัจจุบัน", "จำนวนจุดความร้อนสะสมในฤดูเผาปัจจุบันถึงวันฐาน"),
        "days_into_burn_season": ("ตำแหน่งในฤดูเผา", "จำนวนวันที่ผ่านไปนับจากจุดเริ่มต้นฤดูเผา"),
        "median_fire_gap": ("ประวัติทั้งหมดก่อนวันฐาน", "ค่ามัธยฐานของช่วงห่างระหว่างวันที่พบไฟในเซลล์"),
        "days_since_last_fire_norm": ("อดีตถึงวันฐาน", "จำนวนวันจากไฟครั้งล่าสุดหารด้วยช่วงห่างไฟมัธยฐานและจำกัดค่าสูงสุด"),
        "same_week_fire_last_year": ("ช่วงวันเดียวกันของปีก่อน", "ระบุการพบไฟในช่วงวันของปี ±7 วันของปีก่อน"),
        "fire_count_same_month_hist": ("เดือนเดียวกันของปีก่อนหน้า", "ค่าเฉลี่ยจำนวนไฟของเดือนเดียวกันจากปีก่อนหน้าเท่านั้น"),
    }
    if feature in recurrence:
        scope, description = recurrence[feature]
        return "C", "ประวัติไฟสะสมและการเกิดซ้ำ", "NASA FIRMS VIIRS + วันที่", scope, description
    if feature == "distance_to_nearest_city_km":
        return "C", "ภูมิศาสตร์", "พิกัดกริด + รายชื่อศูนย์กลางเมือง", "ค่าคงที่ต่อเซลล์", "ระยะทางจากศูนย์กลางเซลล์ถึงเมืองที่ใกล้ที่สุด หน่วยกิโลเมตร"
    if feature in {"tree_cover_pct_2000", "tree_loss_pct_recent"}:
        description = "สัดส่วนพื้นที่ปกคลุมด้วยต้นไม้ในปี 2000" if feature == "tree_cover_pct_2000" else "สัดส่วนการสูญเสียพื้นที่ป่าในช่วงล่าสุด"
        return "C", "พืชพรรณ", "Hansen Global Forest Change", "ค่าคงที่ต่อเซลล์", description
    if feature in {"lat_grid", "lon_grid"}:
        axis = "ละติจูด" if feature == "lat_grid" else "ลองจิจูด"
        return "C", "ตำแหน่งกริด", "พิกัดกริด 0.1°", "ค่าคงที่ต่อเซลล์", f"{axis}ของจุดศูนย์กลางเซลล์"

    # A. Cell fire and FRP history
    source = "NASA FIRMS VIIRS"
    if feature.startswith(("fire_lag_", "frp_lag_")) and lag is not None:
        return "A", "ค่าล่าช้า (lag)", source, f"ย้อนหลัง {lag} วัน", f"{metric_label(feature)}ของเซลล์เดิมย้อนหลัง {lag} วัน"
    if feature.startswith(("fire_sum_", "frp_sum_", "frp_max_", "active_days_")) and window is not None:
        if feature.startswith("active_days_"):
            description = f"จำนวนวันที่พบไฟในอดีต {window} วัน"
        elif feature.startswith("frp_max_"):
            description = f"ค่า FRP สูงสุดในอดีต {window} วัน"
        else:
            description = f"ผลรวม{metric_label(feature)}ในอดีต {window} วัน"
        return "A", "สถิติย้อนหลังแบบ rolling", source, f"อดีต {window} วัน ไม่รวมวันฐาน", description
    if feature.startswith("frp_trend_") and window is not None:
        return "A", "แนวโน้มและความผันผวน", source, f"เปรียบเทียบย้อนหลัง {window} วัน", f"แนวโน้ม FRP จากผลต่างของค่า lag ในช่วง {window} วัน"
    if feature == "fire_std_7d":
        return "A", "แนวโน้มและความผันผวน", source, "อดีต 7 วัน ไม่รวมวันฐาน", "ส่วนเบี่ยงเบนมาตรฐานของจำนวนไฟในอดีต 7 วัน"
    today_descriptions = {
        "fire_count_today": "จำนวนจุดความร้อนที่ตรวจพบในวันฐาน",
        "frp_sum_today": "ผลรวม FRP ของจุดความร้อนในวันฐาน",
        "bright_mean_today": "ค่าความสว่างเฉลี่ยของจุดความร้อนในวันฐาน",
        "confidence_mean_today": "ค่าความเชื่อมั่นเฉลี่ยของการตรวจจับในวันฐาน",
        "night_fire_count": "จำนวนจุดความร้อนที่ตรวจพบในเวลากลางคืนของวันฐาน",
        "afternoon_fire_count": "จำนวนจุดความร้อนที่ตรวจพบในช่วงบ่ายของวันฐาน",
        "n_satellites_today": "จำนวนผลิตภัณฑ์ดาวเทียมที่ให้ข้อมูลในวันฐาน",
    }
    if feature in today_descriptions:
        return "A", "สัญญาณไฟวันฐาน", source, "วันฐาน", today_descriptions[feature]
    derived = {
        "fire_acceleration": ("อดีต 7 และ 14 วัน", "ความเร่งของกิจกรรมไฟจากอัตราเฉลี่ย 7 วันเทียบกับ 14 วัน"),
        "frp_intensity_7d": ("อดีต 7 วัน", "FRP รวมต่อจำนวนไฟในอดีต 7 วัน"),
        "night_fire_ratio": ("วันฐาน", "สัดส่วนไฟกลางคืนต่อจำนวนไฟทั้งหมดในวันฐาน"),
        "fire_streak_today": ("ต่อเนื่องถึงวันฐาน", "จำนวนวันที่ตรวจพบไฟต่อเนื่องถึงวันฐาน"),
        "fire_momentum_7_30": ("อดีต 7 และ 30 วัน", "อัตรากิจกรรมไฟเฉลี่ย 7 วันเทียบกับค่าเฉลี่ย 30 วัน"),
        "frp_per_fire_today": ("วันฐาน", "FRP รวมต่อจำนวนไฟในวันฐาน"),
        "fire_count_60d": ("อดีต 60 วัน ไม่รวมวันฐาน", "ชื่ออ้างอิงอีกชื่อของจำนวนไฟรวมในอดีต 60 วัน"),
        "fire_frequency_rate": ("อดีต 60 วัน ไม่รวมวันฐาน", "สัดส่วนวันที่พบไฟในช่วง 60 วัน"),
    }
    if feature in derived:
        scope, description = derived[feature]
        return "A", "ตัวแปรอนุพันธ์พฤติกรรมไฟ", source, scope, description

    raise ValueError(f"Unclassified feature: {feature}")


def main() -> None:
    with METADATA.open(encoding="utf-8") as handle:
        metadata = json.load(handle)

    features = metadata["features"]
    top_importance = {
        row["feature"]: row["importance"]
        for row in metadata.get("feature_importance_top", [])
    }
    schema_types = {
        field.name: str(field.type)
        for field in pq.read_schema(FEATURE_TABLE)
    }

    rows = []
    for order, feature in enumerate(features, start=1):
        code, subcategory_th, source, temporal_scope, description_th = classify(feature)
        category_th, category_en = CATEGORIES[code]
        rows.append({
            "feature_order": order,
            "feature_name": feature,
            "category_code": code,
            "category_th": category_th,
            "category_en": category_en,
            "subcategory_th": subcategory_th,
            "source": source,
            "temporal_scope": temporal_scope,
            "description_th": description_th,
            "data_type": schema_types.get(feature, ""),
            "top20_importance_pct": top_importance.get(feature, ""),
            "used_in_promoted_model": True,
        })

    catalog = pd.DataFrame(rows)
    if len(catalog) != metadata["feature_count"]:
        raise ValueError(
            f"Expected {metadata['feature_count']} features, exported {len(catalog)}"
        )
    if not catalog["feature_name"].is_unique:
        raise ValueError("Feature names are not unique")
    if catalog[["category_code", "description_th"]].isna().any().any():
        raise ValueError("Catalog has missing classifications")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    print(OUTPUT)
    print(catalog.groupby(["category_code", "category_th"]).size().to_string())


if __name__ == "__main__":
    main()
