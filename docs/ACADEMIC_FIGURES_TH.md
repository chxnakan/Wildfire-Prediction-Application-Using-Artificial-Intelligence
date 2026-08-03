# ชุดกราฟสำหรับงานวิชาการ

สร้างใหม่ได้จาก project root:

```bash
pip install -r requirements-figures.txt
.venv/bin/python scripts/academic_figures.py
```

ไฟล์ทั้งหมดอยู่ที่ `outputs/figures/academic/` และส่งออกเป็น PNG 300 dpi,
SVG และ PDF โดยอัตโนมัติ

## ภาพที่แนะนำให้ใช้เป็นหลัก

| ลำดับ | ไฟล์ | ใช้ตอบคำถาม | คำอธิบายภาพแนะนำ |
|---:|---|---|---|
| 1 | `00_academic_evaluation_summary` | โมเดลดีเพียงใดโดยรวม | สรุปผลการจำแนก ความน่าเชื่อถือของความน่าจะเป็น ตัวแปรสำคัญ ความชุกของเหตุการณ์ และผลที่จุดปฏิบัติการบนชุดทดสอบตามเวลา |
| 2 | `01_roc_precision_recall` | โมเดลแยกพื้นที่เสี่ยงได้ดีเพียงใด | เส้น ROC และ Precision–Recall ของโมเดลบนชุดทดสอบที่ไม่ใช้ในการฝึกหรือเลือก threshold โดย PR curve เหมาะกับข้อมูลไฟป่าที่มี class imbalance มากกว่า accuracy เพียงค่าเดียว |
| 3 | `02_probability_calibration` | ค่าความน่าจะเป็นตรงกับอัตราเกิดจริงหรือไม่ | Reliability diagram และการกระจายคะแนนความเสี่ยงบนชุดทดสอบ ฟองขนาดใหญ่หมายถึง bin ที่มีจำนวนตัวอย่างมาก |
| 4 | `03_operational_confusion_matrices` | WATCH และ HIGH แลก precision/recall อย่างไร | Confusion matrix แบบ normalize ตามคลาสจริงที่ threshold WATCH และ HIGH พร้อมจำนวนตัวอย่างจริงในวงเล็บ |
| 5 | `04_feature_importance` | โมเดลใช้ข้อมูลใดมาก | ความสำคัญสัมพัทธ์ของตัวแปร 20 อันดับแรกจาก LightGBM ensemble ไม่ควรตีความเป็นเหตุ–ผล |
| 6 | `05_baseline_comparison` | AI ดีกว่าวิธีพื้นฐานหรือไม่ | เปรียบเทียบ ROC-AUC และ Average Precision กับ persistence และ climatology บนช่วงทดสอบเดียวกัน |
| 7 | `07_topk_prioritization` | หากตรวจเฉพาะพื้นที่เสี่ยงสูงจะครอบคลุมไฟได้เท่าใด | Precision และสัดส่วนเหตุการณ์ที่จับได้เมื่อเลือกตรวจ cell-day ที่มีคะแนนสูงสุด K% |
| 8 | `09_rolling_origin_backtest` | ผลคงที่ข้ามปีหรือไม่ | ผล rolling-origin backtest ในฤดูไฟของแต่ละปี โดยฝึกจากอดีตและทดสอบกับปีถัดไปเท่านั้น |

## ภาพสนับสนุนวิธีวิจัย

| ไฟล์ | จุดประสงค์ |
|---|---|
| `06_threshold_tradeoff` | แสดงผลของการเปลี่ยน threshold ต่อ precision, recall, F1 และ alert rate |
| `08_split_protocol_prevalence` | อธิบาย chronological split, purge gap และ prevalence drift ระหว่างช่วงเวลา |
| `10_daily_fire_activity` | แสดงจำนวน hotspot และจำนวน grid cell ที่พบไฟจริงรายวัน |
| `11_top_feature_correlation` | ตรวจความสัมพันธ์ซ้ำซ้อนระหว่างตัวแปรสำคัญ |
| `12_weather_feature_ablation` | ผลทดลองเพิ่ม missingness/drought features บน evaluation block แยกต่างหาก ไม่ใช่ผล frozen test ของโมเดล deployed |

## ข้อควรระวังในการเขียนรายงาน

1. งานนี้ทำนาย `P(พบ hotspot อย่างน้อยหนึ่งจุดใน cell ภายใน 3 วัน)` ไม่ได้ทำนายวันเกิดไฟแบบระบุวันที่
2. ใช้ ROC-AUC ร่วมกับ Average Precision เพราะชุดทดสอบมีเหตุการณ์บวกประมาณ 3.24%; accuracy เพียงค่าเดียวอาจดูสูงจากคลาสลบจำนวนมาก
3. WATCH และ HIGH ถูกเลือกจาก calibration split เท่านั้น กราฟบน test split เป็นการประเมิน ไม่ใช่การปรับ threshold ย้อนหลัง
4. Feature importance และ correlation ไม่ยืนยันความสัมพันธ์เชิงสาเหตุ
5. กราฟ ablation และ rolling backtest เป็นการทดลองคนละ protocol กับ frozen test ต้องระบุให้ชัดเมื่อใช้ในเอกสารเดียวกัน
6. ไฟล์ feature snapshot อาจถูกสร้างใหม่หลังการฝึกโมเดล กราฟ ROC/PR, calibration,
   confusion matrix, threshold และ Top-K ในชุดนี้จึงคำนวณใหม่จาก snapshot ปัจจุบันชุดเดียวกัน
   ส่วนกราฟ baseline ใช้ค่าที่บันทึกไว้ ณ เวลาฝึกและระบุไว้ในคำบรรยายภาพ
