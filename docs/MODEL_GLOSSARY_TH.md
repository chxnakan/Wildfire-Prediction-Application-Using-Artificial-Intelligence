# คลังศัพท์ FireWatch Thailand

เอกสารนี้อธิบายคำที่ใช้ในระบบปัจจุบัน ตัวเลขจริงต้องอ่านจาก
`outputs/metadata/dataset_info.json` เสมอ

## โจทย์ของโมเดล

| คำ | ความหมายในโครงงาน |
|---|---|
| **Binary classification** | ตอบว่าใน cell จะมี FIRMS hotspot ในช่วง 1–3 วันข้างหน้าหรือไม่ |
| **Probability** | ค่าระหว่าง 0–1 ของเหตุการณ์ข้างต้น เช่น 0.12 = 12% |
| **Label / target** | `1` เมื่อ cell มี hotspot อย่างน้อยหนึ่งจุดใน t+1…t+3, ไม่เช่นนั้น `0` |
| **Prediction horizon** | ช่วงเวลาที่โมเดลรองรับ ปัจจุบันคือ 3 วัน |
| **Cell-day** | หน่วยข้อมูลหนึ่งแถว: cell หนึ่งตำแหน่ง ณ base date หนึ่งวัน |
| **0.1° grid** | cell ประมาณ 11 กม. ในแนวเหนือ–ใต้ ไม่ใช่พิกัดไฟละเอียดระดับอาคาร |
| **Historically active universe** | cell ที่เคยมี FIRMS อย่างน้อยหนึ่งครั้ง และเริ่มเข้าชุดข้อมูลตั้งแต่วันตรวจพบครั้งแรก |

`predicted_fire_date`, `days_until_fire`, MAE/RMSE/R² แบบจำนวนวัน เป็น field
legacy เพื่อ compatibility เท่านั้น ไม่ใช่ผลลัพธ์ของโมเดลที่ใช้งานจริง

## โมเดลและการ train

| คำ | ความหมาย |
|---|---|
| **LightGBM** | Gradient-boosted decision trees สำหรับข้อมูลตาราง |
| **Ensemble** | เฉลี่ย probability จากโมเดลหลาย seed เพื่อลดความแปรปรวน |
| **Hyperparameter** | ค่าที่กำหนดก่อน train เช่น learning rate, depth, leaves |
| **Bayesian search** | เลือกชุด hyperparameter ถัดไปจากผลรอบก่อนอย่างมีระบบ |
| **Early stopping** | หยุดเพิ่มต้นไม้เมื่อคะแนน validation ไม่ดีขึ้น |
| **Class imbalance** | positive พบได้น้อยกว่า negative มาก ซึ่งเป็นธรรมชาติของ rare-event prediction |
| **Negative undersampling** | ลด negative เฉพาะ train เพื่อควบคุม memory/สมดุลการเรียนรู้ ห้ามทำกับ calibration/test |
| **Platt calibration** | ใช้ logistic regression แปลง raw score ให้ใกล้ probability จริง |
| **Promotion gate** | เงื่อนไขก่อน candidate จะเปลี่ยน model ที่กำลัง serve; ถ้าไม่ผ่านให้เก็บ incumbent |

## การแบ่งข้อมูลที่กัน leakage

| คำ | ความหมาย |
|---|---|
| **Train** | ใช้ fit พารามิเตอร์โมเดล |
| **Validation** | ใช้ early stopping และเลือก model |
| **Calibration** | ใช้ fit calibrator และเลือก threshold ก่อนแตะ test |
| **Held-out test** | ช่วงเวลาสุดท้ายที่ประเมินครั้งเดียว ไม่ใช้เลือก model/threshold |
| **Whole-date split** | วันหนึ่งต้องอยู่ partition เดียว ห้ามแถวของวันเดียวกันข้ามชุด |
| **Purge gap** | ตัดช่วง 7 วันระหว่าง partition เพื่อไม่ให้ forward label window ซ้อนกัน |
| **Censored tail** | แถวท้ายที่ยังมองอนาคตครบ 3 วันไม่ได้ จึงไม่ใช้เป็น label “ไม่มีไฟ” |
| **Time leakage** | feature หรือขั้นตอนเลือก model แอบใช้ข้อมูลจากอนาคต ทำให้ test ดูดีเกินจริง |
| **Causal feature** | feature ที่ใช้ข้อมูลซึ่งทราบได้ ณ เวลา prediction เท่านั้น |

## Metrics

| Metric | ความหมาย | อ่านอย่างไร |
|---|---|---|
| **ROC-AUC** | โอกาสที่โมเดลจะ rank positive สูงกว่า negative | 0.5 = random, 1 = perfect; ไม่ใช่ accuracy |
| **Average Precision / PR-AUC** | คุณภาพ precision–recall ตลอด threshold | สำคัญเมื่อ positive มีน้อย ต้องอ่านคู่ positive rate |
| **Precision** | จาก alert ทั้งหมด เป็น positive จริงกี่ส่วน | สูง = false alarm น้อย |
| **Recall / Sensitivity** | จาก positive จริงทั้งหมด จับได้กี่ส่วน | สูง = miss น้อย |
| **F1** | harmonic mean ของ precision และ recall | ใช้เมื่ออยากสมดุลสองค่า |
| **Brier score** | ค่าเฉลี่ยความคลาดเคลื่อนกำลังสองของ probability | ต่ำกว่า = ดีกว่า |
| **ECE** | ช่องว่างเฉลี่ยระหว่าง probability ที่ทายกับอัตราเกิดจริงใน bin | ต่ำกว่า = calibrated ดีกว่า |
| **Alert rate** | สัดส่วน cell-days ที่ข้าม threshold | บอกภาระงานของ operator |
| **Confusion matrix** | TP, FP, FN, TN ณ threshold หนึ่ง | ต้องดูจำนวนจริง ไม่ใช่เปอร์เซ็นต์อย่างเดียว |
| **Baseline** | วิธีง่าย ๆ สำหรับเทียบ เช่น “เคยมีไฟใน 7 วันล่าสุด” | โมเดลควรแสดง uplift เหนือ baseline บน test เดียวกัน |

### TP / FP / FN / TN

| คำ | ความหมาย |
|---|---|
| **TP (true positive)** | เตือนและมี hotspot ใน target window จริง |
| **FP (false positive)** | เตือนแต่ไม่พบ hotspot ใน target window |
| **FN (false negative)** | ไม่เตือนแต่มี hotspot |
| **TN (true negative)** | ไม่เตือนและไม่พบ hotspot |

## Absolute tier กับ relative ranking

| แนวคิด | ความหมาย |
|---|---|
| **HIGH** | probability ข้าม threshold แบบเน้น precision |
| **WATCH** | threshold ต่ำกว่า HIGH เพื่อเฝ้าระวังและเพิ่ม recall |
| **Background** | ต่ำกว่า alert threshold; ไม่ได้แปลว่าปลอดภัยแน่นอน |
| **Top-K / Top 20%** | เลือกตามอันดับสัมพัทธ์เพื่อแบ่งทรัพยากร แม้ probability absolute จะต่ำ |
| **Historical coverage** | สัดส่วน positive held-out ที่อยู่ในกลุ่มอันดับนั้น ไม่ใช่การรับประกันวันใดวันหนึ่ง |

WATCH recall และ Top-20% coverage เป็นคนละ metric ห้ามเรียกแทนกัน

## Drift และความสดของข้อมูล

| คำ | ความหมาย |
|---|---|
| **Base-rate / prevalence** | สัดส่วน positive ในประชากรที่กำลังวัด |
| **Seasonal drift** | prevalence หรือ distribution ต่างกันตามฤดู ทำให้ threshold เดิมถ่ายข้ามช่วงได้ไม่ดี |
| **Data freshness** | เวลาของ observation ล่าสุดเทียบกับปัจจุบัน |
| **Stale** | ยังมี cache แต่เก่าเกินเกณฑ์ ต้องแสดงคำเตือน |
| **Unavailable** | request/source ล้มเหลว ต้องไม่แสดงเป็นศูนย์ที่ดูเหมือนยืนยันแล้ว |
| **Confirmed zero** | ดึง source สำเร็จและผลลัพธ์ว่างจริง |

## แหล่งข้อมูล

| Source | บทบาท |
|---|---|
| **NASA FIRMS VIIRS** | history และ label หลักของโมเดล |
| **GISTDA VIIRS** | live observation layer แยกจาก prediction |
| **ERA5 / Open-Meteo** | weather context และ feature เมื่อมีข้อมูล |
| **Hansen Global Forest Change** | tree cover และ tree loss |

FIRMS กับ GISTDA เป็นผลิตภัณฑ์ satellite hotspot คนละ source แต่ไม่ควรเรียกว่า
“หลักฐานอิสระสมบูรณ์” เพราะมีข้อจำกัดด้าน overpass/cloud คล้ายกัน

## คำที่ควรใช้ในการนำเสนอ

- “ความเสี่ยงพบ hotspot ภายใน 3 วัน”
- “จัดลำดับพื้นที่ให้ operator ตรวจสอบ”
- “held-out chronological test”
- “absolute threshold” และ “relative ranking”
- “ข้อมูลไม่พร้อมใช้งาน” เมื่อ source ล้มเหลว

คำที่ไม่ควรใช้:

- “ทำนายวันเกิดไฟ”
- “จุดนี้จะเกิดไฟแน่นอน”
- “AUC คือ accuracy”
- “ไม่มีไฟ” เมื่อ live feed ล่ม

ดูตัวเลขและข้อจำกัดล่าสุดที่
[`MODEL_CARD.md`](MODEL_CARD.md) และคู่มือนำเสนอที่
[`PRESENTATION_TH.md`](PRESENTATION_TH.md)
