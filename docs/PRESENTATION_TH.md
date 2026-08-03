# FireWatch Thailand — คู่มือนำเสนอฉบับข้อมูลปัจจุบัน

> ใช้กับ artifact ที่ train วันที่ **16 ก.ค. 2026** เท่านั้น
> แหล่งตัวเลขหลัก:
> `outputs/metadata/dataset_info.json`
> อัปเดตเอกสาร: **24 ก.ค. 2026**

## ประโยคหลักที่ต้องพูดให้ตรง

FireWatch Thailand ไม่ได้ทำนาย “วันและจุดที่จะเกิดไฟแน่นอน” แต่ตอบว่า:

> ใน cell ขนาด 0.1° นี้ มีโอกาสพบ NASA FIRMS hotspot อย่างน้อย 1 จุด
> ในช่วง 1–3 วันข้างหน้ากี่เปอร์เซ็นต์

ผลลัพธ์ใช้เพื่อ **จัดลำดับพื้นที่ให้คนตรวจสอบก่อน** ไม่ใช่สั่งอพยพหรือส่งทีม
อัตโนมัติโดยไม่มี operator

## จุดเด่นที่พูดได้เต็มที่

1. **จัดอันดับความเสี่ยงได้ดี:** held-out ROC-AUC = **0.9051**
2. **ใช้ทรัพยากรแบบ relative ranking ได้ดี:** top 20% ของ cell ที่ rank สูง
   ครอบคลุมไฟใน held-out history **82.93%**
3. **ประเมินแบบกัน leakage:** แบ่ง train/validation/calibration/test ตามวัน,
   เว้น purge 7 วัน และแตะ test ครั้งเดียว
4. **เทียบ baseline บนประชากรเดียวกัน:** AUC ของโมเดล 0.9051 เทียบกับ
   persistence 3 วัน 0.5898 และ persistence 7 วัน 0.6403
5. **แสดงความไม่แน่นอนตรงไปตรงมา:** feed ล้มเหลว = unavailable, sample น้อย
   = แสดง count และไม่โชว์เปอร์เซ็นต์ที่ดูมั่นใจเกินจริง
6. **ตรวจสอบย้อนหลังได้:** หน้า Predicted vs Actual ใช้ risk window ที่ปิดแล้ว
   และวันที่ FIRMS ที่ตรงกับ window
7. **หลายแหล่งข้อมูล:** FIRMS สำหรับ label/history, GISTDA สำหรับ live
   observation, ERA5/Open-Meteo สำหรับ weather context, Hansen สำหรับ land cover

## ตัวเลขปัจจุบันที่ใช้ตอบกรรมการ

### ข้อมูลและ protocol

| รายการ | ค่า |
|---|---:|
| ช่วงข้อมูล | 2025-06-11 ถึง 2026-07-16 |
| จำนวนวัน | 401 |
| historically active cells | 11,833 |
| features | 164 |
| training rows หลัง sampling เฉพาะ train | 485,190 |
| held-out test cell-days | 709,002 |
| positive cell-days ใน test | 22,992 |
| test prevalence | 3.24% |

### ผล held-out

| ค่า | ผล | วิธีพูด |
|---|---:|---|
| ROC-AUC | **0.9051** | โมเดลจัดอันดับ cell ที่จะมี hotspot สูงกว่า cell ปลอดเหตุได้ดี |
| Average Precision | 0.2175 | PR performance ภายใต้ positive rate เพียง 3.24% |
| Brier score | 0.0390 | error ของ probability โดยรวม |
| ECE | 0.0503 | calibration error โดยรวม แต่ต้องอ่านคู่ seasonal drift |

### เกณฑ์ใช้งานจริง

| Tier | threshold | precision | recall | alert rate |
|---|---:|---:|---:|---:|
| HIGH | 0.5236 | 38.50% | 11.68% | 0.98% |
| WATCH | 0.4376 | 22.72% | **31.55%** | 4.50% |

### Relative ranking

| กลุ่มอันดับ | ไฟ held-out ที่ครอบคลุม |
|---|---:|
| Top 5% | 33.91% |
| Top 10% | 52.47% |
| Top 20% | **82.93%** |

ห้ามเอา 82.93% ไปเรียกว่า WATCH recall เพราะเป็นคนละนโยบาย:

- WATCH = เกณฑ์ probability คงที่
- Top 20% = เลือกตามอันดับสัมพัทธ์เมื่อทรัพยากรมีจำกัด

## ทำไม WATCH recall เป็น 32%

ไม่ใช่ frontend bug และห้ามแก้เลขให้สวยขึ้น

- prevalence ตอน calibration = **39.33%**
- prevalence ตอน held-out test = **3.24%**
- ต่างกัน **12.13 เท่า**

เกณฑ์เดียวที่ lock ในช่วงไฟชุกจึงถ่ายไปช่วง off-season ได้ไม่ดี ทำให้ recall
ของเกณฑ์ absolute ลดลง แต่ความสามารถในการเรียงลำดับยังสูง (AUC 0.9051)

คำตอบสั้น:

> “32% คือ recall ของเกณฑ์ WATCH แบบคงที่ใน held-out off-season ครับ
> เราไม่ซ่อนข้อจำกัดนี้ แต่แสดงคู่กับผล ranking ซึ่ง top 20% ครอบคลุมไฟ
> ในอดีต 83% เพื่อให้ operator เลือกวิธีใช้ให้ตรงโจทย์”

## ลำดับเดโม 5 นาที

### 1. Map — 90 วินาที

1. ชี้ `Base date` และ `Data as of`
2. อธิบายว่าแต่ละจุดคือ cell risk สำหรับ **3-day window**
3. เปิด popup หนึ่ง cell:
   - probability;
   - absolute tier;
   - FIRMS history;
   - seasonal context;
   - weather source
4. ถ้าจุด `<0.01%`:
   - เป็นค่าบวกจริงที่เล็กมาก;
   - ช่วง ก.ค. เป็นฤดูฝน;
   - history สูงไม่ได้แปลว่าวันนี้เสี่ยงสูง
5. สาธิต `ซ่อน <0.01%` และบอกว่าเป็น display-only; CSV ยังครบ

### 2. Live Fires — 45 วินาที

อธิบายว่า GISTDA เป็น observation feed แยกจาก model prediction ให้ดูสถานะ
available / stale / unavailable และเวลาตรวจพบ ห้ามตีความ `0` เป็น “ไม่มีไฟ”
ถ้า request ล้มเหลว

### 3. Analytics — 45 วินาที

ชี้สถานะ feed ก่อนชี้ KPI ถ้า feed พร้อมแต่ไม่มี record จึงเรียกว่า confirmed
zero; ถ้า feed ไม่พร้อม ระบบต้องแสดง unavailable

### 4. Predicted vs Actual — 60 วินาที

เลือก snapshot ที่ risk window ปิดแล้ว อธิบายว่า:

- predicted window = base_date+1 ถึง base_date+3;
- FIRMS ต้องมี coverage ครบ window นั้น;
- count เล็กแสดงเป็นจำนวนจริง;
- precision/recall/F1 จะแสดงเมื่อ denominator เพียงพอเท่านั้น;
- snapshot audit ไม่ใช่ held-out training metric

### 5. Reports / Model performance — 60 วินาที

เน้น AUC 0.9051, baseline comparison, Top-20% coverage 82.93% แล้วจบด้วย
drift disclosure และ WATCH 31.55% อย่างตรงไปตรงมา

## คำถามที่น่าจะโดนถาม

### “0.00% หรือ <0.01% แปลว่าโมเดลเสียไหม?”

ไม่เสีย เป็น probability ที่เล็กแต่ไม่เป็นศูนย์ เช่น 0.000013 = 0.0013%.
ช่วงฤดูฝน โมเดลอาจให้ความเสี่ยงวันนี้ต่ำ แม้ cell นั้นเคยเกิดไฟบ่อยในฤดูไฟ

### “ทำไม HIGH/WATCH เป็นศูนย์ แต่มีจุดเต็มแผนที่?”

จุด Background ยังเป็น cell ที่ระบบประเมินแล้ว เพียงไม่มีจุดใดข้าม threshold
คงที่ ระบบไม่บังคับให้ต้องมี HIGH ทุกวัน จึงไม่สร้าง alert ปลอมในวันเสี่ยงต่ำ

### “ถ้า 32% ต่ำ ทำไมบอกว่าโมเดลดี?”

32% เป็น threshold recall ภายใต้ seasonal drift ส่วน AUC 0.9051 และ top-20%
coverage 82.93% วัด ranking ซึ่งยังแข็งแรง ต้องเลือก metric ให้ตรงกับงาน

### “ทำไมไม่ลด threshold ให้ recall สูงขึ้น?”

จุด trade-off ที่บันทึกไว้ไม่มีตัวไหนให้ recall สูงกว่า 31.55% โดยไม่เพิ่ม
false alarms มากเกินข้อมูลที่มีอยู่ และ frontend ไม่มีสิทธิ์สร้าง threshold ใหม่
ต้องแก้ด้วย calibration/retraining ที่ครอบคลุมหลายฤดู

### “Predicted vs Actual เป็น 0% แปลว่า model ไม่มี skill?”

ไม่จำเป็น หน้านี้อาจเป็น snapshot เล็กหนึ่งช่วง ต้องดูจำนวน alert/outcome และ
coverage ก่อน ส่วน AUC เป็นผล 709,002 cell-days ใน held-out test คนละหลักฐาน

### “Weather มาจากไหน?”

ใช้ ERA5 cache ที่ backend ก่อน หากไม่มีจึงเรียก Open-Meteo fallback และแสดง
source ชัดเจน ถ้าทั้งคู่ล้มเหลวจะแสดง unavailable ไม่เติมค่าปลอม

### “GISTDA เป็น ground truth ของ model ไหม?”

ไม่ใช่ label หลักของโมเดลคือ NASA FIRMS ส่วน GISTDA เป็น live observation
อีกแหล่งหนึ่ง ไม่ควรเรียกว่า independent proof แบบสมบูรณ์เพราะต่างก็เป็น
satellite hotspot products

### “ทำนายจุดไฟใหม่ที่ไม่เคยเกิดได้ไหม?”

ยังไม่ได้ population ปัจจุบันคือ cell ที่เคยมี FIRMS อย่างน้อยหนึ่งครั้ง และ
เข้า dataset ตั้งแต่วันที่ตรวจพบครั้งแรก นี่เป็นข้อจำกัดสำคัญที่เปิดเผยไว้

## สิ่งที่ห้ามพูด

- “ทำนายวันเกิดไฟได้”
- “จุดนี้จะเกิดไฟแน่นอน”
- “AUC 90.5% คือ accuracy 90.5%”
- “WATCH จับไฟได้ 83%”
- “0 live hotspots ยืนยันว่าไม่มีไฟ” เมื่อ feed unavailable
- “Compare 0% พิสูจน์ว่าโมเดลล้มเหลว” โดยไม่ดู sample/coverage
- ตัวเลขเก่า: AUC 0.827, recall 67%, threshold 0.05, percentile tier แบบบังคับ

## Opening pitch 45 วินาที

> “FireWatch Thailand เป็นระบบช่วยจัดลำดับพื้นที่เสี่ยงที่จะพบ hotspot
> ภายใน 3 วัน ไม่ใช่ระบบฟันธงวันเกิดไฟ โมเดลประเมิน cell ขนาดประมาณ 11
> กิโลเมตรจากข้อมูลดาวเทียม ประวัติไฟ พื้นที่ข้างเคียง ฤดูกาล ป่าไม้ และอากาศ
> การทดสอบแบบ chronological held-out 709,002 cell-days ได้ ROC-AUC 0.9051
> และ top 20% ของ cell ที่ rank สูงครอบคลุมไฟในประวัติ held-out 82.93%
> Dashboard ยังแยก prediction, live observation และ retrospective audit
> เพื่อให้ผู้ใช้เห็นทั้งจุดแข็งและข้อจำกัดก่อนตัดสินใจ”

## เช็กลิสต์คืนนี้

```bash
cd /home/chanakan/Documents/wild-fire

cd web
npm run typecheck
npm test
npm run build
cd ..

source .venv/bin/activate
python -m compileall -q src scripts
SCHEDULER_ENABLED=false python -m pytest
```

จากนั้น:

1. เปิดด้วย `./run.sh --demo --open` — โหมดนี้คง GISTDA live polling ไว้ แต่
   ปิด FIRMS sync และ scheduled retraining เพื่อไม่ให้ artifact เปลี่ยนกลางเดโม
2. refresh ทุก tab โดยตรง (`#analytics`, `#compare`, `#reports`, `#notify`)
3. เปิด popup และเช็ค weather source
4. ลองปิด network เพื่อดู error state
5. เตรียม screenshot/video fallback เผื่อ internet หรือ GISTDA ล่ม
6. อย่า refresh/retrain model ระหว่างกรรมการดู; `--demo` จะปฏิเสธ flag ที่
   fetch, predict หรือ train เพื่อป้องกันการสั่งโดยไม่ตั้งใจ

## แผนสำรองเมื่อ internet ล่ม

- Map และ metric จาก artifact local ยังเปิดได้
- อธิบาย timestamp อย่างตรงไปตรงมา
- Live/Analytics ต้องแสดง unavailable ไม่ใช่ zero
- Weather cache local ยังใช้ได้; ถ้า cell ไม่มี cache จะแสดง unavailable
- ใช้ screenshot/video ที่เตรียมไว้เพื่อสาธิต live state ก่อนหน้า โดยบอกว่าเป็น
  ภาพบันทึก ไม่ใช่ข้อมูลสด
