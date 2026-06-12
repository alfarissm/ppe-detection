# PPE Detection — Construction Safety Vision

Deteksi **Alat Pelindung Diri** (Personal Protective Equipment) pada citra lokasi
kerja konstruksi menggunakan **YOLO11** + **Flask**. Sistem memeriksa kelengkapan
*helmet*, *vest*, dan *shoes*, lalu menerbitkan **status kepatuhan** beserta
tingkat keyakinan deteksi per kelas.

![Screenshot aplikasi](docs/screenshot.png)

## Fitur

- Upload citra (drag-drop) lalu inferensi YOLO11 satu klik
- Tampilan **source vs analyzed** berdampingan dengan *bounding box*
- **Verdict kepatuhan** — Compliant / Non-Compliant berdasar kelengkapan APD
- **Detection log** — jumlah objek + confidence + status per kelas
- UI tema *dark console monitoring*, tanpa database, tanpa login

## Model

| Item | Nilai |
|------|-------|
| Arsitektur | YOLO11s (*small*) |
| Parameter | 9.413.961 (±9,4 jt) · 21,3 GFLOPs |
| Training | Transfer learning dari `yolo11s.pt`, 100 epoch |
| Resolusi | 640×640 |
| Kelas | `helmet`, `vest`, `shoes` (3 kelas) |
| Dataset | Roboflow *ppe-detection-yolo* v3 — 400 citra |
| Threshold inferensi | conf 0.25 |

## Akurasi (data validasi)

**Keseluruhan:** Precision **0.558** · Recall **0.733** · mAP@0.5 **0.653** · mAP@0.5:0.95 **0.521**

| Kelas | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|--------|:---------:|:------:|:-------:|:------------:|
| helmet | 0.807 | 1.000 | **0.986** | 0.896 |
| vest   | 0.810 | 0.963 | **0.928** | 0.653 |
| shoes  | 0.056 | 0.236 | **0.046** | 0.014 |

**Catatan jujur:** model **sangat akurat** untuk *helmet* & *vest* (objek besar,
kontras). Kelas *shoes* **lemah** (mAP@0.5 0.046) — objek kecil, sering tertutup,
kontras rendah dengan tanah, banyak terklasifikasi sebagai *background*. Ini
karakteristik umum *single-stage detector* pada objek kecil. Perbaikan ke depan:
tambah data *shoes* beragam sudut + teknik objek kecil (tiling / resolusi lebih
tinggi).

## Struktur

```
.
├─ app/                  # aplikasi Flask
│  ├─ app.py             # server + route /detect
│  ├─ best.pt            # model YOLO11s terlatih
│  ├─ requirements.txt
│  ├─ Dockerfile         # deploy container / HF Spaces
│  ├─ Procfile           # deploy Render / Railway
│  ├─ templates/         # dashboard.html, result.html
│  └─ static/            # uploads & results (runtime)
└─ docs/                 # screenshot
```

## Jalankan lokal

```bash
cd app
pip install -r requirements.txt
python app.py
```

Buka http://127.0.0.1:5001

## Deploy (gratis)

**Rekomendasi: HuggingFace Spaces (Docker)** — RAM/disk besar, muat torch + model.

1. Buat Space baru → SDK **Docker** → Blank.
2. Upload isi folder `app/` (termasuk `Dockerfile` & `best.pt`).
3. Space auto-build & jalan di port 7860.

Alternatif: **Render** (Web Service, free) pakai `Procfile`. Catatan: free tier
512MB RAM bisa kurang untuk torch — HF Spaces lebih aman.

## Tech stack

Python · Flask · Ultralytics YOLO11 · PyTorch · OpenCV · Jinja2 · HTML/CSS/JS
