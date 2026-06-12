# PPE Detection — Construction Safety Vision

Deteksi **Alat Pelindung Diri** (Personal Protective Equipment) pada citra lokasi
kerja konstruksi menggunakan **YOLO11** + **Flask**. Sistem memeriksa kelengkapan
*helmet*, *vest*, dan *shoes*, lalu menerbitkan **status kepatuhan** beserta
tingkat keyakinan deteksi per kelas.

![Screenshot aplikasi](docs/screenshot.png)

---

## Daftar Isi
- [Fitur](#fitur)
- [Cara Kerja](#cara-kerja)
- [Dataset](#dataset)
- [Training](#training)
- [Hasil & Akurasi](#hasil--akurasi)
- [Struktur Proyek](#struktur-proyek)
- [Menjalankan](#menjalankan)
- [Tech Stack](#tech-stack)

---

## Fitur

- Upload citra (drag-drop) lalu inferensi YOLO11 satu klik
- Tampilan **source vs analyzed** berdampingan dengan *bounding box*
- **Verdict kepatuhan** — Compliant / Non-Compliant berdasar kelengkapan APD
- **Detection log** — jumlah objek + confidence + status per kelas
- UI tema *dark console monitoring*, tanpa database, tanpa login

## Cara Kerja

```
Citra di-upload  →  YOLO11 (best.pt) prediksi  →  Hitung objek per kelas
      →  Cek kelengkapan (helmet + vest + shoes)  →  Render verdict + log
```

1. Pengguna upload citra lewat dashboard.
2. `model.predict(conf=0.25)` menghasilkan *bounding box* + label + confidence.
3. Backend menghitung jumlah & rata-rata confidence tiap kelas.
4. Verdict **Compliant** bila ketiga APD terdeteksi; selain itu **Non-Compliant**
   dengan daftar APD yang hilang.

## Dataset

Sumber: **Roboflow Universe** — proyek *ppe-detection-yolo* v3, format YOLO.

| Split | Jumlah citra |
|-------|:------------:|
| Train | 369 |
| Valid | 15 |
| Test  | 16 |
| **Total** | **400** |

- **Pra-pemrosesan:** auto-orientation, resize 640×640
- **Augmentasi:** horizontal flip (50%), brightness ±15%, Gaussian blur (0–1 px),
  salt-and-pepper noise (0,1%)
- **Kelas:** `0: helmet`, `1: shoes`, `2: vest`

## Training

| Konfigurasi | Nilai |
|-------------|-------|
| Base model | `yolo11s.pt` (pretrained, transfer learning) |
| Arsitektur | YOLO11s — 101 layer, 9.413.961 param, 21,3 GFLOPs |
| Epoch | 100 |
| Image size | 640×640 |
| Optimizer | default Ultralytics (auto) |
| Output | `best.pt` (model terbaik) |

## Hasil & Akurasi

**Keseluruhan (data validasi):**
Precision **0.558** · Recall **0.733** · mAP@0.5 **0.653** · mAP@0.5:0.95 **0.521**

| Kelas | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|--------|:---------:|:------:|:-------:|:------------:|
| helmet | 0.807 | 1.000 | **0.986** | 0.896 |
| vest   | 0.810 | 0.963 | **0.928** | 0.653 |
| shoes  | 0.056 | 0.236 | **0.046** | 0.014 |

### Precision–Recall Curve
Kurva *helmet* (0.986) & *vest* (0.928) menempel pojok kanan-atas (ideal),
sedangkan *shoes* (0.046) datar di bawah — menarik rata-rata mAP turun.

![Precision-Recall curve](docs/pr_curve.png)

### F1–Confidence Curve
Titik F1 optimal untuk kelas kuat berada di rentang confidence menengah.

![F1 curve](docs/f1_curve.png)

### Confusion Matrix (normalized)
*helmet* & *vest* terklasifikasi benar dengan baik; kesalahan terkonsentrasi pada
*shoes* yang banyak tertukar menjadi *background*.

![Confusion matrix](docs/confusion_matrix.png)

### Contoh Deteksi (validation batch)
![Sample predictions](docs/sample_predictions.jpg)

### Analisis
- **helmet & vest — sangat baik.** Objek besar, kontras tinggi terhadap latar,
  jumlah anotasi memadai.
- **shoes — lemah (mAP@0.5 0.046).** Objek kecil, sering *occlusion*, kontras
  rendah dengan permukaan tanah, dan data validasi terbatas (15 citra). Ini
  karakteristik umum *single-stage detector* pada objek kecil.
- **Perbaikan ke depan:** tambah variasi data *shoes* (sudut/jarak beragam),
  pakai teknik objek kecil (tiling / resolusi lebih tinggi), tambah epoch.

## Struktur Proyek

```
.
├─ app/                  # aplikasi Flask
│  ├─ app.py             # server + route /detect
│  ├─ best.pt            # model YOLO11s terlatih
│  ├─ requirements.txt
│  ├─ templates/         # dashboard.html, result.html
│  └─ static/            # uploads & results (runtime)
└─ docs/                 # screenshot + grafik akurasi
```

## Menjalankan

```bash
cd app
pip install -r requirements.txt
python app.py
```

Buka **http://127.0.0.1:5001**

## Tech Stack

Python · Flask · Ultralytics YOLO11 · PyTorch · OpenCV · Jinja2 · HTML/CSS/JS
