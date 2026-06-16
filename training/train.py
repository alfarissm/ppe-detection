# -*- coding: utf-8 -*-
"""
Retrain YOLO11 PPE detector dengan augmentasi agresif + resolusi tinggi.
Tujuan: naikin akurasi kelas 'shoes' (objek kecil) tanpa nambah data.

Jalankan (GPU lokal / Colab):
    pip install ultralytics
    python train.py

Hasil: runs/detect/ppe_aug/weights/best.pt  -> salin ke ../app/best.pt
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # cegah crash konflik OpenMP (Anaconda)
from ultralytics import YOLO

# --- path dataset ---
# arahkan ke data.yaml. Default: dataset/ di root proyek.
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get(
    "DATA_YAML",
    os.path.join(HERE, "..", "dataset", "data.yaml"),
)

# --- model dasar (transfer learning) ---
model = YOLO("yolo11s.pt")

# --- training ---
model.train(
    data=DATA,
    epochs=150,
    imgsz=1024,          # resolusi tinggi -> objek kecil (shoes); 4GB tahan ~1024
    batch=4,             # OOM -> turunkan ke 2
    name="ppe_aug",
    patience=40,         # early stop kalau stagnan
    seed=0,
    cache=False,
    workers=0,           # Windows: hindari crash multiprocessing

    # ---- augmentasi (fokus objek kecil/jarang) ----
    mosaic=1.0,          # gabung 4 citra
    close_mosaic=15,     # matikan mosaic 15 epoch terakhir (stabilkan)
    copy_paste=0.3,      # duplikat instance objek -> bantu kelas jarang
    mixup=0.15,          # blend 2 citra
    scale=0.9,           # variasi skala objek
    fliplr=0.5,          # flip horizontal
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,   # variasi warna/cahaya
    degrees=5.0,         # rotasi kecil
    translate=0.1,
)

# --- evaluasi pada data validasi ---
metrics = model.val(data=DATA, imgsz=1024)
print("\n=== RINGKASAN ===")
print(f"mAP@0.5      : {metrics.box.map50:.3f}")
print(f"mAP@0.5:0.95 : {metrics.box.map:.3f}")
for i, name in model.names.items():
    print(f"  {name:8s} mAP@0.5 = {metrics.box.maps[i]:.3f}")
print("\nbest.pt ada di:", os.path.join("runs", "detect", "ppe_aug", "weights", "best.pt"))
print("Salin ke ../app/best.pt untuk dipakai aplikasi.")
