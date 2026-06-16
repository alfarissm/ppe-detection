from flask import Flask, render_template, request
from ultralytics import YOLO
import os
import uuid
import shutil

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
RESULT_FOLDER = os.path.join(BASE_DIR, "static", "results")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

model = YOLO(os.path.join(BASE_DIR, "best.pt"))


@app.route("/")
def dashboard():
    return render_template("dashboard.html")


@app.route("/detect", methods=["POST"])
def detect():
    file = request.files["image"]
    filename = str(uuid.uuid4()) + ".jpg"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # Jalankan deteksi (hanya helmet & vest; shoes di-drop karena akurasi rendah)
    results = model.predict(source=filepath, conf=0.25, save=True, classes=[0, 2])

    # ← TAMBAH: Salin hasil deteksi ke static/results/
    result_dir = results[0].save_dir          
    result_src = os.path.join(result_dir, filename)   
    result_dst = os.path.join(RESULT_FOLDER, filename)  
    shutil.copy(result_src, result_dst)       

    # Hitung objek + confidence rata-rata per kelas (helmet & vest)
    helmet = vest = 0
    conf_sum = {"helmet": 0.0, "vest": 0.0}
    names = model.names

    for box in results[0].boxes:
        cls = int(box.cls[0])
        label = names[cls]
        c = float(box.conf[0])
        if label == "helmet":
            helmet += 1
            conf_sum["helmet"] += c
        elif label == "vest":
            vest += 1
            conf_sum["vest"] += c

    counts = {"helmet": helmet, "vest": vest}
    total = helmet + vest

    # Confidence rata-rata (persen) per kelas, 0 bila tak terdeteksi
    conf = {
        k: round(conf_sum[k] / counts[k] * 100) if counts[k] else 0
        for k in counts
    }

    # Verdict kepatuhan: butuh helmet & vest
    missing = [k for k in ("helmet", "vest") if counts[k] == 0]
    compliant = len(missing) == 0

    return render_template(
        "result.html",
        image=filename,       # filename yang sama dipakai untuk hasil
        helmet=helmet,
        vest=vest,
        total=total,
        conf=conf,
        missing=missing,
        compliant=compliant,
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    debug = os.environ.get("FLASK_DEBUG", "1") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)