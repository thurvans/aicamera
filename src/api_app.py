import os, sqlite3
import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from datetime import datetime
import sys

# ==== robust import agar bisa `uvicorn src.api_app:app` ====
try:
    from src.utils import ensure_schema_sqlite, log_detection
except ModuleNotFoundError:
    REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if REPO_ROOT not in sys.path:
        sys.path.append(REPO_ROOT)
    from utils import ensure_schema_sqlite, log_detection
# ==========================================================

DB_PATH = os.getenv("SQLITE_PATH", "detections.db")

def pick_weights() -> str:
    # Urutan preferensi: ENV → best → last → yolov8m.pt
    envw = os.getenv("WEIGHTS_PATH", "").strip()
    if envw and os.path.exists(envw):
        return envw
    for cand in (
        "models/yolov8_xray/weights/best.pt",
        "models/yolov8_xray/weights/last.pt",
        "models/yolov8_xray4/weights/best.pt",
        "models/yolov8_xray4/weights/last.pt",
    ):
        if os.path.exists(cand):
            return cand
    return "yolov8m.pt"

WEIGHTS = pick_weights()

app = FastAPI(title="AI Camera API", debug=True)
ensure_schema_sqlite(DB_PATH)
model = YOLO(WEIGHTS)

@app.get("/health")
def health():
    return {"status": "ok", "weights": WEIGHTS}

@app.post("/upload")
async def upload_image(file: UploadFile = File(...),
                       conf: float = Query(0.35),
                       imgsz: int = Query(640)):
    raw = await file.read()
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    results = model.predict(source=img, imgsz=imgsz, conf=conf, verbose=False)

    dets = []
    for r in results:
        names = r.names
        for b in r.boxes:
            confv = float(b.conf.cpu().numpy())
            cls_id = int(b.cls.cpu().numpy())
            x1, y1, x2, y2 = map(int, b.xyxy.cpu().numpy()[0])
            det = {
                "timestamp": datetime.utcnow().isoformat(),
                "camera_id": "api_upload",
                "class": names[cls_id],
                "confidence": confv,
                "bbox": [x1, y1, x2, y2]
            }
            dets.append(det)
            log_detection(DB_PATH, "api_upload", names[cls_id], confv, det["bbox"])

    return JSONResponse({"detections": dets})

@app.get("/detections")
def list_detections(limit: int = 50, class_name: str | None = None):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    if class_name:
        cur.execute(
            "SELECT id,timestamp,camera_id,class,confidence,bbox,image_path "
            "FROM detections WHERE class=? ORDER BY id DESC LIMIT ?", (class_name, limit)
        )
    else:
        cur.execute(
            "SELECT id,timestamp,camera_id,class,confidence,bbox,image_path "
            "FROM detections ORDER BY id DESC LIMIT ?", (limit,)
        )
    rows = cur.fetchall()
    conn.close()
    items = [{
        "id": r[0], "timestamp": r[1], "camera_id": r[2], "class": r[3],
        "confidence": r[4], "bbox": r[5], "image_path": r[6]
    } for r in rows]
    return JSONResponse({"items": items})
