import argparse, cv2, os, sys
from ultralytics import YOLO

# ==== robust import agar bisa dipanggil `python src/inference_cam.py` ====
try:
    from src.utils import ensure_schema_sqlite, log_detection
except ModuleNotFoundError:
    REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if REPO_ROOT not in sys.path:
        sys.path.append(REPO_ROOT)
    from utils import ensure_schema_sqlite, log_detection
# =======================================================================

def pick_weights(path: str) -> str:
    if os.path.isfile(path):
        return path
    # coba best→last di folder yang diberikan
    for cand in ("best.pt", "last.pt"):
        p = os.path.join(path, cand)
        if os.path.isfile(p):
            return p
        p2 = os.path.join(path, "weights", cand)
        if os.path.isfile(p2):
            return p2
    # fallback
    return "yolov8m.pt"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="models/.../[best|last].pt ATAU folder run")
    ap.add_argument("--source", default="0", help="id kamera (mis. '0') atau path video file/URL")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--db", default="detections.db")
    args = ap.parse_args()

    weights = pick_weights(args.weights)
    print(f"[infer] Memakai weights: {weights}")

    ensure_schema_sqlite(args.db)
    model = YOLO(weights)

    # dukung camera id numerik atau path video
    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Tidak bisa membuka sumber video: {args.source}")

    print("Tekan 'q' untuk keluar.")
    camera_id = f"cam_{args.source}"

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model.predict(source=frame, imgsz=args.imgsz, conf=args.conf, verbose=False)
        annotated = frame.copy()

        for r in results:
            names = r.names
            for b in r.boxes:
                conf = float(b.conf.cpu().numpy())
                cls_id = int(b.cls.cpu().numpy())
                x1, y1, x2, y2 = map(int, b.xyxy.cpu().numpy()[0])
                label = f"{names[cls_id]} {conf:.2f}"
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(annotated, label, (x1, max(0, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                log_detection(args.db, camera_id, names[cls_id], conf, [x1, y1, x2, y2])

        cv2.imshow("AI Camera – X-Ray Detector", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
