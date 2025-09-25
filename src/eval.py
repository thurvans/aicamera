import argparse, os, glob
from ultralytics import YOLO

def pick_weights(path_or_dir: str) -> str:
    """Terima file .pt langsung, atau folder models/.../weights/ (auto pilih best→last),
    atau kosong -> fallback ke yolov8m.pt."""
    if os.path.isfile(path_or_dir):
        return path_or_dir
    candidates = []
    if os.path.isdir(path_or_dir):
        candidates += sorted(glob.glob(os.path.join(path_or_dir, "best.pt")))
        candidates += sorted(glob.glob(os.path.join(path_or_dir, "last.pt")))
    # kalau user mengirim folder project model (models/run_name/)
    if os.path.isdir(path_or_dir) and not candidates:
        wdir = os.path.join(path_or_dir, "weights")
        candidates += sorted(glob.glob(os.path.join(wdir, "best.pt")))
        candidates += sorted(glob.glob(os.path.join(wdir, "last.pt")))
    # fallback
    if not candidates:
        print("[eval] Peringatan: weights tidak ditemukan, fallback ke yolov8m.pt (pretrained).")
        return "yolov8m.pt"
    return candidates[0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="path ke .pt atau folder run (otomatis best→last)")
    ap.add_argument("--cfg", required=True, help="path data.yaml")
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    weights = pick_weights(args.weights)
    print(f"[eval] Memakai weights: {weights}")

    model = YOLO(weights)
    metrics = model.val(data=args.cfg, imgsz=args.imgsz)
    print("mAP@0.5      :", metrics.results_dict.get("metrics/mAP50"))
    print("mAP@0.5:0.95 :", metrics.results_dict.get("metrics/mAP50-95"))

if __name__ == "__main__":
    main()
