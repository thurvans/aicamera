import argparse
from ultralytics import YOLO
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True, help="data.yaml")
    ap.add_argument("--model", default="yolov8s.pt", help="yolov8s.pt / yolov8m.pt / yolov8l.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--project", default="models")
    ap.add_argument("--name", default="yolov8_xray")
    args = ap.parse_args()

    Path(args.project).mkdir(parents=True, exist_ok=True)
    model = YOLO(args.model)
    model.train(
        data=args.cfg,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        optimizer="SGD",
        patience=30,
        project=args.project,
        name=args.name,
        save=True,
        val=True
    )
    print("Training selesai.")

if __name__ == "__main__":
    main()
