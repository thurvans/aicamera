import argparse, os, shutil, random
from pathlib import Path
from tqdm import tqdm
import yaml

def ensure_dirs(base: Path):
    (base/"train/images").mkdir(parents=True, exist_ok=True)
    (base/"train/labels").mkdir(parents=True, exist_ok=True)
    (base/"val/images").mkdir(parents=True, exist_ok=True)
    (base/"val/labels").mkdir(parents=True, exist_ok=True)

def gather_images(root: Path):
    exts = (".jpg",".jpeg",".png",".bmp")
    imgs = []
    for p in root.rglob("*"):
        if p.suffix.lower() in exts and "images" in p.as_posix():
            imgs.append(p)
    return imgs

def split_copy(imgs, out_dir: Path, split: float):
    random.shuffle(imgs)
    n_train = int(len(imgs)*split)
    train, val = imgs[:n_train], imgs[n_train:]

    for subset, arr in [("train", train), ("val", val)]:
        for img in tqdm(arr, desc=f"Copy {subset}"):
            # label text dengan nama sama di folder labels
            lbl = Path(str(img).replace("/images/","/labels/")).with_suffix(".txt")
            out_img = out_dir/subset/"images"/img.name
            out_lbl = out_dir/subset/"labels"/lbl.name
            shutil.copy2(img, out_img)
            if lbl.exists():
                shutil.copy2(lbl, out_lbl)

def write_yaml(path: Path, out_dir: Path, classes):
    cfg = {
        "train": str(out_dir/"train/images"),
        "val": str(out_dir/"val/images"),
        "names": classes
    }
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="folder dataset YOLO (punya images/ dan labels/)")
    ap.add_argument("--out", required=True, help="keluarannya (data/processed)")
    ap.add_argument("--split", type=float, default=0.9)
    ap.add_argument("--classes", nargs="+", default=['gun','knife','wrench','pliers','scissors','hammer'])
    args = ap.parse_args()

    src = Path(args.source)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ensure_dirs(out)

    imgs = gather_images(src)
    if not imgs:
        raise SystemExit("Tidak ditemukan gambar di subfolder 'images'. Pastikan struktur YOLO (images/ + labels/).")

    split_copy(imgs, out, args.split)
    write_yaml(out/"data.yaml", out, args.classes)
    print("Selesai. data.yaml dibuat di:", out/"data.yaml")

if __name__ == "__main__":
    main()
