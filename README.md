# 🛡️ AI Camera – X-Ray Dangerous Object Detector (Baggage/X-ray)

Proyek ini adalah sistem kamera AI untuk mendeteksi **benda berbahaya** (misalnya: gun, knife, wrench, pliers, scissors, hammer) pada gambar/video hasil X-ray bagasi.  
Dibangun menggunakan **YOLOv8 + FastAPI**, mendukung training, evaluasi, inferensi kamera/gambar/video, dan logging hasil ke SQLite.

---

## 📦 Persiapan Lingkungan

Gunakan Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -U pip
# Torch CPU-only (ringan di server / Codespaces)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Paket utama
pip install ultralytics fastapi uvicorn numpy pandas pillow pyyaml tqdm opencv-python-headless
