# AI Camera – Hidden Dangerous Object Detection (Baggage/X-ray)

Model deteksi objek berbahaya (gun, knife, pliers, wrench, scissors, hammer) pada citra X-ray bagasi/tas. 
Pipeline lengkap: download dataset (Kaggle), konversi label ke YOLO, augmentasi sintetik, training YOLOv8, evaluasi, 
inferensi kamera real-time (OpenCV), logging ke DB, dan REST API (FastAPI).

## Cepat Mulai
1) Python 3.10+, pasang dependency:
