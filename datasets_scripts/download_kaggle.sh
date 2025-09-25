#!/usr/bin/env bash
set -e

# Pastikan ~/.kaggle/kaggle.json ada dan permission 600
mkdir -p ./data/raw

# Contoh dataset X-ray (ganti slug jika Anda pakai dataset lain):
kaggle datasets download -d orvile/x-ray-baggage-anomaly-detection -p ./data/raw/xray_baggage --unzip
kaggle datasets download -d orvile/hums-x-ray-dataset -p ./data/raw/hums --unzip

echo "Selesai download. Silakan cek folder data/raw/"
