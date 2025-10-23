#!/usr/bin/env bash
# ===== Render build fix for MoviePy + Tesseract =====

echo "Installing system packages..."
apt-get update && apt-get install -y tesseract-ocr

echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Force reinstall of MoviePy and FFmpeg-related packages
pip install --force-reinstall moviepy imageio[ffmpeg] numpy

echo "Build complete!"
