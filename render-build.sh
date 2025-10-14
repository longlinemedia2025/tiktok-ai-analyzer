#!/usr/bin/env bash
# ===== Render build fix for MoviePy =====
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Force reinstall of MoviePy and FFmpeg-related packages
pip install --force-reinstall moviepy imageio[ffmpeg] numpy

echo "Build complete!"
