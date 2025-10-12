import sys
import site
import os

# ===== Render Path Fix =====
# Ensure Render's virtual environment site-packages directory is visible to Python
site.addsitedir("/opt/render/project/src/.venv/lib/python3.13/site-packages")
sys.path.append("/opt/render/project/src/.venv/lib/python3.13/site-packages")

print("PYTHONPATH:", sys.path)  # Debug: confirm path visibility on Render

# ===== Core Imports =====
from flask import Flask, request, jsonify
import datetime
import csv
from moviepy.editor import VideoFileClip  # <- this should now import fine
import numpy as np
import openai

# ===== CONFIG =====
openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)

# ===== HELPER FUNCTIONS =====
def analyze_video_properties(video_path):
    clip = VideoFileClip(video_path)
    duration = clip.duration
    fps = clip.fps
    resolution = clip.size
    clip.close()
    return {
        "duration": duration,
        "fps": fps,
        "resolution": resolution
    }

# ===== ROUTES =====
@app.route("/")
def home():
    return jsonify({"message": "TikTok Algorithm Rater Tool is running successfully!"})

@app.route("/analyze", methods=["POST"])
def analyze():
    if "video" not in request.files:
        return jsonify({"error": "No video uploaded"}), 400
    
    video = request.files["video"]
    video_path = os.path.join("/tmp", video.filename)
    video.save(video_path)

    result = analyze_video_properties(video_path)
    os.remove(video_path)

    return jsonify(result)

# ===== MAIN =====
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
