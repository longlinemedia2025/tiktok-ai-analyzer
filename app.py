from flask import Flask, request, jsonify
import os
import sys
import datetime
import csv
import numpy as np
import importlib.util

# ========== DEBUG SECTION FOR RENDER ==========
print("\n========== Render Build Diagnostic ==========")
print("Python version:", sys.version)
print("Working directory:", os.getcwd())
print("PYTHONPATH:", sys.path)
print("MoviePy module found?:", importlib.util.find_spec("moviepy"))
print("================================================\n")

# Try to import moviepy and show a clear error if it fails
try:
    from moviepy.editor import VideoFileClip
    print("✅ MoviePy successfully imported!\n")
except ModuleNotFoundError as e:
    print("❌ MoviePy failed to import:", e)
    raise

# ========== CONFIG ==========
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")  # Make sure this is set in Render environment

app = Flask(__name__)

# ========== HELPER FUNCTIONS ==========

def analyze_video_properties(video_path):
    """Extract basic video stats for AI analysis."""
    clip = VideoFileClip(video_path)
    duration = clip.duration
    fps = clip.fps
    width, height = clip.size
    aspect_ratio = width / height
    clip.close()

    return {
        "duration": duration,
        "fps": fps,
        "width": width,
        "height": height,
        "aspect_ratio": aspect_ratio
    }

def generate_ai_analysis(video_stats):
    """Ask OpenAI to analyze stats and suggest captions/hashtags."""
    prompt = f"""
    You are a TikTok content strategist. A video has these properties:
    Duration: {video_stats['duration']}s
    FPS: {video_stats['fps']}
    Resolution: {video_stats['width']}x{video_stats['height']} ({video_stats['aspect_ratio']:.2f} ratio)

    Based on these, give:
    1. A rating (1–10) for algorithm potential
    2. 3 engaging caption ideas
    3. 5 recommended hashtags
    """

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content.strip()

def save_analysis_to_csv(video_stats, ai_analysis):
    """Save analysis results to a CSV file for tracking."""
    file_exists = os.path.isfile("video_analysis.csv")
    with open("video_analysis.csv", "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(["Timestamp", "Duration", "FPS", "Width", "Height", "Aspect Ratio", "AI Analysis"])
        writer.writerow([
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            video_stats["duration"],
            video_stats["fps"],
            video_stats["width"],
            video_stats["height"],
            video_stats["aspect_ratio"],
            ai_analysis
        ])

# ========== ROUTES ==========

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "TikTok Algorithm Rater Tool is running!"})

@app.route("/analyze", methods=["POST"])
def analyze_video():
    """Handle uploaded video and return AI-based suggestions."""
    if "video" not in request.files:
        return jsonify({"error": "No video file uploaded"}), 400

    video_file = request.files["video"]
    video_path = os.path.join("/tmp", video_file.filename)
    video_file.save(video_path)

    try:
        stats = analyze_video_properties(video_path)
        ai_feedback = generate_ai_analysis(stats)
        save_analysis_to_csv(stats, ai_feedback)
        os.remove(video_path)
        return jsonify({"video_stats": stats, "ai_feedback": ai_feedback})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ========== MAIN ENTRY POINT ==========

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Server running on port {port}")
    app.run(host="0.0.0.0", port=port)
