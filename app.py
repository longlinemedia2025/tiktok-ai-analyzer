from flask import Flask, request, jsonify, render_template
import os
import datetime
import csv
from moviepy.editor import VideoFileClip
from openai import OpenAI
import numpy as np
from PIL import Image

# ========== CONFIG ==========
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
app = Flask(__name__)

# ========== HELPER FUNCTIONS ==========

def analyze_video_properties(video_path):
    """Extract key properties like duration, resolution, fps, brightness, and aspect ratio."""
    clip = VideoFileClip(video_path)
    duration = clip.duration
    width, height = clip.size
    fps = clip.fps

    # Compute brightness (sample every N frames)
    brightness_samples = []
    for frame in clip.iter_frames(fps=max(1, int(fps / 5)), dtype="uint8"):
        brightness = np.mean(frame)
        brightness_samples.append(brightness)
    avg_brightness = round(np.mean(brightness_samples), 2)

    clip.close()

    aspect_ratio = round(width / height, 3)
    tone = "bright" if avg_brightness > 150 else "dark" if avg_brightness < 80 else "neutral or mixed"

    return {
        "duration_seconds": round(duration, 2),
        "resolution": f"{width}x{height}",
        "frame_rate": round(fps, 2),
        "aspect_ratio": aspect_ratio,
        "brightness": avg_brightness,
        "tone": tone
    }


def generate_ai_analysis(video_path, video_info):
    """Generate full viral insights with OpenAI."""
    prompt = f"""
You are an AI TikTok strategist who specializes in viral video optimization.

Analyze this TikTok video based on its data:
- Duration: {video_info['duration_seconds']}s
- Resolution: {video_info['resolution']}
- Aspect Ratio: {video_info['aspect_ratio']}
- Brightness: {video_info['brightness']}
- Tone: {video_info['tone']}
- Frame Rate: {video_info['frame_rate']}

The file name may suggest its niche: "{os.path.basename(video_path)}"

Please produce a **comprehensive viral report** in this format:

🎬 Video Analysis Report
✅ Basic Video Details:
- Duration
- Resolution
- Aspect Ratio
- Brightness
- Tone
- Frame Rate

💬 AI-Generated Viral Insights:
1. Scroll-Stopping Caption
2. 5 Viral Hashtags
3. Actionable Improvement Tip
4. Viral Optimization Score (1–100)
5. Motivation Tip for Increasing Virality

🔥 Viral Comparison Results:
Include 3 short examples of similar viral TikToks in the same niche, describe why they went viral, and how to replicate that success.

📋 Final Actionable Checklist:
Give 4–5 bullet points on what the creator should do next to improve their TikTok performance.

Be detailed, accurate, niche-specific (based on the filename), and formatted in Markdown style.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a professional social media strategist for viral TikToks."},
                {"role": "user", "content": prompt}
            ]
        )

        ai_text = response.choices[0].message.content.strip()
        return {"ai_suggestions": ai_text}

    except Exception as e:
        return {"error": str(e)}


# ========== ROUTES ==========

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze_video():
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    video_file = request.files["video"]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = os.path.join("uploads", f"video_{timestamp}.mp4")

    os.makedirs("uploads", exist_ok=True)
    video_file.save(video_path)

    # Analyze properties and run AI
    video_info = analyze_video_properties(video_path)
    ai_results = generate_ai_analysis(video_path, video_info)
    result = {**video_info, **ai_results}

    # Save results in a CSV log
    log_path = "video_analysis_log.csv"
    file_exists = os.path.isfile(log_path)
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=result.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)

    return jsonify(result)


# ========== RUN LOCAL ==========
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
