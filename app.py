from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import cv2
import numpy as np
import pandas as pd
from moviepy.editor import VideoFileClip
from openai import OpenAI
import tempfile
import datetime

app = Flask(__name__, template_folder="templates")
CORS(app)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Helper Functions ---

def analyze_video_properties(video_path):
    """Extract duration, brightness, resolution, etc."""
    clip = VideoFileClip(video_path)
    duration = clip.duration
    frame = clip.get_frame(0)
    brightness = np.mean(frame)
    height, width, _ = frame.shape
    aspect_ratio = round(width / height, 3)
    clip.reader.close()
    return {
        "duration": round(duration, 2),
        "brightness": round(brightness, 2),
        "resolution": f"{width}x{height}",
        "aspect_ratio": aspect_ratio
    }

def load_csv_data(csv_file):
    """Load performance data from CSV if provided."""
    try:
        df = pd.read_csv(csv_file)
        return df.describe(include="all").to_dict()
    except Exception:
        return None

def generate_ai_response(prompt):
    """Send the constructed prompt to OpenAI."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert in social media algorithms, engagement, and virality prediction."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.8,
    )
    return response.choices[0].message.content.strip()

def format_results(platform, video_name, video_info, ai_text):
    """Keep consistent display format across platforms."""
    return f"""
✅ {platform} Video Analysis Complete!

🎬 Video: {video_name}  
📏 Duration: {video_info['duration']}s  
🖼 Resolution: {video_info['resolution']}  
📱 Aspect Ratio: {video_info['aspect_ratio']}  
💡 Brightness: {video_info['brightness']}  

{ai_text}
"""

# --- TikTok Analyzer ---
@app.route("/analyze_tiktok", methods=["POST"])
def analyze_tiktok():
    video = request.files.get("video")
    csv_file = request.files.get("csv")

    if not video:
        return jsonify({"error": "No video file uploaded."}), 400

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        video.save(tmp.name)
        video_path = tmp.name

    video_info = analyze_video_properties(video_path)
    csv_data = load_csv_data(csv_file) if csv_file else None

    prompt = f"""
Analyze this TikTok video for virality potential based on duration, brightness, aspect ratio, and style.
Video stats: {video_info}.
If CSV data is available, use it to learn what has performed best before: {csv_data}.
Provide results in this format exactly:

💬 AI-Generated Viral Insights:
1. Scroll-Stopping Caption
2. 5 Viral + 3 Low-Competition Hashtags
3. Actionable Improvement Tip for Engagement
4. Viral Optimization Score (1–100)
5. Motivation Tip
6. 3 Viral Comparison Examples
7. Takeaway Strategy
8. Actionable Checklist
9. Best Time to Post (Platform & Day Specific)
"""

    ai_text = generate_ai_response(prompt)
    results_text = format_results("TikTok", video.filename, video_info, ai_text)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs("output", exist_ok=True)
    csv_output_path = f"output/tiktok_ai_results_{timestamp}.csv"
    pd.DataFrame([{"results": results_text}]).to_csv(csv_output_path, index=False)

    return jsonify({"results": results_text, "csv_saved": csv_output_path})


# --- YouTube Analyzer ---
@app.route("/analyze_youtube", methods=["POST"])
def analyze_youtube():
    video = request.files.get("video")
    csv_file = request.files.get("csv")

    if not video:
        return jsonify({"error": "No video file uploaded."}), 400

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        video.save(tmp.name)
        video_path = tmp.name

    video_info = analyze_video_properties(video_path)
    csv_data = load_csv_data(csv_file) if csv_file else None

    prompt = f"""
Analyze this YouTube video using YouTube’s virality and SEO algorithm signals.
Video stats: {video_info}.
If CSV data is available, analyze it to adjust recommendations: {csv_data}.
Consider CTR (click-through rate), retention, average view duration, and title optimization.

Provide your output in this format exactly:

💬 AI-Generated YouTube Insights:
1. Optimized Viral Title
2. 10 Keyword Suggestions (mix of high and low competition)
3. Description Template for SEO
4. Thumbnail & Hook Suggestions
5. Retention Improvement Tip
6. Predicted CTR Range (%)
7. Estimated Viral Score (1–100)
8. 3 Successful YouTube Comparisons
9. Takeaway Strategy
10. Actionable Checklist
11. Best Time to Publish (Platform & Day Specific)
"""

    ai_text = generate_ai_response(prompt)
    results_text = format_results("YouTube", video.filename, video_info, ai_text)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs("output", exist_ok=True)
    csv_output_path = f"output/youtube_ai_results_{timestamp}.csv"
    pd.DataFrame([{"results": results_text}]).to_csv(csv_output_path, index=False)

    return jsonify({"results": results_text, "csv_saved": csv_output_path})


# --- Main Route ---
@app.route("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
