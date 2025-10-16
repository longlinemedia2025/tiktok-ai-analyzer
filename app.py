from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import cv2
import numpy as np
from moviepy.editor import VideoFileClip
from openai import OpenAI
import tempfile
import pandas as pd

app = Flask(__name__, template_folder="templates")
CORS(app)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Helper: Analyze video properties ---
def analyze_video_properties(video_path):
    clip = VideoFileClip(video_path)
    duration = clip.duration
    width, height = clip.size
    aspect_ratio = round(width / height, 3)

    # Use OpenCV to analyze brightness
    cap = cv2.VideoCapture(video_path)
    brightness_values = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        brightness = hsv[..., 2].mean()
        brightness_values.append(brightness)
    cap.release()

    avg_brightness = np.mean(brightness_values) if brightness_values else 0
    tone = "bright" if avg_brightness > 160 else "neutral or mixed" if avg_brightness > 90 else "dark"

    return {
        "duration": round(duration, 2),
        "resolution": f"{width}x{height}",
        "aspect_ratio": aspect_ratio,
        "brightness": round(avg_brightness, 2),
        "tone": tone,
    }

# --- Helper: Get AI response ---
def get_ai_analysis(prompt):
    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )
    return response.output[0].content[0].text

# --- Route: Home ---
@app.route('/')
def home():
    return render_template('index.html')

# --- Route: Analyze TikTok Video ---
@app.route('/analyze_tiktok', methods=['POST'])
def analyze_tiktok():
    video_file = request.files.get('video')
    csv_file = request.files.get('csv')

    if not video_file:
        return jsonify({'error': 'No video file uploaded'}), 400

    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
        video_file.save(tmp.name)
        video_path = tmp.name

    video_data = analyze_video_properties(video_path)

    # Load CSV context if uploaded
    csv_context = ""
    if csv_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_csv:
            csv_file.save(tmp_csv.name)
            df = pd.read_csv(tmp_csv.name)
            csv_context = df.to_string(index=False)

    prompt = f"""
You are a TikTok virality analyzer AI.
Analyze this video data and CSV info to generate optimized insights.

Video Info:
Duration: {video_data['duration']}s
Resolution: {video_data['resolution']}
Aspect Ratio: {video_data['aspect_ratio']}
Brightness: {video_data['brightness']}
Tone: {video_data['tone']}

CSV Insights:
{csv_context[:2000]}

Generate the following output in this format:

🎬 Video Analysis Summary
💡 Viral Caption
🔥 5 Hashtags
🎯 Optimization Score (1-100)
💬 Improvement Tip
📊 Keyword Suggestions (for SEO + hashtags)
📈 Viral Comparison & Strategy
📋 Actionable Checklist
    """

    result = get_ai_analysis(prompt)
    os.remove(video_path)
    return jsonify({'results': result})

# --- Route: Analyze YouTube Video ---
@app.route('/analyze_youtube', methods=['POST'])
def analyze_youtube():
    video_file = request.files.get('video')
    csv_file = request.files.get('csv')

    if not video_file:
        return jsonify({'error': 'No video file uploaded'}), 400

    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
        video_file.save(tmp.name)
        video_path = tmp.name

    video_data = analyze_video_properties(video_path)

    csv_context = ""
    if csv_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_csv:
            csv_file.save(tmp_csv.name)
            df = pd.read_csv(tmp_csv.name)
            csv_context = df.to_string(index=False)

    prompt = f"""
You are a YouTube content strategy AI.
Analyze this uploaded YouTube video data and the optional CSV dataset.
Provide a full breakdown following YouTube's algorithm principles (CTR, AVD, engagement, SEO).

Video Info:
Duration: {video_data['duration']}s
Resolution: {video_data['resolution']}
Aspect Ratio: {video_data['aspect_ratio']}
Brightness: {video_data['brightness']}
Tone: {video_data['tone']}

CSV Insights:
{csv_context[:2000]}

Generate the following:
🎬 Video Overview
🔑 Top 5 YouTube Keyword Suggestions (SEO)
💬 Engaging Video Title Suggestion
🔥 5 Hashtags for YouTube Shorts/Community
📊 Algorithm Optimization Score (1–100)
⚙️ Retention & Engagement Improvement Tip
📈 Comparison with Top Viral Videos (same niche)
🎯 Strategic Takeaways for Growth
📋 Actionable Checklist (3–5 quick steps)
    """

    result = get_ai_analysis(prompt)
    os.remove(video_path)
    return jsonify({'results': result})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
