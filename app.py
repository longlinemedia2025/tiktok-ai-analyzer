from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import cv2
import numpy as np
from moviepy.editor import VideoFileClip
from openai import OpenAI
import tempfile
import pandas as pd
import re
from datetime import datetime

app = Flask(__name__, template_folder="templates")
CORS(app)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def analyze_video_features(video_path):
    """Extracts key video metrics like brightness, duration, and resolution."""
    clip = VideoFileClip(video_path)
    duration = clip.duration
    frame = clip.get_frame(0)
    height, width, _ = frame.shape
    brightness = np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 2])
    aspect_ratio = round(width / height, 3)
    return {
        "duration": round(duration, 2),
        "resolution": f"{width}x{height}",
        "aspect_ratio": aspect_ratio,
        "brightness": round(brightness, 2),
    }

def detect_niche_tone_keywords(video_name, csv_data=None):
    """Uses AI to auto-detect niche, tone, and keywords."""
    prompt = f"""
    Analyze the following TikTok or YouTube video name and CSV (if provided) 
    to detect its niche, tone, and most relevant keywords.
    
    Video: {video_name}
    CSV sample data: {csv_data if csv_data else 'No CSV provided'}
    
    Return a JSON with keys: niche, tone, keywords.
    """

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )

    text = response.output[0].content[0].text
    match = re.findall(r'"?(\w+)"?:\s*"([^"]+)"', text)
    result = {k: v for k, v in match}
    return {
        "niche": result.get("niche", "General"),
        "tone": result.get("tone", "Neutral"),
        "keywords": result.get("keywords", "None detected")
    }

def generate_ai_analysis(platform, video_name, metrics, detected, csv_data=None):
    """Generates the AI-enhanced viral analysis with your exact format."""
    duration = metrics["duration"]
    resolution = metrics["resolution"]
    aspect_ratio = metrics["aspect_ratio"]
    brightness = metrics["brightness"]

    niche = detected["niche"]
    tone = detected["tone"]
    keywords = detected["keywords"]

    prompt = f"""
    Create a viral video analysis for {platform} with this structure EXACTLY:

    🎬 Drag and drop your {platform} video file here: "{video_name}"
    🎥 Running {platform} Viral Optimizer...

    🤖 Generating AI-powered analysis, captions, and viral tips...

    🔥 Fetching viral video comparisons and strategic insights...

    ✅ {platform} Video Analysis Complete!

    🎬 Video: {video_name}
    📏 Duration: {duration}s
    🖼 Resolution: {resolution}
    📱 Aspect Ratio: {aspect_ratio}
    💡 Brightness: {brightness}
    🎨 Tone: {tone}
    ⭐ Heuristic Score: 8/10 (High brightness and clear resolution contribute to visual appeal.)

    🎯 Detected Attributes:
    - Niche: {niche}
    - Tone: {tone}
    - Keywords: {keywords}

    💬 AI-Generated Viral Insights:
    ### 1. Scroll-Stopping Caption
    (Give one creative, high-performing caption idea.)

    ### 2. 5 Viral {"Hashtags" if platform == "TikTok" else "Tags"}
    (Provide 5 platform-relevant tags.)

    ### 3. Actionable Improvement Tip for Engagement
    (Provide a short suggestion for improving engagement.)

    ### 4. Viral Optimization Score (1–100)
    (Give a score with an explanation for why it earned that score.)

    ### 5. Short Motivation on How to Increase Virality
    (Add an encouraging tip for the creator.)

    🔥 Viral Comparison Results:
    ### Comparison with Viral {platform}s in the Same Niche
    #### Viral Example 1
    - **Video Concept Summary:** ...
    - **What Made It Go Viral:** ...
    - **How to Replicate Success:** ...

    #### Viral Example 2
    - **Video Concept Summary:** ...
    - **What Made It Go Viral:** ...
    - **How to Replicate Success:** ...

    #### Viral Example 3
    - **Video Concept Summary:** ...
    - **What Made It Go Viral:** ...
    - **How to Replicate Success:** ...

    ### Takeaway Strategy
    (Summarize actionable insights.)

    📋 Actionable Checklist:
    - Hook viewers in under 2 seconds.
    - Add trending sound if relevant.
    - Post during high activity times (Fri–Sun, 6–10pm).
    - Encourage comments by asking a question.

    🎯 **Detected Niche:** {niche}
    🕓 **Best Time to Post for {niche} (Thu)**:
    ⏰ 4–7 PM EST
    💡 Peak engagement around 8:43 PM EST.
    """

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )

    return response.output[0].content[0].text.strip()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    platform = request.form.get("platform", "TikTok")
    video = request.files.get("video")
    csv_file = request.files.get("csv")

    if not video:
        return jsonify({"error": "No video uploaded."}), 400

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        video.save(tmp.name)
        metrics = analyze_video_features(tmp.name)

    csv_data = None
    if csv_file:
        df = pd.read_csv(csv_file)
        csv_data = df.head(5).to_dict()

    detected = detect_niche_tone_keywords(video.filename, csv_data)
    result_text = generate_ai_analysis(platform, video.filename, metrics, detected, csv_data)

    return jsonify({"result": result_text})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
