from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import cv2
import numpy as np
from moviepy.editor import VideoFileClip
from openai import OpenAI
import tempfile
import re
from datetime import datetime

app = Flask(__name__, template_folder="templates")
CORS(app)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    platform = request.form.get("platform", "tiktok").lower()
    video = request.files.get("video")

    video_path = None
    if video:
        temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        video.save(temp_video.name)
        video_path = temp_video.name

    # --- Extract video metadata ---
    video_info = ""
    brightness = 0
    tone = "neutral or mixed"
    duration = 0
    width = 0
    height = 0

    if video_path:
        try:
            clip = VideoFileClip(video_path)
            duration = clip.duration
            frame = clip.get_frame(0)
            height, width, _ = frame.shape
            aspect_ratio_val = width / height
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            tone = "bright" if brightness > 130 else "dark" if brightness < 60 else "neutral or mixed"
            video_info = (
                f"📏 Duration: {duration:.2f}s\n"
                f"🖼 Resolution: {width}x{height}\n"
                f"📱 Aspect Ratio: {aspect_ratio_val:.3f}\n"
                f"💡 Brightness: {brightness:.2f}\n"
                f"🎨 Tone: {tone}"
            )
            clip.close()
        except Exception as e:
            video_info = f"Error analyzing video: {e}"

    # --- Smart AI prompt for full viral breakdown ---
    filename_hint = os.path.splitext(video.filename)[0] if video else "Unknown Video"
    weekday = datetime.now().strftime("%A")

    prompt = f"""
You are an expert viral strategist and short-form content analyst.

Analyze this uploaded video and generate a detailed, platform-specific viral optimization report.
Use both the visual traits and filename context to infer the correct niche and creative direction.

Video metadata:
- Platform: {platform.capitalize()}
- Filename: "{filename_hint}"
- Duration: {duration:.2f}s
- Brightness: {brightness:.2f}
- Tone: {tone}
- Aspect ratio: {width}x{height}

Provide a full breakdown in **this exact structure with emojis and markdown**:

🎬 **Video Overview**
Briefly describe what this video is likely about (infer from filename and tone).

🎯 **Detected Niche**
Choose the most accurate niche (e.g., Beauty, Barbering, Fitness, Gaming, Automotive, Education, Food, etc.).

💬 **Scroll-Stopping Caption Idea**
Write a short, engaging caption tailored to {platform.capitalize()} audience style.

🏷 **Top 5 Viral Hashtags**
List 5 hashtags that fit this video's niche.

🚀 **Actionable Engagement Tip**
Give one practical tip to increase audience interaction and watch time.

📈 **Viral Optimization Score (0–100)**
Explain the score, including strengths and improvement areas.

🔥 **3 Viral Video Examples Related to This Niche**
For each example, provide:
1. **Summary of the viral video**
2. **What made it go viral**
3. **How to replicate it for this uploaded video**

🎯 **Takeaway Strategy**
Provide a short, motivational next step strategy.

📋 **Actionable Checklist**
Give 4 specific checklist items for the creator to follow.

🕓 **Best Time to Post ({weekday}, EST)**
Include a single time window (e.g., 6–9 PM EST) and note when engagement typically peaks.
"""

    ai_response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a creative strategist who deeply understands viral video psychology, platform algorithms, and engagement optimization."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.8,
    )

    ai_text = ai_response.choices[0].message.content.strip()

    # --- Clean up duplicate post-time blocks ---
    ai_text = re.sub(r"🕓\s*\*\*Best Time to Post.*?(?:\n💡.*)?", "", ai_text, flags=re.DOTALL)
    ai_text = re.sub(r"\n{3,}", "\n\n", ai_text).strip()

    # --- Extract niche dynamically ---
    niche_match = re.search(r"(?i)\*\*Detected Niche\*\*[:\-–]?\s*(.*)", ai_text)
    detected_niche = niche_match.group(1).strip() if niche_match else "General"

    # --- Extract best time section if available ---
    time_match = re.search(r"(⏰ .*?EST[^\n]*)", ai_text)
    time_text = time_match.group(1) if time_match else "⏰ 6–10 PM EST"
    peak_match = re.search(r"(💡 .*?EST[^\n]*)", ai_text)
    peak_text = peak_match.group(1) if peak_match else "💡 Peak engagement around 8 PM EST."

    # --- Final report formatting ---
    final_output = f"""
🎬 Drag and drop your {platform.capitalize()} video file here: "{video.filename if video else 'N/A'}"
🎥 Running {platform.capitalize()} Viral Optimizer...

🤖 Generating AI-powered analysis, captions, and viral tips...

🔥 Fetching 3 relevant viral video examples for the same niche...

✅ {platform.capitalize()} Video Analysis Complete!

🎬 Video: {video.filename if video else 'N/A'}
{video_info}

{ai_text}

🎯 **Detected Niche:** {detected_niche}
🕓 **Best Time to Post for {detected_niche} ({platform.capitalize()}, {weekday})**:
{time_text}
{peak_text}
"""

    return jsonify({"result": final_output})


if __name__ == "__main__":
    app.run(debug=True)
