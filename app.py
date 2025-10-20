from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import cv2
import numpy as np
from moviepy.editor import VideoFileClip
from openai import OpenAI
import tempfile
import re

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
    prompt = f"""
You are an expert viral strategist for {platform.capitalize()} and short-form content.

Analyze this uploaded video based on its visual traits and generate a detailed viral optimization report.

Video traits:
- Platform: {platform.capitalize()}
- Duration, brightness, and tone: {brightness:.2f} brightness, {tone} tone
- Aspect ratio: typical {platform.capitalize()} format
- Goal: Increase engagement, shares, and retention

Provide results in this exact structure with emojis and markdown:

🎬 **Video Overview**
Briefly describe what kind of content this video likely represents (infer niche).

🎯 **Detected Niche**
Guess the niche (e.g., Beauty, Fitness, Automotive, Education, Food, etc.).

💬 **Scroll-Stopping Caption Idea**
Write one caption.

🏷 **Top 5 Viral Hashtags**
List 5 optimized hashtags for the niche.

🚀 **Actionable Engagement Tip**
One tip to boost audience interaction.

📈 **Viral Optimization Score (0–100)**
Include explanation of strengths and weaknesses.

🔥 **3 Viral Video Examples Related to This Niche**
For each example, provide:
1. **Summary of the viral video**
2. **What made it go viral**
3. **How to replicate it for this uploaded video**

🎯 **Takeaway Strategy**
Concise, motivational next steps.

📋 **Actionable Checklist**
4 clear checklist items.

🕓 **Best Time to Post (EST)**
Give a single recommended window and best engagement time.
"""

    ai_response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a creative strategist who deeply understands viral video psychology, audience retention, and platform algorithms."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.8,
    )

    ai_text = ai_response.choices[0].message.content.strip()

    # --- Remove duplicate or irrelevant post time blocks ---
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
🕓 **Best Time to Post for {detected_niche} ({platform.capitalize()})**:
{time_text}
{peak_text}
"""

    return jsonify({"result": final_output})


if __name__ == "__main__":
    app.run(debug=True)
