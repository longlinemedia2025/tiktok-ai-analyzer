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

# Mapping of platform + niche → best time window & peak (EST)
BEST_TIME_MAP = {
    "tiktok": {
        "beauty": ("Thu 7–9 PM EST", "Peak ~8 PM EST"),
        "fitness": ("Mon 6–8 AM EST", "Peak ~7 AM EST"),
        "gaming": ("Wed 4–6 PM EST", "Peak ~5 PM EST"),
        "general": ("Thu 7–11 AM EST", "Peak ~9 AM EST"),
    },
    "instagram": {
        "beauty": ("Wed 6–8 PM EST", "Peak ~7 PM EST"),
        "lifestyle": ("Fri 5–7 PM EST", "Peak ~6 PM EST"),
        "food": ("Tue 11 AM–1 PM EST", "Peak ~12 PM EST"),
        "general": ("Wed 10 AM–1 PM EST", "Peak ~11 AM EST"),
    },
    "youtube": {
        "beauty": ("Sat 2–4 PM EST", "Peak ~3 PM EST"),
        "education": ("Thu 3–5 PM EST", "Peak ~4 PM EST"),
        "gaming": ("Fri 5–7 PM EST", "Peak ~6 PM EST"),
        "general": ("Fri 2–4 PM EST", "Peak ~3 PM EST"),
    },
    "facebook": {
        "community": ("Sun 9–11 AM EST", "Peak ~10 AM EST"),
        "lifestyle": ("Mon 8–10 AM EST", "Peak ~9 AM EST"),
        "news": ("Tue 7–9 AM EST", "Peak ~8 AM EST"),
        "general": ("Tue 9–11 AM EST", "Peak ~10 AM EST"),
    }
}

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

    # --- Infer niche from filename or simple keywords ---
    filename_hint = os.path.splitext(video.filename)[0] if video else ""
    hint_lower = filename_hint.lower()
    if any(k in hint_lower for k in ["barber", "haircut", "salon", "fade", "barbershop"]):
        detected_niche = "beauty"
    elif any(k in hint_lower for k in ["gym", "workout", "fitness", "training"]):
        detected_niche = "fitness"
    elif any(k in hint_lower for k in ["game", "gaming", "playthrough", "fps"]):
        detected_niche = "gaming"
    else:
        detected_niche = "general"

    # Determine best post time & peak from mapping
    platform_map = BEST_TIME_MAP.get(platform, {})
    niche_map = platform_map.get(detected_niche, platform_map.get("general", ("TBD", "TBD")))
    best_window, best_peak = niche_map

    # --- Smart AI prompt for full viral breakdown ---
    weekday = datetime.now().strftime("%A")

    prompt = f"""
You are an expert viral strategist and short-form content analyst.

Analyze this uploaded video and generate a detailed, platform-specific viral optimization report.
Use visual traits + filename context to infer the correct niche and creative direction.

Video metadata:
- Platform: {platform.capitalize()}
- Filename: "{filename_hint}"
- Duration: {duration:.2f}s
- Brightness: {brightness:.2f}
- Tone: {tone}
- Aspect ratio approx: {width}×{height}

Provide results in this exact structure:

🎬 **Video Overview**
Briefly describe what this video is likely about (infer from filename and tone).

🎯 **Detected Niche**
Choose the most accurate niche (Beauty, Fitness, Gaming, Automotive, etc.).

💬 **Scroll-Stopping Caption Idea**
Write one engaging caption ideal for {platform.capitalize()}.

🏷 **Top 5 Viral Hashtags**
List 5 hashtags that fit this niche.

🚀 **Actionable Engagement Tip**
One practical tip to boost interaction.

📈 **Viral Optimization Score (0-100)**
Explain strengths & where to improve.

🔥 **3 Viral Video Examples Related to This Niche**
For each example:
1. **Summary of the viral video**
2. **What made it go viral**
3. **How to replicate it for this uploaded video**

🎯 **Takeaway Strategy**
Short motivational next‐step.

📋 **Actionable Checklist**
4 clear checklist items.

🕓 **Best Time to Post ({weekday}, EST)**
⏰ {best_window}
💡 Peak engagement around {best_peak}
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

    # --- Clean up duplicate post-time blocks if any ---
    ai_text = re.sub(r"🕓\s*\*\*Best Time to Post.*?(?:\n💡.*)?", "", ai_text, flags=re.DOTALL)
    ai_text = re.sub(r"\n{3,}", "\n\n", ai_text).strip()

    # --- Final formatted output ---
    final_output = f"""
🎬 Drag and drop your {platform.capitalize()} video file here: "{video.filename if video else 'N/A'}"
🎥 Running {platform.capitalize()} Viral Optimizer...

🤖 Generating AI-powered analysis, captions, and viral tips...

🔥 Fetching 3 relevant viral video examples for the same niche...

✅ {platform.capitalize()} Video Analysis Complete!

🎬 Video: {video.filename if video else 'N/A'}
{video_info}

{ai_text}

🎯 **Detected Niche:** {detected_niche.capitalize()}
🕓 **Best Time to Post for {detected_niche.capitalize()} ({platform.capitalize()}, {weekday})**:
⏰ {best_window}
💡 Peak engagement around {best_peak}
"""

    return jsonify({"result": final_output})


if __name__ == "__main__":
    app.run(debug=True)
