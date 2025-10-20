from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import cv2
import base64
import numpy as np
from moviepy.editor import VideoFileClip
from datetime import datetime
from openai import OpenAI
import tempfile

app = Flask(__name__, template_folder="templates")
CORS(app)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Extract a few video frames for AI visual analysis ---
def extract_video_frames(video_path, max_frames=5):
    frames = []
    clip = VideoFileClip(video_path)
    duration = clip.duration
    for t in np.linspace(0, duration, num=max_frames):
        frame = clip.get_frame(t)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode(".jpg", frame_bgr)
        frames.append(base64.b64encode(buffer).decode("utf-8"))
    return frames, duration, clip.size

# --- Helper: Brightness and Tone estimation ---
def analyze_visual_properties(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    brightness_values = []
    for i in range(0, frame_count, max(1, frame_count // 10)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        brightness = hsv[..., 2].mean()
        brightness_values.append(brightness)
    cap.release()
    avg_brightness = float(np.mean(brightness_values))
    tone = "bright" if avg_brightness > 160 else "neutral or mixed" if avg_brightness > 80 else "dark"
    return avg_brightness, tone

# --- Helper: Detect current weekday ---
def get_current_weekday():
    return datetime.now().strftime("%A")

# --- Helper: Get best posting times ---
def get_best_posting_time(niche, platform):
    current_day = get_current_weekday()
    schedule = {
        "TikTok": {
            "Beauty": {"Thu": "4–7 PM", "Fri": "6–9 PM", "Sat": "5–8 PM"},
            "Fitness": {"Mon": "6–8 AM", "Tue": "5–7 PM", "Wed": "8–10 AM"},
            "Food": {"Fri": "12–2 PM", "Sat": "5–8 PM", "Sun": "11 AM–2 PM"},
            "Default": {"Fri": "6–9 PM", "Sat": "6–9 PM", "Sun": "6–9 PM"}
        },
        "Instagram": {
            "Beauty": {"Mon": "6–8 PM", "Wed": "7–9 PM", "Thu": "6–9 PM"},
            "Fitness": {"Tue": "7–9 AM", "Thu": "6–8 PM", "Sat": "8–10 AM"},
            "Food": {"Mon": "12–2 PM", "Wed": "6–8 PM", "Fri": "11 AM–1 PM"},
            "Default": {"Mon": "6–9 PM", "Wed": "6–9 PM", "Fri": "6–9 PM"}
        },
        "YouTube": {
            "Beauty": {"Mon": "3–6 PM", "Wed": "4–7 PM", "Fri": "5–8 PM"},
            "Fitness": {"Mon": "5–8 AM", "Wed": "6–9 PM", "Sat": "8–10 AM"},
            "Food": {"Tue": "11 AM–2 PM", "Thu": "12–3 PM", "Sat": "5–8 PM"},
            "Default": {"Mon": "5–8 PM", "Wed": "5–8 PM", "Fri": "5–8 PM"}
        }
    }
    platform_schedule = schedule.get(platform, schedule["TikTok"])
    niche_schedule = platform_schedule.get(niche, platform_schedule["Default"])
    best_time = niche_schedule.get(current_day[:3], "6–9 PM")
    peak = best_time.split("–")[0]
    return current_day, best_time, peak

# --- AI: Visual + Text Analysis ---
def analyze_video_with_ai(frames, platform, caption=""):
    messages = [
        {
            "role": "system",
            "content": (
                "You are a viral video analysis AI. You receive frames from a video and must identify its topic, niche, and virality factors. "
                "Then you generate captions, hashtags, optimization tips, and short motivational insights. "
                "Finally, generate 3 example viral videos for the same niche and platform with: (1) summary, (2) what made it viral, (3) how to replicate it."
            )
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"Platform: {platform}\nCaption: {caption}\nAnalyze this video visually and describe what it’s about."},
                *[
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}}
                    for img in frames
                ],
            ]
        },
    ]
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=1200,
        temperature=0.8
    )
    return response.choices[0].message.content.strip()

# --- Flask route for analysis ---
@app.route("/analyze", methods=["POST"])
def analyze_video():
    try:
        video = request.files["video"]
        platform = request.form.get("platform", "TikTok")
        caption = request.form.get("caption", "")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            video.save(tmp.name)
            frames, duration, (width, height) = extract_video_frames(tmp.name)
            brightness, tone = analyze_visual_properties(tmp.name)

        ai_response = analyze_video_with_ai(frames, platform, caption)
        niches = ["Beauty", "Fitness", "Food", "Fashion", "Travel", "Gaming", "Comedy", "Education", "Tech"]
        niche = next((n for n in niches if n.lower() in ai_response.lower()), "General")

        heuristic_score = round((brightness / 255) * 10, 1)
        day, best_time, peak = get_best_posting_time(niche, platform)

        result = f"""
🎬 Drag and drop your {platform} video file here: "{video.filename}"

🎥 Running {platform} Viral Optimizer...

🤖 Generating AI-powered analysis, captions, and viral tips...

🔥 Fetching viral video comparisons and strategic insights...

✅ {platform} Video Analysis Complete!

🎬 Video: {video.filename}
📏 Duration: {duration:.2f}s
🖼 Resolution: {width}x{height}
📱 Aspect Ratio: {round(width/height,3)}
💡 Brightness: {round(brightness,2)}
🎨 Tone: {tone}
⭐ Heuristic Score: {heuristic_score}/10 (High brightness and clear resolution contribute to visual appeal.)

💬 AI-Generated Viral Insights:
{ai_response}

🎯 **Detected Niche:** {niche}
🕓 **Best Time to Post for {niche} ({platform}, {day})**:
⏰ {best_time} EST
💡 Peak engagement around {peak} EST.
"""
        return jsonify({"result": result})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/")
def home():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
