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


# --- Helper: Extract a few frames from the video for visual analysis ---
def extract_video_frames(video_path, max_frames=5):
    frames = []
    clip = VideoFileClip(video_path)
    duration = clip.duration
    for t in np.linspace(0, duration, num=max_frames):
        frame = clip.get_frame(t)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode(".jpg", frame_bgr)
        frames.append(base64.b64encode(buffer).decode("utf-8"))
    return frames


# --- Helper: Get current weekday ---
def get_current_weekday():
    return datetime.now().strftime("%A")


# --- AI: Analyze video content visually + via caption ---
def analyze_video_with_ai(frames, platform, caption=""):
    # Create messages for multimodal AI input
    messages = [
        {
            "role": "system",
            "content": (
                "You are a viral content analyst AI. You will receive video frames and a caption. "
                "Describe what the video is about, identify the main niche (like Beauty, Fitness, Food, Fashion, Travel, Gaming, Comedy, etc.), "
                "and explain why it might perform well on social media."
            ),
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"Platform: {platform}\nCaption: {caption}"},
                *[
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}}
                    for img in frames
                ],
            ],
        },
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini", messages=messages, max_tokens=500
    )

    return response.choices[0].message.content.strip()


# --- Helper: Predict best post time by niche, platform, and weekday ---
def get_best_posting_time(niche, platform):
    current_day = get_current_weekday()

    schedule_data = {
        "TikTok": {
            "Beauty": {"Monday": "6–9 PM", "Tuesday": "5–8 PM", "Wednesday": "7–10 PM"},
            "Fitness": {"Monday": "6–8 AM", "Tuesday": "5–7 PM", "Wednesday": "8–10 AM"},
            "Food": {"Monday": "11 AM–1 PM", "Tuesday": "12–2 PM", "Wednesday": "6–9 PM"},
            "Default": {"Monday": "6–9 PM", "Tuesday": "6–9 PM", "Wednesday": "6–9 PM"},
        },
        "Instagram": {
            "Beauty": {"Monday": "6–8 PM", "Tuesday": "7–9 PM", "Wednesday": "6–9 PM"},
            "Fitness": {"Monday": "7–9 AM", "Tuesday": "6–8 PM", "Wednesday": "8–10 AM"},
            "Food": {"Monday": "12–2 PM", "Tuesday": "6–8 PM", "Wednesday": "11 AM–1 PM"},
            "Default": {"Monday": "6–9 PM", "Tuesday": "6–9 PM", "Wednesday": "6–9 PM"},
        },
        "YouTube": {
            "Beauty": {"Monday": "3–6 PM", "Tuesday": "4–7 PM", "Wednesday": "5–8 PM"},
            "Fitness": {"Monday": "5–8 AM", "Tuesday": "6–9 PM", "Wednesday": "8–10 AM"},
            "Food": {"Monday": "11 AM–2 PM", "Tuesday": "12–3 PM", "Wednesday": "5–8 PM"},
            "Default": {"Monday": "5–8 PM", "Tuesday": "5–8 PM", "Wednesday": "5–8 PM"},
        },
    }

    niche_times = schedule_data.get(platform, {}).get(niche, schedule_data.get(platform, {}).get("Default", {}))
    best_time = niche_times.get(current_day, "6–9 PM")
    peak_hour = best_time.split("–")[0]

    return {
        "day": current_day,
        "best_time": best_time,
        "peak": peak_hour,
    }


# --- Flask Routes ---
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze_video():
    try:
        video_file = request.files["video"]
        platform = request.form.get("platform", "TikTok")
        caption = request.form.get("caption", "")

        # Save video temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            video_file.save(tmp.name)
            frames = extract_video_frames(tmp.name)

        # AI Video Analysis
        ai_analysis = analyze_video_with_ai(frames, platform, caption)

        # Detect niche keyword
        possible_niches = ["Beauty", "Fitness", "Food", "Fashion", "Travel", "Gaming", "Comedy"]
        niche = next((n for n in possible_niches if n.lower() in ai_analysis.lower()), "General")

        # Get posting time prediction
        timing = get_best_posting_time(niche, platform)

        # Format result exactly like before
        result_text = (
            f"🎯 **Detected Niche:** {niche}\n"
            f"🧠 **Video Summary:** {ai_analysis}\n"
            f"📱 **Platform:** {platform}\n\n"
            f"📆 ({timing['day']}, EST)\n"
            f"⏰ **Best Time to Post:** {timing['best_time']} EST\n"
            f"💡 Peak engagement around ~{timing['peak']} EST"
        )

        return jsonify({"result": result_text})

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
