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

# --- Extract several representative frames from the video ---
def extract_video_frames(video_path, max_frames=5):
    frames = []
    clip = VideoFileClip(video_path)
    duration = clip.duration
    for t in np.linspace(0, duration, num=max_frames):
        frame = clip.get_frame(t)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode(".jpg", frame_bgr)
        frames.append(base64.b64encode(buffer).decode("utf-8"))
    clip.close()
    return frames, duration, clip.size

# --- Analyze brightness/tone ---
def analyze_visual_properties(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    brightness_vals = []
    for i in range(0, frame_count, max(1, frame_count // 8)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        brightness_vals.append(hsv[..., 2].mean())
    cap.release()
    avg_brightness = float(np.mean(brightness_vals))
    tone = "bright" if avg_brightness > 160 else "neutral or mixed" if avg_brightness > 80 else "dark"
    return avg_brightness, tone

# --- Current weekday ---
def get_current_day():
    return datetime.now().strftime("%A")

# --- AI visual + contextual analysis ---
def analyze_video_with_ai(frames, platform, caption=""):
    messages = [
        {
            "role": "system",
            "content": (
                "You are a top-tier short-form content strategist trained on current TikTok, Instagram, and YouTube Shorts algorithms (2025). "
                "You detect a video's niche, describe its main content visually and contextually, and suggest optimized strategies to go viral."
            )
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"Platform: {platform}\nCaption: {caption}\nAnalyze what this video is about visually, infer its niche, and summarize its theme."},
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
        temperature=0.8,
    )
    return response.choices[0].message.content.strip()

# --- Get viral strategy breakdown ---
def get_viral_strategy(niche, platform, caption=""):
    query = f"""
You are a 2025 short-form content strategist. For a {platform} video in the **{niche}** niche, generate the following:

🎯 Short Summary (1 sentence)
💬 Optimized Caption
🏷️ 5 Viral Hashtags (relevant and high reach)
📈 3 Similar Viral Video Concepts (brief descriptions)
⚙️ Optimization Score (0–100 based on content appeal, clarity, and trend relevance)
🔥 Motivational Line (1 sentence encouraging creator growth)
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Return clean formatted output in the given structure."},
            {"role": "user", "content": query},
        ],
        max_tokens=400,
        temperature=0.8,
    )
    return response.choices[0].message.content.strip()

# --- Dynamic posting time detection ---
def get_dynamic_best_post_time(niche, platform):
    today = datetime.now().strftime("%A")
    query = f"""
Based on current engagement behavior and 2025 algorithm trends, determine the best time window (in EST)
and the peak engagement time for the {niche} niche on {platform} on {today}. Return only the times formatted like this:

🕓 **Best Time to Post for {niche} ({platform}, {today})**:
⏰ [Time Window]
💡 Peak engagement around [Time].
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert in algorithmic posting optimization."},
            {"role": "user", "content": query},
        ],
        max_tokens=200,
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()

# --- Flask route ---
@app.route("/analyze", methods=["POST"])
def analyze_video():
    try:
        video = request.files["video"]
        platform = request.form.get("platform", "TikTok").capitalize()
        caption = request.form.get("caption", "")

        # --- Save temp file ---
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            video.save(tmp.name)
            frames, duration, (width, height) = extract_video_frames(tmp.name)
            brightness, tone = analyze_visual_properties(tmp.name)

        # --- AI analysis ---
        ai_analysis = analyze_video_with_ai(frames, platform, caption)

        # --- Detect niche dynamically ---
        possible_niches = ["Beauty", "Fitness", "Food", "Fashion", "Travel", "Gaming", "Comedy", "Education", "Tech", "Lifestyle", "Music", "Sports", "Business", "Motivation", "Pets", "Finance", "DIY"]
        detected_niche = next((n for n in possible_niches if n.lower() in ai_analysis.lower()), "General")

        # --- Dynamic times + viral breakdown ---
        best_time_block = get_dynamic_best_post_time(detected_niche, platform)
        viral_strategy = get_viral_strategy(detected_niche, platform, caption)

        # --- Heuristic scoring ---
        heuristic_score = round((brightness / 255) * 10, 1)

        # --- Final formatted output ---
        result = f"""
🎬 Drag and drop your {platform} video file here: "{video.filename}"

🎥 Running {platform} Viral Optimizer...

🤖 Generating AI-powered analysis, captions, and viral insights...

✅ {platform} Video Analysis Complete!

🎬 Video: {video.filename}
📏 Duration: {duration:.2f}s
🖼 Resolution: {width}x{height}
📱 Aspect Ratio: {round(width/height,3)}
💡 Brightness: {round(brightness,2)}
🎨 Tone: {tone}
⭐ Visual Quality Score: {heuristic_score}/10

🎯 **Detected Niche:** {detected_niche}

💬 **AI-Generated Analysis:**
{ai_analysis}

{best_time_block}

🚀 **Viral Strategy Breakdown:**
{viral_strategy}
"""
        return jsonify({"result": result})

    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/")
def home():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
