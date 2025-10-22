from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import cv2
import tempfile
import numpy as np
import datetime
from moviepy.editor import VideoFileClip
from openai import OpenAI
import base64
import re

app = Flask(__name__, template_folder="templates")
CORS(app)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# --- Helper: lightweight frame sampling for visual content ---
def analyze_video_visuals(video_path, sample_frames=6):
    """
    Sample frames (up to sample_frames) from the video, compute average color/brightness,
    and return brightness, tone and a short base64 list of frames (first 3).
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames <= 0:
        cap.release()
        return 127.0, "neutral or mixed", []

    # pick up to `sample_frames` evenly spaced frames across the video
    indices = np.linspace(0, total_frames - 1, min(sample_frames, total_frames), dtype=int)
    avg_colors = []
    frames_b64 = []

    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        # compute average color and brightness (resize small to speed)
        small = cv2.resize(frame, (64, 64))
        mean_bgr = cv2.mean(small)[:3]  # B, G, R
        avg_colors.append(mean_bgr)

        # encode a small jpg base64 for the model (limit to first 3 frames)
        if len(frames_b64) < 3:
            _, buf = cv2.imencode(".jpg", cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
            frames_b64.append(base64.b64encode(buf).decode("ascii"))

    cap.release()

    if avg_colors:
        avg_col = np.mean(avg_colors, axis=0)  # BGR mean
        # Convert BGR average to approximate brightness
        brightness = float(np.mean(avg_col))
    else:
        brightness = 127.0

    if brightness < 60:
        tone = "dark and moody"
    elif brightness < 130:
        tone = "neutral or mixed"
    else:
        tone = "bright and energetic"

    return round(brightness, 2), tone, frames_b64


# --- Helper: determine best posting time ---
def best_posting_time(platform, niche):
    """
    Return current weekday (full name), a recommended posting window and peak time
    for the given platform and niche from internal defaults.
    """
    today = datetime.datetime.now()
    day_name = today.strftime("%A")

    defaults = {
        "TikTok": {
            "Beauty": ("Thu 4–7 PM", "8:43 PM"),
            "Gaming": ("Fri 5–8 PM", "7:15 PM"),
            "Fitness": ("Mon 6–9 PM", "8:00 PM"),
            "Default": ("Wed 6–9 PM", "7:30 PM"),
        },
        "Instagram": {
            "Beauty": ("Sun 5–8 PM", "6:45 PM"),
            "Gaming": ("Mon 6–9 PM", "8:10 PM"),
            "Fitness": ("Tue 5–8 PM", "7:00 PM"),
            "Default": ("Wed 6–9 PM", "7:30 PM"),
        },
        "YouTube": {
            "Beauty": ("Sat 2–6 PM", "3:45 PM"),
            "Gaming": ("Fri 6–10 PM", "8:00 PM"),
            "Fitness": ("Sun 8–11 AM", "9:30 AM"),
            "Default": ("Thu 6–9 PM", "8:00 PM"),
        },
        "Facebook": {
            "Beauty": ("Fri 4–7 PM", "5:45 PM"),
            "Gaming": ("Thu 6–8 PM", "7:00 PM"),
            "Fitness": ("Wed 5–8 PM", "7:30 PM"),
            "Default": ("Tue 6–8 PM", "7:00 PM"),
        },
    }

    # normalize key
    platform_key = platform if platform in defaults else platform.capitalize()
    platform_defaults = defaults.get(platform_key, defaults.get("TikTok"))
    time_range, peak = platform_defaults.get(niche, platform_defaults.get("Default"))
    return day_name, time_range, peak


# --- Core routes ---
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        platform = request.form.get("platform", "TikTok")
        # Normalize platform (some clients send lowercase)
        platform = platform if platform in ["TikTok", "YouTube", "Instagram", "Facebook"] else platform.capitalize()

        video_file = request.files.get("video")
        csv_file = request.files.get("csv")

        if not video_file:
            return jsonify({"error": "No video file provided."})

        # --- Save uploaded video temporarily ---
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            video_path = temp_video.name
            video_file.save(video_path)
            original_filename = video_file.filename or "uploaded_video.mp4"

        # --- If CSV was provided, just save it (we won't change index.html) ---
        csv_summary = ""
        if csv_file:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_csv:
                    csv_path = temp_csv.name
                    csv_file.save(csv_path)
                csv_summary = f"CSV uploaded: {csv_file.filename}"
            except Exception as e:
                csv_summary = f"CSV error: {e}"

        # --- Extract metadata via moviepy ---
        try:
            clip = VideoFileClip(video_path)
            duration = round(clip.duration, 2)
            fps = clip.fps or 30
            width, height = clip.size
            clip.close()
        except Exception as e:
            return jsonify({"error": f"Video processing failed: {str(e)}"}), 500

        # --- Visual analysis (frames, brightness, tone) ---
        brightness, tone, frames_b64 = analyze_video_visuals(video_path, sample_frames=8)

        # --- Build text prompt for the model (explicitly request visual-first analysis) ---
        # We include a short base64 list (max 3) and a compact visual descriptor so the model
        # understands the visuals are the primary data source (not just filename).
        frame_count_note = len(frames_b64)
        frames_display = frames_b64[:3]  # small set for prompt
        visual_descriptor = (
            f"brightness={brightness}, tone='{tone}', aspect_ratio={round(width/height,3)}, "
            f"sample_frames={frame_count_note}"
        )

        prompt = f"""
You are a professional social media content analyst and viral strategist.

CRITICAL: Prioritize the visual content information (brightness, tone, sampled frames) and metadata.
Do NOT rely solely on the filename or uploaded name. Use the provided frames and visual descriptors first to determine niche and optimization advice.

Video metadata:
- Platform: {platform}
- Filename (for reference only): "{original_filename}"
- Duration: {duration}s
- Resolution: {width}x{height}
- Framerate: {round(fps)}
- Visual descriptor: {visual_descriptor}
- CSV context: {csv_summary}

Included frame samples (base64 JPEG strings). These are provided so you can reason about visual content:
{frames_display}

Tasks (respond using the exact structure shown below, tailored to {platform}):

### 🎬 Video Summary
📏 Duration: {duration}s | {width}x{height}px | {round(fps)}fps  
💡 Visual Tone: {tone} | Brightness: {brightness}

### 💬 AI-Generated Viral Insights:
1️⃣ **Scroll-Stopping Caption ({platform} only)**  
(Provide one caption optimized specifically for the requested platform only.)

2️⃣ **5 Hashtags ({platform} only)**  
(Provide 5 hashtags ideally suited to this platform and detected niche.)

3️⃣ **Engagement Tip**  
(One concise actionable tip.)

4️⃣ **Viral Optimization Score (1–100)**  
(Provide a numeric score and one-line rationale.)

5️⃣ **Motivational Tip**  
(Brief encouraging line.)

### 🔥 Viral Comparison:
Provide 3 real public viral videos from {platform} within the detected niche. For each include:
- Video title or short URL
- What made it go viral (concrete signals: audio trend, edit style, hook, challenge, timeframe)
- How to replicate that success with this uploaded video

### 🧠 Optimization Advice:
List 3–5 specific improvements (editing, hook, audio, text overlay, thumbnail/first frame) that will increase ranking on {platform},
explicitly comparing to the viral examples you listed.

Finally, provide:
🎯 **Detected Niche:** [exact niche]
🕓 **Best Time to Post for [niche] ({platform}, today)**:
⏰ [time range]
💡 Peak engagement around [time].
"""

        # --- Call the model (text-only content; frames provided as base64 strings in the prompt) ---
        ai_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert in social media optimization and video analysis."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.85,
            max_tokens=1800,
        )

        # --- Extract AI text safely ---
        ai_text = ""
        try:
            ai_text = ai_response.choices[0].message.content.strip()
        except Exception:
            ai_text = ""

        if not ai_text:
            return jsonify({"error": "⚠ No results received from AI."}), 500

        # --- Clean up duplicated final sections if the model repeated them ---
        # remove consecutive duplicate "🎯 **Detected Niche:**" blocks (keep last)
        # and collapse excessive blank lines
        ai_text = re.sub(r"\n{3,}", "\n\n", ai_text).strip()
        # If model appended multiple Detected Niche blocks, keep the last one by splitting
        if ai_text.count("🎯 **Detected Niche:**") > 1:
            parts = ai_text.split("🎯 **Detected Niche:**")
            # keep everything before first occurance and the last block, rejoin cleanly
            ai_text = parts[0].strip() + "\n\n🎯 **Detected Niche:**" + parts[-1].strip()

        # --- Try to extract niche from AI text; fallback to keyword scan ---
        niche = None
        m = re.search(r"🎯 \*\*Detected Niche:\*\*\s*(.+)", ai_text)
        if m:
            niche = m.group(1).strip().splitlines()[0]
        else:
            # fallback keyword scan
            for possible in ["Beauty", "Gaming", "Fitness", "Food", "Travel", "Education", "Music", "Fashion", "Sports", "Comedy", "Photography", "Lifestyle"]:
                if possible.lower() in ai_text.lower():
                    niche = possible
                    break
        if not niche:
            niche = "General"

        # --- Best posting time ---
        day_name, time_range, peak_time = best_posting_time(platform, niche)

        # --- Append a single final footer with correct day/time (no duplicates) ---
        final_output = f"""{ai_text}

🎯 **Detected Niche:** {niche}
🕓 **Best Time to Post for {niche} ({platform}, {day_name})**:
⏰ {time_range}
💡 Peak engagement around {peak_time}.
"""

        return jsonify({"results": final_output})

    except Exception as e:
        # surface error for debugging (returns HTTP 500)
        print("Error in /analyze:", str(e))
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
