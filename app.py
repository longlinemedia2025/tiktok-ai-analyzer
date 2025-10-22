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


def analyze_video_visuals(video_path, sample_frames=6):
    """
    Sample frames from the video, compute average color/brightness,
    and return brightness, tone and up to 3 base64-encoded frame thumbnails.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames <= 0:
        cap.release()
        return 127.0, "neutral or mixed", []

    indices = np.linspace(0, total_frames - 1, min(sample_frames, total_frames), dtype=int)
    avg_colors = []
    frames_b64 = []

    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        small = cv2.resize(frame, (64, 64))
        mean_bgr = cv2.mean(small)[:3]
        avg_colors.append(mean_bgr)

        if len(frames_b64) < 3:
            # convert to RGB jpg bytes then base64
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            _, buf = cv2.imencode(".jpg", rgb, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
            frames_b64.append(base64.b64encode(buf).decode("ascii"))

    cap.release()

    brightness = float(np.mean(np.array(avg_colors))) if avg_colors else 127.0

    if brightness < 60:
        tone = "dark and moody"
    elif brightness < 130:
        tone = "neutral or mixed"
    else:
        tone = "bright and energetic"

    return round(brightness, 2), tone, frames_b64


def best_posting_time(platform, niche):
    """
    Return the current weekday name and recommended posting window & peak for platform+niche.
    """
    now = datetime.datetime.now()
    day_name = now.strftime("%A")

    defaults = {
        "TikTok": {
            "Beauty": ("Thu 4–7 PM EST", "8:43 PM EST"),
            "Gaming": ("Fri 5–8 PM EST", "7:15 PM EST"),
            "Fitness": ("Mon 6–9 PM EST", "8:00 PM EST"),
            "Default": ("Wed 6–9 PM EST", "7:30 PM EST"),
        },
        "Instagram": {
            "Beauty": ("Sun 5–8 PM EST", "6:45 PM EST"),
            "Gaming": ("Mon 6–9 PM EST", "8:10 PM EST"),
            "Fitness": ("Tue 5–8 PM EST", "7:00 PM EST"),
            "Default": ("Wed 6–9 PM EST", "7:30 PM EST"),
        },
        "YouTube": {
            "Beauty": ("Sat 2–6 PM EST", "3:45 PM EST"),
            "Gaming": ("Fri 6–10 PM EST", "8:00 PM EST"),
            "Fitness": ("Sun 8–11 AM EST", "9:30 AM EST"),
            "Default": ("Thu 6–9 PM EST", "8:00 PM EST"),
        },
        "Facebook": {
            "Beauty": ("Fri 4–7 PM EST", "5:45 PM EST"),
            "Gaming": ("Thu 6–8 PM EST", "7:00 PM EST"),
            "Fitness": ("Wed 5–8 PM EST", "7:30 PM EST"),
            "Default": ("Tue 6–8 PM EST", "7:00 PM EST"),
        },
    }

    platform_key = platform if platform in defaults else platform.capitalize()
    platform_defaults = defaults.get(platform_key, defaults.get("TikTok"))
    time_range, peak = platform_defaults.get(niche, platform_defaults.get("Default"))
    return day_name, time_range, peak


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        platform = request.form.get("platform", "TikTok")
        platform = platform if platform in ["TikTok", "YouTube", "Instagram", "Facebook"] else platform.capitalize()

        video_file = request.files.get("video")
        csv_file = request.files.get("csv")

        if not video_file:
            return jsonify({"error": "No video file provided."})

        # Save video
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            video_path = temp_video.name
            video_file.save(video_path)
            original_name = video_file.filename or "uploaded_video.mp4"

        # Save CSV if present (we'll attach a short summary only)
        csv_summary = ""
        if csv_file:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_csv:
                    csv_path = tmp_csv.name
                    csv_file.save(csv_path)
                csv_summary = f"CSV uploaded: {csv_file.filename}"
            except Exception as e:
                csv_summary = f"CSV upload error: {e}"

        # Extract metadata
        try:
            clip = VideoFileClip(video_path)
            duration = round(clip.duration, 2)
            fps = round(clip.fps or 30)
            width, height = clip.size
            clip.close()
        except Exception as e:
            return jsonify({"error": f"Video processing failed: {str(e)}"}), 500

        # Visual sampling
        brightness, tone, frames_b64 = analyze_video_visuals(video_path, sample_frames=8)

        # Build prompt that tells model NOT to return detected-niche or best-time footers.
        # Backend will append these to ensure there's only one final footer.
        visual_descriptor = f"brightness={brightness}, tone='{tone}', aspect_ratio={round(width/height,3)}, sample_frames={len(frames_b64)}"
        frames_display = frames_b64[:3]

        prompt = f"""
You are an expert social media content analyst and viral strategist.

IMPORTANT: Use the visual information (frame samples and the visual descriptor) as the PRIMARY source for niche detection and recommendations.
Do NOT output a final 'Detected Niche' block or any 'Best Time to Post' block — the backend will append one consistent footer. Also do not repeat the day/time at the end. Keep to the required structure only.

Video metadata (for context):
- Platform: {platform}
- Filename (reference only): "{original_name}"
- Duration: {duration}s
- Resolution: {width}x{height}
- FPS: {fps}
- Visual descriptor: {visual_descriptor}
- CSV context: {csv_summary}

Frame samples (base64 JPEG strings): {frames_display}

Provide output in this exact structure tailored to the specified platform. Do NOT include detected niche or best time blocks — backend will add those.

### 🎬 Video Summary
📏 Duration: {duration}s | {width}x{height}px | {fps}fps  
💡 Visual Tone: {tone} | Brightness: {brightness}

### 💬 AI-Generated Viral Insights:
1️⃣ **Scroll-Stopping Caption ({platform} only)**  
(Provide one caption optimized specifically for the requested platform only.)

2️⃣ **5 Hashtags ({platform} only)**  
(Provide 5 hashtags ideally suited to this platform and detected niche.)

3️⃣ **Engagement Tip**  
(One concise actionable tip.)

4️⃣ **Viral Optimization Score (1–100)**  
(Provide a numeric score and a one-line rationale.)

5️⃣ **Motivational Tip**  
(Short encouraging line.)

### 🔥 Viral Comparison:
Provide 3 real public viral videos from {platform} within the detected niche. For each:
- Video title or short URL
- What made it go viral (concrete signals: audio trend, edit style, hook, challenge, timeframe)
- How to replicate that success with this uploaded video

### 🧠 Optimization Advice:
List 3–5 specific improvements (editing, hook, audio, overlays, first frame) to increase ranking on {platform}, explicitly comparing to the viral examples listed above.
"""

        # Call OpenAI (text-only prompt). Model will not be allowed to return posting time / niche footers.
        ai_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert in short-form video analysis and platform optimization."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.9,
            max_tokens=1800,
        )

        ai_text = ""
        try:
            ai_text = ai_response.choices[0].message.content.strip()
        except Exception:
            ai_text = ""

        if not ai_text:
            return jsonify({"error": "⚠ No results received from AI."}), 500

        # Clean excessive blank lines
        ai_text = re.sub(r"\n{3,}", "\n\n", ai_text).strip()

        # Attempt to detect niche from the model output (prefers explicit mention in the "Viral Comparison" or the descriptions)
        niche = None
        # try to find "niche-like" words in the AI text
        for candidate in ["Beauty", "Gaming", "Fitness", "Food", "Travel", "Education", "Music", "Fashion", "Sports", "Comedy", "Photography", "Lifestyle", "Barbershop", "Hair", "Haircut", "Beauty/Barber", "Barber"]:
            if candidate.lower() in ai_text.lower():
                niche = candidate
                break
        # If not found explicitly, fallback to a simple heuristic: look for keywords
        if not niche:
            keywords_map = {
                "hair": "Barbershop",
                "barber": "Barbershop",
                "makeup": "Beauty",
                "workout": "Fitness",
                "game": "Gaming",
                "recipe": "Food",
                "travel": "Travel",
            }
            lowered = ai_text.lower()
            for kw, mapped in keywords_map.items():
                if kw in lowered:
                    niche = mapped
                    break

        if not niche:
            niche = "General"

        # Compute best posting time (single footer)
        day_name, time_range, peak_time = best_posting_time(platform, niche)

        # Append footer once (no duplicates)
        final_output = f"""{ai_text}

🎯 **Detected Niche:** {niche}
🕓 **Best Time to Post for {niche} ({platform}, {day_name})**:
⏰ {time_range}
💡 Peak engagement around {peak_time}.
"""

        return jsonify({"results": final_output})

    except Exception as exc:
        # Helpful debug printing (server logs)
        print("Error in /analyze:", str(exc))
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(debug=True)
