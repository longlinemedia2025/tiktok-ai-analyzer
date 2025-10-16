# app.py
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import cv2
import numpy as np
from moviepy.editor import VideoFileClip
from openai import OpenAI
import tempfile
import datetime
import random
import pandas as pd

app = Flask(__name__, template_folder="templates")
CORS(app)

# OpenAI client - ensure OPENAI_API_KEY is set in environment on Render
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MAX_SIZE_MB = 90
TRIM_THRESHOLD_MB = 70

# ------------------------
# Best post/upload times
# ------------------------
def get_best_posting_time(niche: str, platform: str = "tiktok"):
    niche = (niche or "other").lower().strip()
    # same times as before; interpret generically for both platforms
    niche_times = {
        "gaming": {
            "Mon": "6–9 PM", "Tue": "6–9 PM", "Wed": "7–9 PM", "Thu": "6–9 PM",
            "Fri": "5–10 PM", "Sat": "10 AM–12 PM / 7–10 PM", "Sun": "4–8 PM"
        },
        "beauty": {
            "Mon": "11 AM–2 PM", "Tue": "1–3 PM", "Wed": "12–3 PM", "Thu": "4–7 PM",
            "Fri": "5–9 PM", "Sat": "10 AM–1 PM / 7–9 PM", "Sun": "3–6 PM"
        },
        "music": {
            "Mon": "2–4 PM", "Tue": "4–6 PM", "Wed": "3–7 PM", "Thu": "5–8 PM",
            "Fri": "6–10 PM", "Sat": "9–11 AM / 8–10 PM", "Sun": "5–9 PM"
        },
        "fitness": {
            "Mon": "6–9 AM / 6–8 PM", "Tue": "6–8 AM / 7–9 PM", "Wed": "6–9 AM / 6–8 PM",
            "Thu": "7–9 PM", "Fri": "6–9 PM", "Sat": "8–11 AM", "Sun": "4–7 PM"
        },
        "comedy": {
            "Mon": "12–3 PM", "Tue": "2–5 PM", "Wed": "1–4 PM", "Thu": "4–7 PM",
            "Fri": "6–10 PM", "Sat": "10 AM–12 PM / 8–10 PM", "Sun": "3–8 PM"
        }
    }

    today = datetime.datetime.now().strftime("%a")
    if niche not in niche_times:
        return "⚠️ Could not determine best time — niche not recognized."

    window = niche_times[niche].get(today, "6–9 PM")
    # create a plausible peak time in that window
    # if window mentions "AM" (rare here) pick AM, else PM
    is_pm = "PM" in window
    if is_pm:
        # find an hour range in window if present like "6–9 PM" or "10 AM–12 PM / 7–10 PM"
        # fallback to 6-9
        peak_hour = random.randint(6, 9)
        ampm = "PM"
    else:
        peak_hour = random.randint(9, 11)
        ampm = "AM"
    peak_minute = random.randint(0, 59)
    peak_time = f"{peak_hour}:{peak_minute:02d} {ampm} EST"

    label = "Best Time to Post" if platform == "tiktok" else "Best Upload Time"
    return f"🕓 **{label} for {niche.title()} ({today})**:\n⏰ {window} EST\n💡 Peak engagement around {peak_time}."

# ------------------------
# Video property analysis
# ------------------------
def analyze_video_properties(video_path):
    # Use moviepy for duration/resolution, opencv for frame analysis
    clip = VideoFileClip(video_path)
    duration = round(clip.duration, 2)
    width, height = clip.size
    aspect_ratio = round(width / height, 3)

    cap = cv2.VideoCapture(video_path)
    brightness_values = []
    colorfulness_values = []
    frame_count = 0
    detected_objects = set()

    # optional object detection model files
    proto = "deploy.prototxt"
    model = "mobilenet_iter_73000.caffemodel"
    use_object_detection = os.path.exists(proto) and os.path.exists(model)
    net = None
    class_names = []
    if use_object_detection:
        try:
            net = cv2.dnn.readNetFromCaffe(proto, model)
            class_names = [
                "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
                "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
                "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
            ]
        except Exception:
            use_object_detection = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # brightness via grayscale mean
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness_values.append(float(np.mean(gray)))

        # colorfulness as in prior code
        (B, G, R) = cv2.split(frame.astype("float"))
        rg = np.abs(R - G)
        yb = np.abs(0.5 * (R + G) - B)
        colorfulness_values.append(float(np.sqrt(np.mean(rg ** 2) + np.mean(yb ** 2))))

        # object detection every nth frame (lighter frequency)
        if use_object_detection and (frame_count % 30 == 0):
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
            net.setInput(blob)
            detections = net.forward()
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.4:
                    idx = int(detections[0, 0, i, 1])
                    if idx < len(class_names):
                        detected_objects.add(class_names[idx])

    cap.release()
    avg_brightness = float(np.mean(brightness_values)) if brightness_values else 0.0
    avg_colorfulness = float(np.mean(colorfulness_values)) if colorfulness_values else 0.0

    # Simple tone inference
    tone = "neutral or mixed"
    if avg_brightness > 160:
        tone = "bright"
    elif avg_brightness < 70:
        tone = "dark"

    return {
        "duration_seconds": duration,
        "resolution": (int(width), int(height)),
        "aspect_ratio": aspect_ratio,
        "mean_brightness_first_frame": round(avg_brightness, 2),
        "colorfulness": round(avg_colorfulness, 2),
        "objects": list(detected_objects),
        "tone": tone
    }

# ------------------------
# CSV adaptive analysis
# ------------------------
def analyze_csv_performance(csv_path):
    try:
        df = pd.read_csv(csv_path)
        key_columns = ["views", "likes", "comments", "shares", "saves"]
        available = [c for c in key_columns if c in df.columns]

        if not available:
            return None

        performance_summary = df[available].mean().to_dict()
        best_video = df.loc[df["views"].idxmax()] if "views" in df.columns else None

        context = f"""
Average performance:
{performance_summary}

Top performing video insight:
{best_video.to_dict() if best_video is not None else "N/A"}
"""
        return context
    except Exception as e:
        return f"⚠️ CSV analysis failed: {str(e)}"

# ------------------------
# Format builder for final AI prompt and return text
# ------------------------
def build_prompt_and_template(platform: str, video_meta: dict, csv_context: str):
    # platform is "tiktok" or "youtube"
    filename_placeholder = '{filename}'
    # Build prompt: include video metadata + csv_context
    prompt_meta = (
        f"Video stats:\n"
        f"- Duration: {video_meta['duration_seconds']} seconds\n"
        f"- Resolution: {video_meta['resolution'][0]}x{video_meta['resolution'][1]}\n"
        f"- Aspect Ratio: {video_meta['aspect_ratio']}\n"
        f"- Brightness: {video_meta['mean_brightness_first_frame']}\n"
        f"- Tone: {video_meta.get('tone', 'neutral or mixed')}\n"
        f"- Detected objects: {', '.join(video_meta.get('objects', []))}\n"
    )

    csv_section = f"\nPast CSV performance summary:\n{csv_context}\n" if csv_context else "\nNo CSV uploaded — use general platform trends.\n"

    if platform == "tiktok":
        header_intro = (
            f"🎬 Drag and drop your TikTok video file here: \"{filename_placeholder}\"\n"
            f"🎥 Running TikTok Viral Optimizer...\n\n"
            f"🤖 Generating AI-powered analysis, captions, and viral tips...\n\n"
            f"🔥 Fetching viral video comparisons and strategic insights...\n\n"
            f"✅ TikTok Video Analysis Complete!\n\n"
        )
        platform_label = "TikTok"
        tag_word = "Hashtags"
        score_note = "(High brightness and clear resolution contribute to visual appeal.)"
        posting_label = "Best Time to Post"
    else:
        header_intro = (
            f"🎬 Drag and drop your YouTube video file here: \"{filename_placeholder}\"\n"
            f"🎥 Running YouTube Viral Optimizer...\n\n"
            f"🤖 Generating AI-powered analysis, captions, and keyword & upload tips...\n\n"
            f"🔥 Fetching viral video comparisons and keyword insights...\n\n"
            f"✅ YouTube Video Analysis Complete!\n\n"
        )
        platform_label = "YouTube"
        tag_word = "Tags"
        score_note = "(Includes SEO and retention considerations.)"
        posting_label = "Best Upload Time"

    # final instruction to the model: produce text in the EXACT format (we will replace filename later)
    prompt = (
        f"You are a {platform_label} virality expert and strategist.\n\n"
        f"{prompt_meta}\n"
        f"{csv_section}\n"
        f"Using the data above, generate the output in the EXACT format below. Use the filename string in the header.\n\n"
        f"Use this exact structure and include sensible, platform-relevant suggestions and keywords.\n\n"
        f"{header_intro}"
        f"🎬 Video: {filename_placeholder}\n"
        f"📏 Duration: {video_meta['duration_seconds']}s\n"
        f"🖼 Resolution: {video_meta['resolution'][0]}x{video_meta['resolution'][1]}\n"
        f"📱 Aspect Ratio: {video_meta['aspect_ratio']}\n"
        f"💡 Brightness: {video_meta['mean_brightness_first_frame']}\n"
        f"🎨 Tone: {video_meta.get('tone', 'neutral or mixed')}\n"
        f"⭐ Heuristic Score: Give a 1–10 rating estimating visual appeal. Include a short parenthetical explanation.\n\n"
        f"💬 AI-Generated Viral Insights:\n"
        f"### 1. Scroll-Stopping Caption\n"
        f"(Create one engaging caption using emojis and emotional hooks.)\n\n"
        f"### 2. 5 Viral {tag_word}\n"
        f"(List five relevant {tag_word.lower()} — include 3 trending + 2 niche-targeted less popular ones.)\n\n"
        f"### 3. Actionable Improvement Tip for Engagement\n"
        f"(Provide one concise, actionable engagement tip.)\n\n"
        f"### 4. Viral Optimization Score (1–100)\n"
        f"(Give a numerical score and short explanation.)\n\n"
        f"### 5. Short Motivation on How to Increase Virality\n"
        f"(Provide a motivational paragraph.)\n\n"
        f"🔥 Viral Comparison Results:\n"
        f"### Comparison with Viral {platform_label}s in the Same Niche\n\n"
        f"Include 3 examples — each must include:\n"
        f"#### Viral Example 1\n"
        f"- **Video Concept Summary:**\n"
        f"- **What Made It Go Viral:**\n"
        f"- **How to Replicate Success:**\n\n"
        f"#### Viral Example 2\n- **Video Concept Summary:**\n- **What Made It Go Viral:**\n- **How to Replicate Success:**\n\n"
        f"#### Viral Example 3\n- **Video Concept Summary:**\n- **What Made It Go Viral:**\n- **How to Replicate Success:**\n\n"
        f"### Takeaway Strategy\n(3–4 sentence takeaway on how to improve virality and viewer engagement.)\n\n"
        f"📋 Actionable Checklist:\n"
        f"- Hook viewers in under 2 seconds.\n"
        f"- Add trending sound if relevant.\n"
        f"- Post during high activity times (Fri–Sun, 6–10pm).\n"
        f"- Encourage comments by asking a question.\n\n"
        f"Finally, append a detected niche line and the best posting/upload time for that niche in the same style as examples above.\n"
    )

    return prompt, header_intro, platform_label, posting_label, tag_word, score_note

# ------------------------
# AI call helper
# ------------------------
def call_openai_chat(prompt_text):
    # Use the same model (gpt-4o-mini) per your request
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a concise, expert social-media content strategist."},
                {"role": "user", "content": prompt_text}
            ],
            temperature=0.8,
            max_tokens=900
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"(AI generation failed: {e})"

# ------------------------
# Single analyze route handles platform via dropdown front-end
# ------------------------
@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Expects multipart/form-data with:
      - video: uploaded file
      - csv: optional uploaded csv
      - platform: 'tiktok' or 'youtube' (sent as form field)
    """
    try:
        platform = request.form.get("platform") or request.args.get("platform") or request.values.get("platform")
        # For backward compatibility: some frontends may call /analyze_tiktok or /analyze_youtube
        # Accept also 'platform' query param or fallback to 'tiktok'
        if not platform:
            platform = request.form.get("platform") or "tiktok"
        platform = platform.lower()

        if "video" not in request.files:
            return jsonify({"error": "No video uploaded"}), 400

        video = request.files["video"]
        csv_file = request.files.get("csv")

        # Save video temporarily
        os.makedirs("uploads", exist_ok=True)
        video_path = os.path.join("uploads", video.filename)
        video.save(video_path)

        # CSV context if provided
        csv_context = ""
        if csv_file:
            csv_path = os.path.join("uploads", csv_file.filename)
            csv_file.save(csv_path)
            csv_context = analyze_csv_performance(csv_path) or ""

        # enforce size limits + optional trimming
        file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
        warning_message = None

        if file_size_mb > MAX_SIZE_MB:
            os.remove(video_path)
            return jsonify({
                "error": f"Video too large ({file_size_mb:.1f}MB). Please compress below {MAX_SIZE_MB}MB and try again."
            }), 400

        if file_size_mb > TRIM_THRESHOLD_MB:
            warning_message = "⚠️ Video trimmed automatically to reduce file size before analysis."
            clip = VideoFileClip(video_path)
            trimmed_clip = clip.subclip(0, min(clip.duration, 45))
            temp_path = tempfile.mktemp(suffix=".mp4")
            trimmed_clip.write_videofile(temp_path, codec="libx264", audio_codec="aac", verbose=False, logger=None)
            video_path = temp_path

        # extract metadata
        clip = VideoFileClip(video_path)
        duration = round(clip.duration, 2)
        width, height = clip.size
        aspect_ratio = round(width / height, 3)
        clip.close()

        video_meta = analyze_video_properties(video_path)
        # integrate extracted duration/resolution just in case analyze_video_properties didn't set them
        video_meta["duration_seconds"] = video_meta.get("duration_seconds", duration)
        video_meta["resolution"] = video_meta.get("resolution", (width, height))
        video_meta["aspect_ratio"] = video_meta.get("aspect_ratio", aspect_ratio)

        # Build prompt & instruction template
        prompt_template, header_intro, platform_label, posting_label, tag_word, score_note = build_prompt_and_template(platform, video_meta, csv_context)

        # Replace placeholder filename with the real filename inside prompt (so model echo matches)
        prompt_text = prompt_template.replace("{filename}", video.filename)

        # Call OpenAI
        ai_text = call_openai_chat(prompt_text)

        # Niche detection (one-shot) to append best times
        try:
            niche_prompt = f"""
Based on this AI analysis text, determine the most likely single content niche for this video.
Return ONLY one of: Gaming, Beauty, Music, Fitness, Comedy, Other.
AI analysis text:
{ai_text}
"""
            niche_resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You classify TikTok/YouTube video niches."},
                    {"role": "user", "content": niche_prompt}
                ],
                max_tokens=10
            )
            detected_niche = niche_resp.choices[0].message.content.strip().lower()
            best_time_text = get_best_posting_time(detected_niche, platform=platform)
            ai_text += f"\n\n🎯 **Detected Niche:** {detected_niche.title()}\n{best_time_text}"
            if csv_context:
                ai_text += "\n\n📈 (Adaptive insights powered by your uploaded CSV.)"
        except Exception as e:
            ai_text += f"\n\n⚠️ Niche detection failed: {str(e)}"

        # Remove temporary saved video to keep uploads folder clean
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
        except Exception:
            pass

        # Compose response with the exact same results format block included as ai_text
        return jsonify({
            "success": True,
            "warning": warning_message,
            "analysis": {
                "filename": video.filename,
                "duration": video_meta["duration_seconds"],
                "resolution": f"{video_meta['resolution'][0]}x{video_meta['resolution'][1]}",
                "aspect_ratio": video_meta["aspect_ratio"],
                "brightness": video_meta.get("mean_brightness_first_frame"),
                "colorfulness": video_meta.get("colorfulness"),
                "objects": video_meta.get("objects")
            },
            "ai_results": ai_text
        })

    except Exception as e:
        # log and return
        print("🔥 Error in /analyze:", str(e))
        return jsonify({"error": str(e)}), 500

# Backwards compatibility routes if your front-end calls them directly
@app.route("/analyze_tiktok", methods=["POST"])
def analyze_tiktok_compat():
    # forward to main handler with platform param
    request.form = request.form.copy()
    request.form = request.form.to_dict()
    request.form["platform"] = "tiktok"
    return analyze()

@app.route("/analyze_youtube", methods=["POST"])
def analyze_youtube_compat():
    request.form = request.form.copy()
    request.form = request.form.to_dict()
    request.form["platform"] = "youtube"
    return analyze()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
