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
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MAX_SIZE_MB = 90
TRIM_THRESHOLD_MB = 70


# ==================================================
# 🔹 Best Time Suggestion by Niche
# ==================================================
def get_best_posting_time(niche: str):
    """Return the best posting window for today based on the inferred niche."""
    niche = niche.lower().strip()
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
    peak_hour = random.randint(6, 9) if "PM" in window else random.randint(9, 11)
    peak_minute = random.randint(0, 59)
    peak_time = f"{peak_hour}:{peak_minute:02d} {'PM' if 'PM' in window else 'AM'} EST"

    return f"🕓 **Best Time to Post for {niche.title()} ({today})**:\n⏰ {window} EST\n💡 Peak engagement around {peak_time}."


# ==================================================
# 🔹 Analyze Video Properties
# ==================================================
def analyze_video_properties(video_path):
    cap = cv2.VideoCapture(video_path)
    brightness_values, colorfulness_values = [], []
    frame_count = 0
    detected_objects = set()

    proto = "deploy.prototxt"
    model = "mobilenet_iter_73000.caffemodel"
    if not os.path.exists(proto) or not os.path.exists(model):
        return {"error": "Missing object detection model files"}

    net = cv2.dnn.readNetFromCaffe(proto, model)
    class_names = [
        "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
        "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
        "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
    ]

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness_values.append(np.mean(gray))
        (B, G, R) = cv2.split(frame)
        rg = np.abs(R - G)
        yb = np.abs(0.5 * (R + G) - B)
        colorfulness_values.append(np.sqrt(np.mean(rg ** 2) + np.mean(yb ** 2)))

        if frame_count % 30 == 0:
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                         0.007843, (300, 300), 127.5)
            net.setInput(blob)
            detections = net.forward()
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.4:
                    idx = int(detections[0, 0, i, 1])
                    detected_objects.add(class_names[idx])

    cap.release()
    return {
        "brightness": float(np.mean(brightness_values)),
        "colorfulness": float(np.mean(colorfulness_values)),
        "objects": list(detected_objects)
    }


# ==================================================
# 🔹 Adaptive CSV-Based Optimization
# ==================================================
def analyze_csv_performance(csv_path):
    """Read CSV and summarize performance insights."""
    try:
        df = pd.read_csv(csv_path)
        key_cols = ["views", "likes", "comments", "shares", "saves"]
        available = [c for c in key_cols if c in df.columns]
        if not available:
            return None

        avg_perf = df[available].mean().to_dict()
        best_video = df.loc[df["views"].idxmax()] if "views" in df.columns else None

        summary = f"""
📊 **CSV Performance Insights Used in Analysis**
Average Performance:
{avg_perf}

Top Performing Video:
{best_video.to_dict() if best_video is not None else "N/A"}
"""
        return summary
    except Exception as e:
        return f"⚠️ CSV analysis failed: {str(e)}"


@app.route("/")
def home():
    return render_template("index.html")


# ==================================================
# 🔹 Combined Analyze Route (Video + CSV)
# ==================================================
@app.route("/analyze", methods=["POST"])
def analyze_video():
    try:
        if "video" not in request.files:
            return jsonify({"error": "No video uploaded"}), 400

        video = request.files["video"]
        csv_file = request.files.get("csv")

        os.makedirs("uploads", exist_ok=True)
        video_path = os.path.join("uploads", video.filename)
        video.save(video_path)

        csv_context = ""
        if csv_file:
            csv_path = os.path.join("uploads", csv_file.filename)
            csv_file.save(csv_path)
            csv_context = analyze_csv_performance(csv_path) or ""

        clip = VideoFileClip(video_path)
        duration = round(clip.duration, 2)
        width, height = clip.size
        aspect_ratio = round(width / height, 3)

        analysis = analyze_video_properties(video_path)
        if "error" in analysis:
            return jsonify(analysis), 500

        # === AI Prompt ===
        prompt = f"""
You are a TikTok algorithm analysis assistant.

Analyze this video and use past performance insights (if any) to adjust your recommendations.

Video Data:
- Brightness: {analysis['brightness']}
- Color intensity: {analysis['colorfulness']}
- Detected objects: {', '.join(analysis['objects'])}
- Duration: {duration}s
- Resolution: {width}x{height}
- Aspect Ratio: {aspect_ratio}

{csv_context if csv_context else "No CSV uploaded — use general TikTok trends."}

Respond in this EXACT format:

🎬 Drag and drop your TikTok video file here: "{video.filename}"
🎥 Running TikTok Viral Optimizer...

🤖 Generating AI-powered analysis, captions, and viral tips...

🔥 Fetching viral video comparisons and strategic insights...

✅ TikTok Video Analysis Complete!

🎬 Video: {video.filename}
📏 Duration: {duration}s
🖼 Resolution: {width}x{height}
📱 Aspect Ratio: {aspect_ratio}
💡 Brightness: {round(analysis['brightness'], 2)}
🎨 Tone: neutral or mixed
⭐ Heuristic Score: Give a 1–10 rating estimating visual appeal.

💬 AI-Generated Viral Insights:
### 1. Scroll-Stopping Caption
(Create one engaging caption using emojis and emotional hooks.)

### 2. 5 Viral Hashtags
(List five relevant hashtags — include 3 trending + 2 niche-targeted less popular ones.)

### 3. Actionable Improvement Tip for Engagement
(Provide one concise, actionable engagement tip.)

### 4. Viral Optimization Score (1–100)
(Give a numerical score and explain why.)

### 5. Short Motivation on How to Increase Virality
(Provide a motivational paragraph that encourages improvement.)

🔥 Viral Comparison Results:
### Comparison with Viral TikToks in the Same Niche
Include 3 examples — each must include:
#### Viral Example 1
- **Video Concept Summary:**
- **What Made It Go Viral:**
- **How to Replicate Success:**

#### Viral Example 2
- **Video Concept Summary:**
- **What Made It Go Viral:**
- **How to Replicate Success:**

#### Viral Example 3
- **Video Concept Summary:**
- **What Made It Go Viral:**
- **How to Replicate Success:**

### Takeaway Strategy
(Provide a 3–4 sentence takeaway on how to improve virality.)

📋 Actionable Checklist:
- Hook viewers in under 2 seconds.
- Add trending sound if relevant.
- Post during high activity times (Fri–Sun, 6–10pm).
- Encourage comments by asking a question.
        """

        ai_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
            max_tokens=900
        )
        ai_text = ai_response.choices[0].message.content.strip()

        # === Detect Niche + Posting Time ===
        try:
            niche_prompt = f"""
Based on this analysis, determine its most likely TikTok content niche.
{ai_text}
Possible niches: Gaming, Beauty, Music, Fitness, Comedy, Other.
Return ONLY the niche name.
"""
            niche_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You classify TikTok video niches."},
                    {"role": "user", "content": niche_prompt}
                ],
                max_tokens=20
            )
            detected_niche = niche_response.choices[0].message.content.strip().lower()
            best_time_text = get_best_posting_time(detected_niche)
            ai_text += f"\n\n🎯 **Detected Niche:** {detected_niche.title()}\n{best_time_text}"
            if csv_context:
                ai_text += "\n\n📈 (Adaptive insights powered by your uploaded TikTok CSV.)"
        except Exception as e:
            ai_text += f"\n\n⚠️ Niche detection failed: {str(e)}"

        return jsonify({
            "success": True,
            "analysis": {
                "filename": video.filename,
                "duration": duration,
                "resolution": f"{width}x{height}",
                "aspect_ratio": aspect_ratio,
                "brightness": analysis["brightness"],
                "colorfulness": analysis["colorfulness"],
                "objects": analysis["objects"]
            },
            "ai_results": ai_text
        })

    except Exception as e:
        print("🔥 Error:", str(e))
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
