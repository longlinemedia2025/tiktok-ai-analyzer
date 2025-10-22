from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os, cv2, tempfile, numpy as np, datetime
from moviepy.editor import VideoFileClip
from openai import OpenAI

app = Flask(__name__, template_folder="templates")
CORS(app)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# --- Helper: lightweight frame sampling for visual content ---
def analyze_video_visuals(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, 10, dtype=int)
    colors = []
    for i in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue
        avg_color = cv2.mean(cv2.resize(frame, (64, 64)))[:3]
        colors.append(avg_color)
    cap.release()

    avg_color = np.mean(colors, axis=0) if colors else [0, 0, 0]
    brightness = np.mean(avg_color)
    tone = (
        "dark and moody"
        if brightness < 60
        else "neutral or mixed"
        if brightness < 130
        else "bright and energetic"
    )
    return round(brightness, 2), tone


# --- Helper: determine best posting time ---
def best_posting_time(platform, niche):
    current_day = datetime.datetime.now().strftime("%A")
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

    t, peak = defaults.get(platform, {}).get(
        niche, defaults.get(platform, {}).get("Default")
    )
    return f"{current_day}, EST", t, peak


# --- Core routes ---
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        platform = request.form.get("platform", "TikTok")
        video_file = request.files.get("video")
        csv_file = request.files.get("csv")

        if not video_file:
            return jsonify({"error": "No video file provided."})

        # --- Save uploaded video temporarily ---
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            video_path = temp_video.name
            video_file.save(video_path)

        # --- Save CSV if provided ---
        csv_path = None
        if csv_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_csv:
                csv_path = temp_csv.name
                csv_file.save(csv_path)

        # --- Extract video info ---
        clip = VideoFileClip(video_path)
        duration = round(clip.duration, 2)
        fps = clip.fps
        width, height = clip.size
        brightness, tone = analyze_video_visuals(video_path)
        clip.close()

        current_day = datetime.datetime.now().strftime("%A")

        # --- AI prompt for deep content + viral analysis ---
        prompt = f"""
You are a professional AI content strategist and viral marketing analyst.
Analyze a short-form video uploaded to {platform} and generate a complete viral optimization report.

The video data:
Duration: {duration}s
Resolution: {width}x{height}px
Framerate: {round(fps)}fps
Visual tone: {tone}
Brightness: {brightness}

Your tasks:
1️⃣ Detect the niche (e.g. Beauty, Fitness, Gaming, Travel, Food, Education, etc.)
2️⃣ Describe what the video is about and its emotional tone.
3️⃣ Provide a full viral optimization report using this exact structure:

### 🎬 Video Summary
📏 Duration: {duration}s | {width}x{height}px | {round(fps)}fps  
💡 Visual Tone: {tone} | Brightness: {brightness}

### 💬 AI-Generated Viral Insights:
1️⃣ **Scroll-Stopping Caption ({platform} only)**
2️⃣ **5 Hashtags ({platform} only)**
3️⃣ **Engagement Tip**
4️⃣ **Viral Optimization Score (1–100)**
5️⃣ **Motivational Tip**

### 🔥 Viral Comparison:
Find 3 real public viral videos on {platform} from the same detected niche.
Include:
- Video title or URL
- What made it go viral
- How to replicate that success

### 🧠 Optimization Advice:
Based on this video and the detected niche, list 3–5 specific things that could improve ranking and performance versus current top viral videos on {platform}.

Finally:
🎯 **Detected Niche:** [insert niche]
🕓 **Best Time to Post for [niche] ({platform}, {current_day}, EST)**:
⏰ [insert time range]
💡 Peak engagement around [insert time].
"""

        # --- Call OpenAI (stronger model) ---
        ai_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are an AI expert in social media virality and platform optimization.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.8,
            max_tokens=1800,
        )

        # --- Extract AI text safely ---
        if (
            ai_response
            and ai_response.choices
            and ai_response.choices[0].message
            and ai_response.choices[0].message.content
        ):
            ai_text = ai_response.choices[0].message.content.strip()
        else:
            ai_text = "⚠ No results received from AI."

        # --- Detect niche for correct posting time ---
        niche = "Default"
        for possible in [
            "Beauty",
            "Gaming",
            "Fitness",
            "Food",
            "Travel",
            "Education",
            "Music",
            "Fashion",
            "Sports",
            "Comedy",
        ]:
            if possible.lower() in ai_text.lower():
                niche = possible
                break

        day, best_time, peak_time = best_posting_time(platform, niche)

        # --- Append summary footer ---
        final_output = f"""
{ai_text}

🎯 **Detected Niche:** {niche}
🕓 **Best Time to Post for {niche} ({platform}, {day})**:
⏰ {best_time}
💡 Peak engagement around {peak_time}.
"""

        return jsonify({"results": final_output})

    except Exception as e:
        print("Error during analysis:", str(e))
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
