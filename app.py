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
    tone = "dark and moody" if brightness < 60 else "neutral or mixed" if brightness < 130 else "bright and energetic"
    return round(brightness, 2), tone

# --- Helper: determine best posting time ---
def best_posting_time(platform, niche):
    day = datetime.datetime.now().strftime("%A")
    defaults = {
        "TikTok": {"Beauty": ("Thu 4–7 PM", "8:43 PM"),
                   "Gaming": ("Fri 5–8 PM", "7:15 PM"),
                   "Fitness": ("Mon 6–9 PM", "8:00 PM"),
                   "Default": ("Wed 6–9 PM", "7:30 PM")},
        "Instagram": {"Beauty": ("Sun 5–8 PM", "6:45 PM"),
                      "Gaming": ("Mon 6–9 PM", "8:10 PM"),
                      "Fitness": ("Tue 5–8 PM", "7:00 PM"),
                      "Default": ("Wed 6–9 PM", "7:30 PM")},
        "YouTube": {"Beauty": ("Sat 2–6 PM", "3:45 PM"),
                    "Gaming": ("Fri 6–10 PM", "8:00 PM"),
                    "Fitness": ("Sun 8–11 AM", "9:30 AM"),
                    "Default": ("Thu 6–9 PM", "8:00 PM")},
        "Facebook": {"Beauty": ("Fri 4–7 PM", "5:45 PM"),
                     "Gaming": ("Thu 6–8 PM", "7:00 PM"),
                     "Fitness": ("Wed 5–8 PM", "7:30 PM"),
                     "Default": ("Tue 6–8 PM", "7:00 PM")}
    }
    t, peak = defaults.get(platform, {}).get(niche, defaults.get(platform, {}).get("Default"))
    return f"{day}, EST", t, peak

# --- Core route ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_video():
    platform = request.form.get('platform', 'TikTok')
    video = request.files['video']
    if not video:
        return jsonify({'error': 'No video uploaded'}), 400

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        video.save(tmp.name)
        path = tmp.name

    clip = VideoFileClip(path)
    duration = round(clip.duration, 2)
    w, h = clip.size
    aspect_ratio = round(w / h, 3)
    brightness, tone = analyze_video_visuals(path)

    # AI: combine video visuals, file name, and metadata for niche detection
    prompt = f"""
You are a content analyst AI. Based on the filename "{video.filename}", brightness {brightness}, tone {tone}, and aspect ratio {aspect_ratio},
determine what niche this video belongs to (e.g. Beauty, Fitness, Gaming, Travel, Food, Education, etc.).
Then generate platform-specific viral optimization data for {platform} using this structured format:

🎬 Video: {video.filename}
📏 Duration: {duration}s
🖼 Resolution: {w}x{h}
📱 Aspect Ratio: {aspect_ratio}
💡 Brightness: {brightness}
🎨 Tone: {tone}

💬 AI-Generated Viral Insights:
1️⃣ Scroll-Stopping Caption (TikTok, Instagram, YouTube, Facebook)
2️⃣ 5 Hashtags for each platform
3️⃣ Engagement Tip
4️⃣ Viral Optimization Score (1–100)
5️⃣ Motivational Tip

🔥 Viral Comparison Results:
3 examples of viral videos in the same niche.
For each example:
- Video Concept Summary
- What made it go viral
- How to replicate success

Finally:
🎯 Detected Niche
🕓 Best Time to Post ({platform}, today)
"""
    ai_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    ai_text = ai_response.choices[0].message.content

    # Extract detected niche for better time prediction
    niche = "Default"
    for possible in ["Beauty", "Gaming", "Fitness", "Food", "Travel", "Education", "Music", "Fashion", "Sports", "Comedy"]:
        if possible.lower() in ai_text.lower():
            niche = possible
            break

    day, best_time, peak_time = best_posting_time(platform, niche)

    summary = f"""
{ai_text}

🎯 **Detected Niche:** {niche}
🕓 **Best Time to Post for {niche} ({platform}, {day})**:
⏰ {best_time}
💡 Peak engagement around {peak_time}.
"""
    return jsonify({'results': summary})

if __name__ == '__main__':
    app.run(debug=True)
