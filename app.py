from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os, cv2, tempfile, base64, numpy as np, datetime
from moviepy.editor import VideoFileClip
from openai import OpenAI

app = Flask(__name__, template_folder="templates")
CORS(app)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Helper: analyze visuals ---
def analyze_video_visuals(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, 5, dtype=int)
    frames_b64 = []
    avg_colors = []

    for i in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue
        resized = cv2.resize(frame, (224, 224))
        avg_color = cv2.mean(resized)[:3]
        avg_colors.append(avg_color)

        # Convert to base64 for AI visual reasoning
        _, buffer = cv2.imencode('.jpg', resized)
        frames_b64.append(base64.b64encode(buffer).decode('utf-8'))

    cap.release()
    avg_color = np.mean(avg_colors, axis=0) if avg_colors else [127, 127, 127]
    brightness = np.mean(avg_color)
    tone = "dark and moody" if brightness < 60 else "neutral or mixed" if brightness < 130 else "bright and energetic"
    return brightness, tone, frames_b64


# --- Dynamic AI-assisted posting time ---
def best_posting_time(platform, niche):
    today = datetime.datetime.now()
    day = today.strftime("%A")

    prompt = f"""
You are a social media analytics AI.
Estimate the best posting window and engagement peak for the {niche} niche on {platform} if today is {day}.
Return in the format:
"⏰ TIME_RANGE_EST  |  💡 PEAK_EST"
Example: "⏰ 5–8 PM EST  |  💡 Peak ~6:45 PM EST"
"""
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        line = resp.choices[0].message.content.strip()
    except Exception:
        line = "⏰ 6–9 PM EST  |  💡 Peak ~7:30 PM EST"

    return day, line


# --- Flask routes ---
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze_video():
    platform = request.form.get('platform', 'TikTok')
    video = request.files.get('video')

    if not video:
        return jsonify({'error': 'No video uploaded'}), 400

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        video.save(tmp.name)
        video_path = tmp.name

    # --- Try to extract metadata ---
    try:
        clip = VideoFileClip(video_path)
        duration = round(clip.duration, 2)
        w, h = clip.size
        aspect_ratio = round(w / h, 3)
        clip.reader.close()
        clip.close()
    except Exception as e:
        return jsonify({'error': f'Video processing failed: {str(e)}'}), 500

    brightness, tone, frames_b64 = analyze_video_visuals(video_path)

    # --- AI visual + metadata niche analysis ---
    prompt = f"""
You are an expert social media content analyst AI.

Based on these details:
- Platform: {platform}
- Video name: "{video.filename}"
- Duration: {duration} seconds
- Aspect Ratio: {aspect_ratio}
- Average Brightness: {brightness}
- Tone: {tone}

Below are 5 sampled video frames (base64 JPEGs):
{frames_b64[:3]}  # sample subset for brevity

1️⃣ Identify the most likely content niche (e.g. Beauty, Fitness, Gaming, Travel, Food, Comedy, Fashion, etc.).
2️⃣ Generate complete viral optimization insights using this exact format:

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
Give **3 examples of real viral videos** from this niche.
For each example:
- Video Concept Summary
- What made it go viral
- How the user can replicate that success

Finally, restate:
🎯 Detected Niche
🕓 Best Time to Post ({platform}, today)
"""

    ai_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    ai_text = ai_response.choices[0].message.content

    # --- Extract niche automatically ---
    niche = "Default"
    for possible in ["Beauty", "Gaming", "Fitness", "Food", "Travel", "Education", "Music", "Fashion", "Sports", "Comedy"]:
        if possible.lower() in ai_text.lower():
            niche = possible
            break

    # --- Get best posting time dynamically ---
    day, best_time_line = best_posting_time(platform, niche)

    final_summary = f"""
{ai_text}

🎯 **Detected Niche:** {niche}
🕓 **Best Time to Post for {niche} ({platform}, {day})**:
{best_time_line}
"""

    return jsonify({'results': final_summary})


if __name__ == '__main__':
    app.run(debug=True)
