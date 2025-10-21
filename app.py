from flask import Flask, request, jsonify, render_template, Response
from flask_cors import CORS
import os, cv2, tempfile, base64, numpy as np, datetime, time, threading, queue
from moviepy.editor import VideoFileClip
from openai import OpenAI

app = Flask(__name__, template_folder="templates")
CORS(app)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Global message queue for SSE feed ---
live_queue = queue.Queue()

def push_feed(msg):
    """Push live feed message to queue for frontend"""
    live_queue.put(msg)


# --- Helper: analyze visuals ---
def analyze_video_visuals(video_path):
    push_feed("Analyzing video frames and tone...")
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

        _, buffer = cv2.imencode('.jpg', resized)
        frames_b64.append(base64.b64encode(buffer).decode('utf-8'))

    cap.release()
    avg_color = np.mean(avg_colors, axis=0) if avg_colors else [127, 127, 127]
    brightness = np.mean(avg_color)
    tone = "dark and moody" if brightness < 60 else "neutral or mixed" if brightness < 130 else "bright and energetic"
    push_feed(f"Average brightness: {brightness:.2f}, tone: {tone}")
    return brightness, tone, frames_b64


# --- Dynamic AI-assisted posting time ---
def best_posting_time(platform, niche):
    today = datetime.datetime.now()
    day = today.strftime("%A")
    push_feed("Calculating best posting time...")

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


# --- Background worker thread for live AI processing ---
def background_analyze(platform, video_path, video_name):
    try:
        push_feed("Extracting video metadata...")
        clip = VideoFileClip(video_path)
        duration = round(clip.duration, 2)
        w, h = clip.size
        aspect_ratio = round(w / h, 3)
        clip.reader.close()
        clip.close()

        brightness, tone, frames_b64 = analyze_video_visuals(video_path)

        push_feed("Contacting AI engine for full viral insights...")
        prompt = f"""
You are an expert social media content analyst AI.

Based on these details:
- Platform: {platform}
- Video name: "{video_name}"
- Duration: {duration} seconds
- Aspect Ratio: {aspect_ratio}
- Average Brightness: {brightness}
- Tone: {tone}

Below are 5 sampled video frames (base64 JPEGs):
{frames_b64[:3]}

1️⃣ Identify the most likely content niche.
2️⃣ Generate full viral optimization insights in this format:
🎬 Video Info
💬 Viral Insights
🔥 3 Viral Video Examples
🎯 Detected Niche
🕓 Best Time to Post
"""

        ai_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )

        ai_text = ai_response.choices[0].message.content

        niche = "Default"
        for possible in ["Beauty", "Gaming", "Fitness", "Food", "Travel", "Education", "Music", "Fashion", "Sports", "Comedy"]:
            if possible.lower() in ai_text.lower():
                niche = possible
                break

        day, best_time_line = best_posting_time(platform, niche)

        final_summary = f"""
{ai_text}

🎯 **Detected Niche:** {niche}
🕓 **Best Time to Post for {niche} ({platform}, {day})**:
{best_time_line}
"""

        push_feed("✅ Analysis complete.")
        live_queue.put({"final": final_summary})
    except Exception as e:
        live_queue.put({"error": str(e)})


# --- Flask routes ---
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze_video():
    platform = request.form.get('platform', 'TikTok')
    video = request.files.get('video')
    csv_file = request.files.get('csv')

    if not video:
        return jsonify({'error': 'No video uploaded'}), 400

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        video.save(tmp.name)
        video_path = tmp.name

    # CSV is optional — no changes if not provided
    if csv_file:
        push_feed(f"CSV '{csv_file.filename}' uploaded successfully.")

    # Start AI analysis in a background thread
    thread = threading.Thread(target=background_analyze, args=(platform, video_path, video.filename))
    thread.start()

    return jsonify({'status': 'processing'})


@app.route('/stream')
def stream():
    """Stream live AI feed via SSE."""
    def event_stream():
        while True:
            msg = live_queue.get()
            if isinstance(msg, dict) and "final" in msg:
                yield f"event: complete\ndata: {msg['final']}\n\n"
                break
            elif isinstance(msg, dict) and "error" in msg:
                yield f"event: error\ndata: {msg['error']}\n\n"
                break
            else:
                yield f"data: {msg}\n\n"
    return Response(event_stream(), mimetype="text/event-stream")


if __name__ == '__main__':
    app.run(debug=True)
