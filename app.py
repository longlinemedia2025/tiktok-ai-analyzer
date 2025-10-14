from flask import Flask, request, jsonify
import os
import datetime
import base64
import traceback
import numpy as np
from moviepy.editor import VideoFileClip
import openai
from io import BytesIO
from PIL import Image

# ========= CONFIG =========
openai.api_key = os.getenv("OPENAI_API_KEY")
app = Flask(__name__)

# ========= HELPER FUNCTIONS =========

def frame_to_base64(frame):
    """Convert a NumPy video frame to base64-encoded JPEG"""
    img = Image.fromarray(frame.astype("uint8"))
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def analyze_video_properties(video_path):
    clip = VideoFileClip(video_path)
    duration = round(clip.duration, 2)
    width, height = clip.size
    aspect_ratio = round(width / height, 3)

    # Sample 3 frames evenly spaced
    frame1 = clip.get_frame(clip.duration * 0.25)
    frame2 = clip.get_frame(clip.duration * 0.5)
    frame3 = clip.get_frame(clip.duration * 0.75)

    brightness = round(np.mean(frame2) / 2.55, 2)
    tone = "bright" if brightness > 70 else "dark" if brightness < 40 else "neutral/mixed"
    heuristic_score = 9 if brightness >= 60 else 7

    # Convert to base64 for AI visual input
    frames_base64 = [frame_to_base64(f) for f in [frame1, frame2, frame3]]
    clip.close()

    return {
        "duration": duration,
        "resolution": f"{width}x{height}",
        "aspect_ratio": aspect_ratio,
        "brightness": brightness,
        "tone": tone,
        "heuristic_score": heuristic_score,
        "frames_base64": frames_base64
    }

def generate_ai_analysis(filename, props):
    images_input = "\n\n".join([f"[FRAME {i+1} - base64 image data: {b[:120]}...]" for i, b in enumerate(props["frames_base64"])])
    
    prompt = f"""
You are a TikTok video optimization expert. You will analyze 3 base64-encoded frames from a video and its basic properties.
Your goal: determine the video's likely niche, main subject, and recommend accurate captions + hashtags.

### Video Properties
- Duration: {props['duration']}s
- Resolution: {props['resolution']}
- Aspect Ratio: {props['aspect_ratio']}
- Brightness: {props['brightness']}
- Tone: {props['tone']}
- Heuristic Score: {props['heuristic_score']}/10

### Video Frames (base64 JPEG data)
{images_input}

### Instructions:
1. Identify what the video is about from the visuals (ignore the file name).
2. Determine the likely niche or topic (e.g., fitness, food, hair transformation, travel, gaming, etc.).
3. Suggest a scroll-stopping caption, 5 hashtags, a viral optimization score, and an engagement tip.

Output format (no extra commentary):

🎬 TikTok Video Analyzer
📱 Niche: (your guess)
💬 Caption:
🏷 Hashtags:
⭐ Viral Optimization Score (1–100):
💡 Engagement Tip:
🔥 Motivation:
📊 Why this could go viral:
    """

    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# ========= ROUTES =========

@app.route("/")
def home():
    return """
    <html>
    <head>
        <title>TikTok AI Analyzer</title>
        <style>
            body {
                background-color: #0f0f0f;
                color: #f5f5f5;
                font-family: Arial, sans-serif;
                text-align: center;
                padding: 80px;
            }
            h1 { color: #00ff88; }
            #drop_zone {
                border: 3px dashed #00ff88;
                padding: 60px;
                width: 70%;
                margin: 0 auto;
                border-radius: 12px;
                transition: background-color 0.3s;
            }
            #drop_zone.dragover { background-color: rgba(0,255,136,0.1); }
            #result {
                margin-top: 30px;
                background-color: #1b1b1b;
                padding: 25px;
                border-radius: 12px;
                white-space: pre-wrap;
                text-align: left;
            }
        </style>
    </head>
    <body>
        <h1>🎬 TikTok AI Analyzer</h1>
        <div id="drop_zone">Drag & Drop your TikTok video here</div>
        <div id="result"></div>

        <script>
        const dropZone = document.getElementById('drop_zone');
        const result = document.getElementById('result');

        dropZone.addEventListener('dragover', e => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        });
        dropZone.addEventListener('dragleave', e => {
            dropZone.classList.remove('dragover');
        });
        dropZone.addEventListener('drop', async e => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
            const file = e.dataTransfer.files[0];
            if (!file) return;

            result.innerHTML = '🎥 Analyzing video visuals and generating TikTok insights...';
            const formData = new FormData();
            formData.append('video', file);

            try {
                const res = await fetch('/analyze', { method: 'POST', body: formData });
                const data = await res.json();
                result.innerHTML = data.output || '⚠️ Something went wrong.';
            } catch (err) {
                result.innerHTML = '⚠️ Request failed: ' + err.message;
            }
        });
        </script>
    </body>
    </html>
    """

@app.route("/analyze", methods=["POST"])
def analyze_video():
    try:
        video = request.files["video"]
        temp_path = os.path.join("/tmp", f"temp_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.mp4")
        video.save(temp_path)

        props = analyze_video_properties(temp_path)
        ai_output = generate_ai_analysis(video.filename, props)
        os.remove(temp_path)

        return jsonify({"output": ai_output})
    except Exception as e:
        error_text = f"⚠️ Internal Server Error:\n{str(e)}\n\n{traceback.format_exc()}"
        print(error_text)
        return jsonify({"output": error_text}), 500

# ========= MAIN =========
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
