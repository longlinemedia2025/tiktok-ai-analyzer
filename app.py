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

openai.api_key = os.getenv("OPENAI_API_KEY")
app = Flask(__name__)

# ========= HELPER FUNCTIONS =========

def frame_to_base64(frame):
    img = Image.fromarray(frame.astype("uint8"))
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def analyze_video_properties(video_path):
    clip = VideoFileClip(video_path)
    duration = round(clip.duration, 2)
    width, height = clip.size
    aspect_ratio = round(width / height, 3)

    frame1 = clip.get_frame(clip.duration * 0.25)
    frame2 = clip.get_frame(clip.duration * 0.5)
    frame3 = clip.get_frame(clip.duration * 0.75)

    brightness = round(np.mean(frame2) / 2.55, 2)
    tone = "bright" if brightness > 70 else "dark" if brightness < 40 else "neutral or mixed"
    heuristic_score = 9 if brightness >= 60 else 7

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
    images_input = "\n".join([f"[FRAME {i+1} - base64 image data: {b[:120]}...]" for i, b in enumerate(props["frames_base64"])])
    prompt = f"""
You are an expert TikTok strategist and AI visual analyst.

Analyze this TikTok video purely based on its visuals and metadata:

🎬 Duration: {props['duration']}s
🖼 Resolution: {props['resolution']}
📱 Aspect Ratio: {props['aspect_ratio']}
💡 Brightness: {props['brightness']}
🎨 Tone: {props['tone']}
⭐ Heuristic Score: {props['heuristic_score']}/10

Extracted Video Frames:
{images_input}

Now output your analysis using the exact structure below:

🎬 TikTok Video Analyzer
📱 Niche:
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
                background-color: #0b0b0b;
                color: #f0f0f0;
                font-family: 'Segoe UI', sans-serif;
                text-align: center;
                padding: 80px;
            }
            h1 { color: #00ffa0; }
            #drop_zone {
                border: 3px dashed #00ffa0;
                padding: 60px;
                width: 70%;
                margin: 0 auto;
                border-radius: 14px;
                transition: background-color 0.3s;
            }
            #drop_zone.dragover { background-color: rgba(0,255,160,0.08); }
            #result {
                margin-top: 30px;
                background-color: #1a1a1a;
                padding: 25px;
                border-radius: 12px;
                white-space: pre-wrap;
                text-align: left;
                font-size: 15px;
            }
            video {
                margin-top: 20px;
                width: 300px;
                border-radius: 12px;
                border: 2px solid #00ffa0;
            }
            .loader {
                margin-top: 30px;
                color: #00ffa0;
                font-size: 16px;
                font-weight: bold;
                letter-spacing: 1px;
                animation: glow 1.5s infinite alternate;
            }
            @keyframes glow {
                from { text-shadow: 0 0 5px #00ffa0; opacity: 0.6; }
                to { text-shadow: 0 0 20px #00ffa0; opacity: 1; }
            }
        </style>
    </head>
    <body>
        <h1>🎬 TikTok AI Analyzer</h1>
        <div id="drop_zone">Drag & Drop your TikTok video here</div>
        <video id="preview" controls style="display:none;"></video>
        <div class="loader" id="loader" style="display:none;">Analyzing your video... please wait ⏳</div>
        <div id="result"></div>

        <script>
        const dropZone = document.getElementById('drop_zone');
        const result = document.getElementById('result');
        const loader = document.getElementById('loader');
        const preview = document.getElementById('preview');

        dropZone.addEventListener('dragover', e => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        });
        dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
        dropZone.addEventListener('drop', async e => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
            const file = e.dataTransfer.files[0];
            if (!file) return;

            result.innerHTML = '';
            loader.style.display = 'block';
            preview.style.display = 'block';
            preview.src = URL.createObjectURL(file);

            const formData = new FormData();
            formData.append('video', file);

            try {
                const res = await fetch('/analyze', { method: 'POST', body: formData });
                const text = await res.text(); // safer than res.json()
                let data;

                try {
                    data = JSON.parse(text);
                } catch {
                    throw new Error("Server returned unexpected response. Please retry.");
                }

                loader.style.display = 'none';
                typeWriterEffect(data.output || '⚠️ Something went wrong.');
            } catch (err) {
                loader.style.display = 'none';
                result.innerHTML = '⚠️ ' + err.message;
            }
        });

        function typeWriterEffect(text) {
            let i = 0;
            result.innerHTML = '';
            const speed = 8;
            function type() {
                if (i < text.length) {
                    result.innerHTML += text.charAt(i);
                    i++;
                    setTimeout(type, speed);
                }
            }
            type();
        }
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
        error_text = f"⚠️ Internal Server Error:\\n{str(e)}\\n\\n{traceback.format_exc()}"
        print(error_text)
        return jsonify({"output": error_text}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
