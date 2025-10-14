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
import time

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
    tone = "bright" if brightness > 70 else "dark" if brightness < 40 else "neutral or mixed"
    heuristic_score = 9 if brightness >= 60 else 7

    # Convert frames to base64 for AI visual understanding
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
You are an expert TikTok content strategist and AI video analyst.

You are analyzing a TikTok video using its **visual frames** and metadata (do not rely on file name).

### Video Properties
🎬 Duration: {props['duration']}s
🖼 Resolution: {props['resolution']}
📱 Aspect Ratio: {props['aspect_ratio']}
💡 Brightness: {props['brightness']}
🎨 Tone: {props['tone']}
⭐ Heuristic Score: {props['heuristic_score']}/10

### Extracted Video Frames
{images_input}

---

Now produce your full analysis using the following exact template and wording — do not change structure, section titles, or emojis:

🎬 Drag and drop your TikTok video file here: "{filename}"
🎥 Running TikTok Viral Optimizer...

🤖 Generating AI-powered analysis, captions, and viral tips...

🔥 Fetching viral video comparisons and strategic insights...

✅ TikTok Video Analysis Complete!

🎬 Video: {filename}
📏 Duration: {props['duration']}s
🖼 Resolution: {props['resolution']}
📱 Aspect Ratio: {props['aspect_ratio']}
💡 Brightness: {props['brightness']}
🎨 Tone: {props['tone']}
⭐ Heuristic Score: {props['heuristic_score']}/10

💬 AI-Generated Viral Insights:
### 1. Scroll-Stopping Caption
(Provide an engaging and emotion-driven caption based on the visuals.)

### 2. 5 Viral Hashtags
(List exactly 5 relevant hashtags matching the video’s niche and visuals.)

### 3. Actionable Improvement Tip for Engagement
(Give one realistic, creative tip to boost viewer interaction.)

### 4. Viral Optimization Score (1–100)
(Give a score and short reason in parentheses.)

### 5. Short Motivation on How to Increase Virality
(2–3 sentences motivating the creator to improve their reach.)

🔥 Viral Comparison Results:
### Comparison with Viral TikToks in the Same Niche

#### Viral Example 1: (Include title, summary, what made it go viral, and replication tips.)
#### Viral Example 2: (Same structure.)
#### Viral Example 3: (Same structure.)

### Takeaway Strategy
(Summarize lessons learned from the viral examples and how to apply them.)

📋 Actionable Checklist:
   - Hook viewers in under 2 seconds.
   - Add trending sound if relevant.
   - Post during high activity times (Fri–Sun, 6–10pm).
   - Encourage comments by asking a question.
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
        <div class="loader" id="loader" style="display:none;">Analyzing your video... please wait ⏳</div>
        <div id="result"></div>

        <script>
        const dropZone = document.getElementById('drop_zone');
        const result = document.getElementById('result');
        const loader = document.getElementById('loader');

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

            result.innerHTML = '';
            loader.style.display = 'block';

            const formData = new FormData();
            formData.append('video', file);

            try {
                const res = await fetch('/analyze', { method: 'POST', body: formData });
                const data = await res.json();
                loader.style.display = 'none';
                typeWriterEffect(data.output || '⚠️ Something went wrong.');
            } catch (err) {
                loader.style.display = 'none';
                result.innerHTML = '⚠️ Request failed: ' + err.message;
            }
        });

        function typeWriterEffect(text) {
            let i = 0;
            result.innerHTML = '';
            const speed = 8; // milliseconds per character
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

# ========= MAIN =========
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
