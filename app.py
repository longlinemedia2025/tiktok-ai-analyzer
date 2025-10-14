from flask import Flask, request, jsonify
import os
import datetime
import traceback
from moviepy.editor import VideoFileClip
import numpy as np
import openai

# ========= CONFIG =========
openai.api_key = os.getenv("OPENAI_API_KEY")
app = Flask(__name__)

# ========= HELPER FUNCTIONS =========
def analyze_video_properties(video_path):
    clip = VideoFileClip(video_path)
    duration = round(clip.duration, 2)
    width, height = clip.size
    aspect_ratio = round(width / height, 3)
    frame = clip.get_frame(clip.duration / 2)
    brightness = round(np.mean(frame) / 2.55, 2)  # Normalize to 0-100 scale
    tone = "bright" if brightness > 70 else "dark" if brightness < 40 else "neutral or mixed"
    heuristic_score = 9 if brightness >= 60 else 7
    clip.close()

    return {
        "duration": duration,
        "resolution": f"{width}x{height}",
        "aspect_ratio": aspect_ratio,
        "brightness": brightness,
        "tone": tone,
        "heuristic_score": heuristic_score
    }

def generate_ai_analysis(filename, props):
    prompt = f"""
You are a TikTok video optimization expert. The video is titled "{filename}".
Its properties are:
- Duration: {props['duration']}s
- Resolution: {props['resolution']}
- Aspect Ratio: {props['aspect_ratio']}
- Brightness: {props['brightness']}
- Tone: {props['tone']}

Generate content in this format ONLY (do not add any other text):

🎬 Drag and drop your TikTok video file here: "C:\\Users\\Administrator1\\Downloads\\{filename}"
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
(Create one viral caption for this video.)

### 2. 5 Viral Hashtags
(List 5 relevant hashtags.)

### 3. Actionable Improvement Tip for Engagement
(Give one short tip.)

### 4. Viral Optimization Score (1–100)
(Estimate a realistic viral potential score and explain briefly.)

### 5. Short Motivation on How to Increase Virality
(Give a short, encouraging motivational note.)

🔥 Viral Comparison Results:
### Comparison with Viral TikToks in the Same Niche
(Provide 3 example summaries of viral TikToks that relate to this one’s content and explain why they succeeded and how to replicate their success.)

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

            result.innerHTML = '🎥 Running TikTok Viral Optimizer... Please wait...';
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
