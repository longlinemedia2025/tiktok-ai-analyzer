from flask import Flask, request, jsonify, render_template_string
import os
import datetime
from moviepy.editor import VideoFileClip
import numpy as np
from openai import OpenAI

# ==============================
# CONFIGURATION
# ==============================
app = Flask(__name__)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ==============================
# SIMPLE HTML FRONTEND
# ==============================
HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>TikTok AI Analyzer</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #0f0f0f;
            color: #fff;
            text-align: center;
            padding: 40px;
        }
        h1 { color: #00ffa6; }
        .drop-zone {
            width: 60%;
            margin: 30px auto;
            padding: 50px;
            border: 3px dashed #00ffa6;
            border-radius: 20px;
            cursor: pointer;
        }
        .drop-zone:hover {
            background-color: rgba(0,255,166,0.1);
        }
        #output {
            margin-top: 30px;
            text-align: left;
            background: #1a1a1a;
            padding: 20px;
            border-radius: 12px;
        }
    </style>
</head>
<body>
    <h1>🎬 TikTok AI Analyzer</h1>
    <p>Drag and drop your TikTok video below to analyze it for viral potential!</p>
    <div class="drop-zone" id="drop-zone">Drop video here or click to upload</div>
    <input type="file" id="file-input" accept="video/*" style="display: none;">
    <div id="output"></div>

    <script>
        const dropZone = document.getElementById('drop-zone');
        const fileInput = document.getElementById('file-input');
        const output = document.getElementById('output');

        dropZone.addEventListener('click', () => fileInput.click());

        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.style.backgroundColor = 'rgba(0,255,166,0.1)';
        });

        dropZone.addEventListener('dragleave', () => {
            dropZone.style.backgroundColor = '';
        });

        dropZone.addEventListener('drop', async (e) => {
            e.preventDefault();
            const file = e.dataTransfer.files[0];
            if (!file) return;
            await uploadVideo(file);
        });

        fileInput.addEventListener('change', async (e) => {
            const file = e.target.files[0];
            if (file) await uploadVideo(file);
        });

        async function uploadVideo(file) {
            output.innerHTML = "⏳ Uploading and analyzing your video...";
            const formData = new FormData();
            formData.append("video", file);

            const response = await fetch("/analyze", {
                method: "POST",
                body: formData
            });
            const result = await response.json();
            output.innerHTML = "<pre>" + JSON.stringify(result, null, 2) + "</pre>";
        }
    </script>
</body>
</html>
"""

# ==============================
# HELPER: ANALYZE VIDEO
# ==============================
def analyze_video_properties(video_path):
    try:
        clip = VideoFileClip(video_path)
        duration = clip.duration
        width, height = clip.size
        fps = clip.fps

        frame = clip.get_frame(0)
        brightness = np.mean(frame)
        aspect_ratio = round(width / height, 3)
        resolution = f"{width}x{height}"

        clip.close()
        return {
            "duration_seconds": round(duration, 2),
            "frame_rate": round(fps, 2),
            "resolution": resolution,
            "aspect_ratio": aspect_ratio,
            "brightness": round(float(brightness), 2)
        }
    except Exception as e:
        return {"error": str(e)}

# ==============================
# ROUTES
# ==============================
@app.route("/")
def index():
    return render_template_string(HTML_PAGE)

@app.route("/analyze", methods=["POST"])
def analyze():
    if "video" not in request.files:
        return jsonify({"error": "No video uploaded."})

    video = request.files["video"]
    video_path = os.path.join("uploads", video.filename)
    os.makedirs("uploads", exist_ok=True)
    video.save(video_path)

    props = analyze_video_properties(video_path)
    if "error" in props:
        return jsonify(props)

    # Generate a detailed AI analysis prompt
    prompt = f"""
You are an AI TikTok content strategist.
Analyze this video for its viral potential using the following details:
Duration: {props['duration_seconds']} seconds
Resolution: {props['resolution']}
Aspect Ratio: {props['aspect_ratio']}
Brightness: {props['brightness']}
Frame Rate: {props['frame_rate']}

Provide a structured report including:
1. Scroll-stopping caption (with emojis)
2. 5 trending hashtags
3. Engagement improvement tip
4. Viral optimization score (1-100)
5. Motivation or insight for creator
6. Comparison with 3 viral TikToks in the same niche
7. Actionable checklist for virality
Format clearly with headers and bullet points.
"""

    try:
        ai_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a professional social media strategist and TikTok algorithm expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.85,
            max_tokens=1200
        )

        result_text = ai_response.choices[0].message.content
        props["ai_analysis"] = result_text
        return jsonify(props)
    except Exception as e:
        props["error"] = str(e)
        return jsonify(props)
    finally:
        if os.path.exists(video_path):
            os.remove(video_path)

# ==============================
# RUN APP
# ==============================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
