from flask import Flask, request, jsonify, render_template_string
import os
import numpy as np
from moviepy.editor import VideoFileClip
from openai import OpenAI
import tempfile
import base64
import cv2

# ========== CONFIG ==========
app = Flask(__name__)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ========== FRONTEND ==========
HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>TikTok AI Video Analyzer</title>
  <style>
    body {
      background-color: #0d0d0d;
      color: #e0e0e0;
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
      text-align: center;
      margin: 0;
      padding: 2rem;
    }
    h1 {
      color: #00b894;
    }
    #drop-zone {
      border: 2px dashed #00b894;
      padding: 40px;
      border-radius: 10px;
      cursor: pointer;
      transition: background-color 0.3s;
      margin: 40px auto;
      width: 60%;
    }
    #drop-zone:hover {
      background-color: #1a1a1a;
    }
    input[type=file] {
      display: none;
    }
    #result {
      white-space: pre-wrap;
      text-align: left;
      background: #111;
      padding: 20px;
      border-radius: 10px;
      width: 80%;
      margin: 2rem auto;
      box-shadow: 0 0 10px #00b894;
    }
  </style>
</head>
<body>
  <h1>🎬 TikTok Viral Optimizer</h1>
  <div id="drop-zone">
    <p>🎬 Drag & Drop your TikTok video here or click to upload</p>
    <input type="file" id="fileInput" accept="video/*" />
  </div>
  <div id="result"></div>

  <script>
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('fileInput');
    const resultDiv = document.getElementById('result');

    dropZone.addEventListener('click', () => fileInput.click());

    dropZone.addEventListener('dragover', (e) => {
      e.preventDefault();
      dropZone.style.backgroundColor = '#1a1a1a';
    });

    dropZone.addEventListener('dragleave', () => {
      dropZone.style.backgroundColor = '';
    });

    dropZone.addEventListener('drop', (e) => {
      e.preventDefault();
      dropZone.style.backgroundColor = '';
      const file = e.dataTransfer.files[0];
      uploadFile(file);
    });

    fileInput.addEventListener('change', (e) => {
      const file = e.target.files[0];
      uploadFile(file);
    });

    async function uploadFile(file) {
      const formData = new FormData();
      formData.append("video", file);

      resultDiv.innerHTML = "🎥 Running TikTok Viral Optimizer...";

      const response = await fetch("/analyze", {
        method: "POST",
        body: formData
      });

      if (!response.ok) {
        resultDiv.innerHTML = "⚠️ Request failed: " + response.statusText;
        return;
      }

      const data = await response.json();
      if (data.error) {
        resultDiv.innerHTML = "⚠️ " + data.error;
      } else {
        resultDiv.textContent = data.result;
      }
    }
  </script>
</body>
</html>
"""

# ========== VIDEO ANALYSIS ==========
def analyze_video_properties(video_path):
    clip = VideoFileClip(video_path)
    duration = round(clip.duration, 2)
    width, height = clip.size
    fps = clip.fps
    aspect_ratio = round(width / height, 3)
    frame = clip.get_frame(clip.duration / 2)
    brightness = np.mean(frame)
    tone = "bright" if brightness > 180 else "dark" if brightness < 75 else "neutral or mixed"
    clip.close()
    return {
        "duration_seconds": duration,
        "resolution": f"{width}x{height}",
        "aspect_ratio": aspect_ratio,
        "frame_rate": fps,
        "brightness": round(brightness, 2),
        "tone": tone
    }

def extract_visual_summary(video_path):
    """Capture several frames and summarize the visuals to help niche detection."""
    vidcap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in np.linspace(0, total_frames - 1, num=5, dtype=int):
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, i)
        success, image = vidcap.read()
        if success:
            small = cv2.resize(image, (224, 224))
            avg_color = np.mean(small, axis=(0, 1)).astype(int)
            frames.append(avg_color.tolist())
    vidcap.release()
    if not frames:
        return "Unable to extract visuals."
    avg_visual = np.mean(frames, axis=0)
    return f"Average visual tone RGB: {avg_visual}"

# ========== ROUTES ==========
@app.route('/')
def index():
    return render_template_string(HTML_PAGE)

@app.route('/analyze', methods=['POST'])
def analyze_video():
    try:
        if "video" not in request.files:
            return jsonify({"error": "No video file uploaded"}), 400

        video_file = request.files["video"]
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            video_file.save(tmp.name)
            video_path = tmp.name

        props = analyze_video_properties(video_path)
        visuals_summary = extract_visual_summary(video_path)

        prompt = f"""
Analyze the TikTok video based on these visual properties and summary:
{props}
{visuals_summary}

Provide the result EXACTLY in this format:

🎬 Drag and drop your TikTok video file here: "{video_file.filename}"
🎥 Running TikTok Viral Optimizer...

🤖 Generating AI-powered analysis, captions, and viral tips...

🔥 Fetching viral video comparisons and strategic insights...

✅ TikTok Video Analysis Complete!

🎬 Video: {video_file.filename}
📏 Duration: {props['duration_seconds']}s
🖼 Resolution: {props['resolution']}
📱 Aspect Ratio: {props['aspect_ratio']}
💡 Brightness: {props['brightness']}
🎨 Tone: {props['tone']}
⭐ Heuristic Score: (estimate 1–10)

💬 AI-Generated Viral Insights:
### 1. Scroll-Stopping Caption
### 2. 5 Viral Hashtags
### 3. Actionable Improvement Tip for Engagement
### 4. Viral Optimization Score (1–100)
### 5. Short Motivation on How to Increase Virality

🔥 Viral Comparison Results:
### Comparison with Viral TikToks in the same niche
(Include 3 examples, what made them go viral, and how to replicate.)

### Takeaway Strategy
(3–4 sentences)

📋 Actionable Checklist:
   - Hook viewers in under 2 seconds.
   - Add trending sound if relevant.
   - Post during high activity times (Fri–Sun, 6–10pm).
   - Encourage comments by asking a question.
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a TikTok video performance and virality expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.8,
            max_tokens=1000
        )

        ai_result = response.choices[0].message.content.strip()
        return jsonify({"result": ai_result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ========== MAIN ==========
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=True)
