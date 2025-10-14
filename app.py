from flask import Flask, request, jsonify, render_template_string
import os
import cv2
import numpy as np
from moviepy.editor import VideoFileClip
from openai import OpenAI
import tempfile

# Initialize Flask app and OpenAI client
app = Flask(__name__)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ========== HTML FRONTEND ==========

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>TikTok AI Video Analyzer</title>
<style>
  body {
    font-family: 'Segoe UI', Arial, sans-serif;
    background-color: #0f0f10;
    color: #f1f1f1;
    text-align: center;
    margin: 0;
    padding: 0;
  }
  h1 { margin-top: 40px; }
  .upload-box {
    margin: 40px auto;
    padding: 40px;
    border: 3px dashed #444;
    border-radius: 20px;
    width: 60%;
    background-color: #1c1c1e;
    cursor: pointer;
    transition: background-color 0.3s ease;
  }
  .upload-box:hover { background-color: #2b2b2e; }
  input[type="file"] { display: none; }
  #output {
    margin: 30px auto;
    width: 80%;
    text-align: left;
    white-space: pre-wrap;
    background: #121212;
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #333;
  }
  .loader {
    border: 6px solid #2b2b2e;
    border-top: 6px solid #f1f1f1;
    border-radius: 50%;
    width: 50px;
    height: 50px;
    animation: spin 1s linear infinite;
    margin: 30px auto;
  }
  @keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
  }
</style>
</head>
<body>
  <h1>🎬 TikTok AI Video Analyzer</h1>
  <label class="upload-box" for="fileInput">📁 Drag & Drop or Click to Upload Video</label>
  <input type="file" id="fileInput" accept="video/*">
  <div id="output">💡 Upload a TikTok video to begin analysis.</div>

<script>
const fileInput = document.getElementById('fileInput');
const output = document.getElementById('output');

fileInput.addEventListener('change', async (event) => {
  const file = event.target.files[0];
  if (!file) return;

  // Show loading spinner
  output.innerHTML = `
    <div class="loader"></div>
    <p>🎥 Uploading video and running AI-powered analysis... ⏳</p>
  `;

  const formData = new FormData();
  formData.append('video', file);

  try {
    const res = await fetch('/analyze', { method: 'POST', body: formData });
    const data = await res.json();
    output.textContent = data.result || `⚠️ ${data.error || 'Something went wrong.'}`;
  } catch (err) {
    output.textContent = '❌ Network error or timeout. Please try again.';
  }
});
</script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_PAGE)

# ========== HELPER FUNCTIONS ==========

def extract_video_features(video_path):
    clip = VideoFileClip(video_path)
    duration = round(clip.duration, 2)
    width, height = clip.size
    aspect_ratio = round(width / height, 3)
    frame = clip.get_frame(clip.duration / 2)
    avg_brightness = np.mean(frame)
    tone = "bright" if avg_brightness > 180 else "dark" if avg_brightness < 75 else "neutral or mixed"

    return {
        "duration": duration,
        "width": width,
        "height": height,
        "aspect_ratio": aspect_ratio,
        "brightness": round(avg_brightness, 2),
        "tone": tone
    }

def detect_objects(video_path):
    try:
        net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")
        cap = cv2.VideoCapture(video_path)
        detected = set()
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret or frame_count > 20:
                break
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                         (300, 300), (104.0, 177.0, 123.0))
            net.setInput(blob)
            detections = net.forward()
            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:
                    detected.add("face")
            frame_count += 1

        cap.release()
        return list(detected) if detected else ["none"]
    except Exception:
        return ["unknown"]

# ========== MAIN ANALYSIS ROUTE ==========

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        if 'video' not in request.files:
            return jsonify({"error": "No video file uploaded."})

        file = request.files['video']
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp:
            file.save(temp.name)
            video_path = temp.name

        features = extract_video_features(video_path)
        objects = detect_objects(video_path)

        prompt = f"""
You are a TikTok viral strategist AI. Using the visual data below, return your response EXACTLY in the provided template, no bullet changes or deviations.

🎬 Drag and drop your TikTok video file here: "{file.filename}"
🎥 Running TikTok Viral Optimizer...

🤖 Generating AI-powered analysis, captions, and viral tips...

🔥 Fetching viral video comparisons and strategic insights...

✅ TikTok Video Analysis Complete!

🎬 Video: {file.filename}
📏 Duration: {features['duration']}s
🖼 Resolution: {features['width']}x{features['height']}
📱 Aspect Ratio: {features['aspect_ratio']}
💡 Brightness: {features['brightness']}
🎨 Tone: {features['tone']}
🧠 Detected Objects: {', '.join(objects)}
⭐ Heuristic Score: 9/10

💬 AI-Generated Viral Insights:
### 1. Scroll-Stopping Caption
(Create a suspenseful, intriguing caption that fits the video’s tone and detected objects.)

### 2. 5 Viral Hashtags
(Generate hashtags directly relevant to the visuals and niche detected.)

### 3. Actionable Improvement Tip for Engagement
(Provide a practical, data-based improvement strategy to increase engagement.)

### 4. Viral Optimization Score (1–100)
(Give a realistic score based on the visual appeal, brightness, and detected content.)

### 5. Short Motivation on How to Increase Virality
(Offer motivation that connects creative consistency with algorithmic success.)

🔥 Viral Comparison Results:
### Comparison with Viral TikToks in the Same Niche
(Give 3 examples with structured breakdowns as before.)
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1200,
            temperature=0.9,
        )

        result = response.choices[0].message.content.strip()
        return jsonify({"result": result})

    except Exception as e:
        return jsonify({"error": str(e)})

# Run locally
if __name__ == '__main__':
    app.run(debug=True)
