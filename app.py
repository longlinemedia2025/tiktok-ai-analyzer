from flask import Flask, request, jsonify, render_template_string
import os
import cv2
import numpy as np
from moviepy.editor import VideoFileClip
from openai import OpenAI

# Initialize
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
app = Flask(__name__)

# ---------- HTML FRONTEND ----------
HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>TikTok AI Video Analyzer</title>
  <style>
    body {
      background: linear-gradient(135deg, #0e0e0e, #1a1a1a);
      color: #f1f1f1;
      font-family: 'Inter', sans-serif;
      text-align: center;
      padding: 60px;
    }
    h1 {
      font-size: 2.4rem;
      margin-bottom: 40px;
      color: #ffffff;
    }
    .upload-box {
      border: 2px dashed #666;
      border-radius: 12px;
      padding: 40px;
      width: 400px;
      margin: 0 auto;
      background-color: #141414;
      transition: 0.3s ease;
    }
    .upload-box:hover {
      border-color: #9b59b6;
      background-color: #1f1f1f;
    }
    input[type=file] {
      display: none;
    }
    label {
      display: block;
      cursor: pointer;
      color: #d1d1d1;
      font-size: 1.1rem;
      padding: 12px 0;
    }
    .result {
      margin-top: 40px;
      padding: 20px;
      text-align: left;
      white-space: pre-wrap;
      background: #1c1c1c;
      border-radius: 10px;
      width: 80%;
      margin-left: auto;
      margin-right: auto;
      box-shadow: 0 0 20px rgba(255,255,255,0.1);
    }
  </style>
</head>
<body>
  <h1>🎬 TikTok AI Video Analyzer</h1>
  <div class="upload-box" id="drop-zone">
    <form id="upload-form" enctype="multipart/form-data">
      <input type="file" id="video-file" name="video" accept="video/*" />
      <label for="video-file">📁 Drag & Drop or Click to Upload Video</label>
    </form>
  </div>
  <div class="result" id="result-box"></div>

<script>
  const form = document.getElementById('upload-form');
  const fileInput = document.getElementById('video-file');
  const resultBox = document.getElementById('result-box');

  fileInput.addEventListener('change', async () => {
    const file = fileInput.files[0];
    if (!file) return;
    resultBox.innerText = "⏳ Analyzing your TikTok video...";
    const formData = new FormData();
    formData.append('video', file);
    try {
      const res = await fetch('/analyze', { method: 'POST', body: formData });
      const data = await res.json();
      resultBox.innerText = data.result || "⚠️ Something went wrong.";
    } catch (err) {
      resultBox.innerText = "⚠️ Request failed: " + err.message;
    }
  });
</script>
</body>
</html>
"""

# ---------- HELPER FUNCTIONS ----------
def analyze_video_properties(video_path):
    """Extract average brightness, duration, resolution, and detect objects."""
    clip = VideoFileClip(video_path)
    duration = clip.duration
    width, height = clip.size
    aspect_ratio = round(width / height, 3)

    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    brightness_vals = []
    hue_vals = []

    # Load pre-trained object detector (COCO MobileNet)
    proto_file = "deploy.prototxt"
    model_file = "mobilenet_iter_73000.caffemodel"
    objects_detected = set()

    net = None
    if os.path.exists(proto_file) and os.path.exists(model_file):
        net = cv2.dnn.readNetFromCaffe(proto_file, model_file)
        layer_names = ["background","aeroplane","bicycle","bird","boat","bottle","bus","car",
                       "cat","chair","cow","diningtable","dog","horse","motorbike","person",
                       "pottedplant","sheep","sofa","train","tvmonitor"]

    for i in range(0, frame_count, max(1, frame_count // 10)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        brightness_vals.append(np.mean(hsv[:, :, 2]))
        hue_vals.append(np.mean(hsv[:, :, 0]))

        if net:
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
            net.setInput(blob)
            detections = net.forward()
            for j in range(detections.shape[2]):
                confidence = detections[0, 0, j, 2]
                if confidence > 0.4:
                    idx = int(detections[0, 0, j, 1])
                    if 0 <= idx < len(layer_names):
                        objects_detected.add(layer_names[idx])

    cap.release()
    clip.close()

    brightness = round(np.mean(brightness_vals), 2)
    avg_hue = round(np.mean(hue_vals), 2)
    tone = "bright" if brightness > 180 else "dark" if brightness < 80 else "neutral/mixed"

    return {
        "duration": duration,
        "resolution": f"{width}x{height}",
        "aspect_ratio": aspect_ratio,
        "brightness": brightness,
        "tone": tone,
        "objects": list(objects_detected)
    }

def generate_ai_analysis(file_name, props):
    """Ask GPT to create analysis using the user's exact result format."""
    prompt = f"""
You are an expert TikTok growth strategist and AI content analyzer. 
The user uploaded a video with these properties:
- Duration: {props['duration']} seconds
- Resolution: {props['resolution']}
- Aspect Ratio: {props['aspect_ratio']}
- Brightness: {props['brightness']}
- Tone: {props['tone']}
- Detected Objects: {', '.join(props['objects']) if props['objects'] else 'none'}

Generate results in **this exact structure**:

🎬 Drag and drop your TikTok video file here: "{file_name}"
🎥 Running TikTok Viral Optimizer...

🤖 Generating AI-powered analysis, captions, and viral tips...

🔥 Fetching viral video comparisons and strategic insights...

✅ TikTok Video Analysis Complete!

🎬 Video: {file_name}
📏 Duration: {props['duration']:.2f}s
🖼 Resolution: {props['resolution']}
📱 Aspect Ratio: {props['aspect_ratio']}
💡 Brightness: {props['brightness']}
🎨 Tone: {props['tone']}
⭐ Heuristic Score: 9/10

💬 AI-Generated Viral Insights:
### 1. Scroll-Stopping Caption
(Create one caption based on visuals and tone.)

### 2. 5 Viral Hashtags
(Provide 5 niche hashtags matching the visual tone and detected objects.)

### 3. Actionable Improvement Tip for Engagement
(Give one concrete improvement for engagement.)

### 4. Viral Optimization Score (1–100)
(Give a score and short reasoning.)

### 5. Short Motivation on How to Increase Virality
(Give a motivational paragraph explaining how to make it more viral.)

🔥 Viral Comparison Results:
### Comparison with Viral TikToks in the {props['tone']} niche

#### Viral Example 1:
- Video Concept Summary:
- What Made The Video Example Go Viral:
- How to Replicate Success:

#### Viral Example 2:
- Video Concept Summary:
- What Made The Video Example Go Viral:
- How to Replicate Success:

#### Viral Example 3:
- Video Concept Summary:
- What Made The Video Example Go Viral:
- How to Replicate Success:

📋 Actionable Checklist:
   - Hook viewers in under 2 seconds.
   - Add trending sound if relevant.
   - Post during high activity times (Fri–Sun, 6–10pm).
   - Encourage comments by asking a question.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1200,
        temperature=0.8,
    )
    return response.choices[0].message.content.strip()

# ---------- ROUTES ----------
@app.route("/")
def index():
    return render_template_string(HTML_PAGE)

@app.route("/analyze", methods=["POST"])
def analyze():
    if "video" not in request.files:
        return jsonify({"result": "⚠️ No video uploaded."})
    video = request.files["video"]
    file_path = os.path.join("uploads", video.filename)
    os.makedirs("uploads", exist_ok=True)
    video.save(file_path)

    try:
        props = analyze_video_properties(file_path)
        result_text = generate_ai_analysis(video.filename, props)
        return jsonify({"result": result_text})
    except Exception as e:
        return jsonify({"result": f"⚠️ Error: {str(e)}"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
