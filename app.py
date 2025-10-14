from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import cv2
import numpy as np
from moviepy.editor import VideoFileClip
from openai import OpenAI

# Initialize Flask and CORS
app = Flask(__name__, template_folder="Templates")
CORS(app)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ========== Helper Functions ==========

def analyze_video_properties(video_path):
    """Extracts brightness, colorfulness, and detects objects using OpenCV"""
    cap = cv2.VideoCapture(video_path)
    brightness_values = []
    colorfulness_values = []
    frame_count = 0
    detected_objects = set()

    # Load MobileNetSSD model
    proto = "deploy.prototxt"
    model = "mobilenet_iter_73000.caffemodel"
    if not os.path.exists(proto) or not os.path.exists(model):
        return {"error": "Missing object detection model files"}

    net = cv2.dnn.readNetFromCaffe(proto, model)
    class_names = [
        "background", "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
        "dog", "horse", "motorbike", "person", "pottedplant",
        "sheep", "sofa", "train", "tvmonitor"
    ]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness_values.append(np.mean(gray))

        (B, G, R) = cv2.split(frame)
        rg = np.abs(R - G)
        yb = np.abs(0.5 * (R + G) - B)
        colorfulness_values.append(np.sqrt(np.mean(rg ** 2) + np.mean(yb ** 2)))

        # Object detection every 30 frames
        if frame_count % 30 == 0:
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                         0.007843, (300, 300), 127.5)
            net.setInput(blob)
            detections = net.forward()
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.4:
                    idx = int(detections[0, 0, i, 1])
                    detected_objects.add(class_names[idx])

    cap.release()
    avg_brightness = np.mean(brightness_values)
    avg_colorfulness = np.mean(colorfulness_values)

    return {
        "brightness": float(avg_brightness),
        "colorfulness": float(avg_colorfulness),
        "objects": list(detected_objects)
    }

# ========== Routes ==========

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze_video():
    try:
        if "video" not in request.files:
            return jsonify({"error": "No video uploaded"}), 400

        video = request.files["video"]
        os.makedirs("uploads", exist_ok=True)
        video_path = os.path.join("uploads", video.filename)
        video.save(video_path)

        # Analyze visuals and objects
        analysis = analyze_video_properties(video_path)
        if "error" in analysis:
            return jsonify(analysis), 500

        # Get video details using moviepy
        clip = VideoFileClip(video_path)
        duration = round(clip.duration, 2)
        width, height = clip.size
        aspect_ratio = round(width / height, 3)

        # Construct the detailed AI prompt
        prompt = f"""
You are a TikTok algorithm analysis assistant.

Analyze this video based on the following:
- Brightness: {analysis['brightness']}
- Color intensity: {analysis['colorfulness']}
- Detected objects: {', '.join(analysis['objects'])}
- Duration: {duration}s
- Resolution: {width}x{height}
- Aspect Ratio: {aspect_ratio}

Generate a full, detailed response in this **exact format**:

🎬 Drag and drop your TikTok video file here: "{video.filename}"
🎥 Running TikTok Viral Optimizer...

🤖 Generating AI-powered analysis, captions, and viral tips...

🔥 Fetching viral video comparisons and strategic insights...

✅ TikTok Video Analysis Complete!

🎬 Video: {video.filename}
📏 Duration: {duration}s
🖼 Resolution: {width}x{height}
📱 Aspect Ratio: {aspect_ratio}
💡 Brightness: {round(analysis['brightness'], 2)}
🎨 Tone: neutral or mixed
⭐ Heuristic Score: Give a 1–10 rating estimating visual appeal.

💬 AI-Generated Viral Insights:
### 1. Scroll-Stopping Caption
(Create one engaging caption using emojis and emotional hooks.)

### 2. 5 Viral Hashtags
(List five relevant hashtags.)

### 3. Actionable Improvement Tip for Engagement
(Provide one concise, actionable engagement tip.)

### 4. Viral Optimization Score (1–100)
(Give a numerical score and explain why.)

### 5. Short Motivation on How to Increase Virality
(Provide a motivational paragraph that encourages improvement.)

🔥 Viral Comparison Results:
### Comparison with Viral TikToks in the Same Niche

Include 3 examples — each must include:
#### Viral Example 1
- **Video Concept Summary:**
- **What Made It Go Viral:**
- **How to Replicate Success:**

#### Viral Example 2
- **Video Concept Summary:**
- **What Made It Go Viral:**
- **How to Replicate Success:**

#### Viral Example 3
- **Video Concept Summary:**
- **What Made It Go Viral:**
- **How to Replicate Success:**

### Takeaway Strategy
(Provide a 3–4 sentence takeaway on how to improve virality and viewer engagement.)

📋 Actionable Checklist:
- Hook viewers in under 2 seconds.
- Add trending sound if relevant.
- Post during high activity times (Fri–Sun, 6–10pm).
- Encourage comments by asking a question.
        """

        # Call OpenAI
        ai_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
            max_tokens=900
        )

        ai_text = ai_response.choices[0].message.content.strip()

        return jsonify({
            "success": True,
            "analysis": {
                "filename": video.filename,
                "duration": duration,
                "resolution": f"{width}x{height}",
                "aspect_ratio": aspect_ratio,
                "brightness": analysis["brightness"],
                "colorfulness": analysis["colorfulness"],
                "objects": analysis["objects"]
            },
            "ai_results": ai_text
        })

    except Exception as e:
        print("🔥 Error:", str(e))
        return jsonify({"error": str(e)}), 500

# ========== Main ==========
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
