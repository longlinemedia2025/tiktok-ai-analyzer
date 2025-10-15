from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import cv2
import numpy as np
from moviepy.editor import VideoFileClip
from openai import OpenAI
import tempfile

# Initialize Flask
app = Flask(__name__, template_folder="templates")
CORS(app)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ========== Helper Function ==========
def analyze_video_properties(video_path):
    """Lightweight analysis — samples a few frames to save RAM"""
    cap = cv2.VideoCapture(video_path)
    brightness_values = []
    colorfulness_values = []
    detected_objects = set()

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

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 24
    frame_interval = fps * 2  # sample every 2 seconds
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness_values.append(np.mean(gray))

            (B, G, R) = cv2.split(frame)
            rg = np.abs(R - G)
            yb = np.abs(0.5 * (R + G) - B)
            colorfulness_values.append(np.sqrt(np.mean(rg ** 2) + np.mean(yb ** 2)))

            # Run detection on sampled frames only
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                         0.007843, (300, 300), 127.5)
            net.setInput(blob)
            detections = net.forward()
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.45:
                    idx = int(detections[0, 0, i, 1])
                    detected_objects.add(class_names[idx])
        frame_count += 1

    cap.release()
    avg_brightness = np.mean(brightness_values) if brightness_values else 0
    avg_colorfulness = np.mean(colorfulness_values) if colorfulness_values else 0

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

        # Reject files over ~60 MB (Render will crash above that)
        if len(video.read()) > 60 * 1024 * 1024:
            return jsonify({"error": "Video too large. Please upload a file under 60MB."}), 400
        video.seek(0)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            video.save(tmp.name)
            video_path = tmp.name

        clip = VideoFileClip(video_path)
        duration = round(clip.duration, 2)
        width, height = clip.size
        aspect_ratio = round(width / height, 3)
        clip.reader.close()
        clip.close()

        analysis = analyze_video_properties(video_path)
        if "error" in analysis:
            os.remove(video_path)
            return jsonify(analysis), 500

        prompt = f"""
You are a TikTok algorithm analysis assistant.

Analyze this video:
- Brightness: {analysis['brightness']}
- Color intensity: {analysis['colorfulness']}
- Detected objects: {', '.join(analysis['objects'])}
- Duration: {duration}s
- Resolution: {width}x{height}
- Aspect Ratio: {aspect_ratio}

Generate a response with captions, hashtags, and viral examples.
"""
        ai_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
            max_tokens=900
        )

        ai_text = ai_response.choices[0].message.content.strip()
        os.remove(video_path)

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
