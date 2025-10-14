from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import cv2
import numpy as np
from moviepy.editor import VideoFileClip
from openai import OpenAI

# Initialize
app = Flask(__name__)
CORS(app)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ========== Helper Functions ==========

def analyze_video_properties(video_path):
    """Extract brightness, colorfulness, and detect objects using OpenCV"""
    cap = cv2.VideoCapture(video_path)
    brightness_values = []
    colorfulness_values = []
    frame_count = 0
    detected_objects = set()

    # Load pre-trained object detection model (MobileNetSSD)
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
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
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

@app.route("/analyze", methods=["POST"])
def analyze_video():
    try:
        if "video" not in request.files:
            return jsonify({"error": "No video uploaded"}), 400

        video = request.files["video"]
        video_path = os.path.join("uploads", video.filename)
        os.makedirs("uploads", exist_ok=True)
        video.save(video_path)

        # Analyze visuals and objects
        analysis = analyze_video_properties(video_path)
        if "error" in analysis:
            return jsonify(analysis), 500

        # Send data to OpenAI for viral comparison
        prompt = f"""
        Analyze this TikTok-style video based on:
        - Visual brightness: {analysis['brightness']}
        - Color intensity: {analysis['colorfulness']}
        - Detected objects: {', '.join(analysis['objects'])}

        🔥 Viral Comparison Results:
        Give 3 examples of similar viral TikToks and for each include:
        1. Video Concept Summary
        2. What Made The Video Example Go Viral
        3. How To Replicate Success
        """

        ai_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=600
        )

        result_text = ai_response.choices[0].message.content
        return jsonify({
            "success": True,
            "analysis": analysis,
            "ai_results": result_text
        })

    except Exception as e:
        print("🔥 Server Error:", str(e))
        return jsonify({"error": str(e)}), 500

# ========== Main ==========
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
