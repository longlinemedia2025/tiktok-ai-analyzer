from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import cv2
import numpy as np
from moviepy.editor import VideoFileClip
from openai import OpenAI

# Initialize Flask and CORS
app = Flask(__name__, template_folder="templates")  # lowercase "templates"
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
        print("❌ Missing object detection model files")
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

    print(f"✅ Video analyzed — Brightness: {avg_brightness:.2f}, Colorfulness: {avg_colorfulness:.2f}")
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
            print("⚠️ No video in request.files")
            return jsonify({"error": "No video uploaded"}), 400

        video = request.files["video"]
        if video.filename == "":
            print("⚠️ Empty filename received")
            return jsonify({"error": "Empty filename"}), 400

        os.makedirs("uploads", exist_ok=True)
        video_path = os.path.join("uploads", video.filename)
        video.save(video_path)
        print(f"✅ Video saved to {video_path}")

        # Analyze visuals and objects
        try:
            analysis = analyze_video_properties(video_path)
        except Exception as e:
            print("❌ Error analyzing video:", str(e))
            return jsonify({"error": f"Video analysis failed: {str(e)}"}), 500

        if "error" in analysis:
            print("❌ Object detection error:", analysis["error"])
            return jsonify(analysis), 500

        try:
            clip = VideoFileClip(video_path)
            duration = round(clip.duration, 2)
            width, height = clip.size
            aspect_ratio = round(width / height, 3)
            clip.close()
        except Exception as e:
            print("❌ MoviePy failed:", str(e))
            return jsonify({"error": f"MoviePy failed to process video: {str(e)}"}), 500

        print("✅ Video analysis succeeded. Calling OpenAI...")

        prompt = f"""
Analyze this TikTok video:
Brightness: {analysis['brightness']}
Colorfulness: {analysis['colorfulness']}
Objects: {', '.join(analysis['objects'])}
Duration: {duration}s
Resolution: {width}x{height}
Aspect Ratio: {aspect_ratio}
Provide captions, hashtags, and engagement tips.
"""

        try:
            ai_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,
                max_tokens=600
            )
            ai_text = ai_response.choices[0].message.content.strip()
        except Exception as e:
            print("❌ OpenAI API error:", str(e))
            ai_text = f"AI analysis failed: {str(e)}"

        print("✅ Returning JSON response to client")
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
        print("🔥 UNCAUGHT ERROR:", str(e))
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


# ========== Main ==========
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 Server running on port {port}")
    app.run(host="0.0.0.0", port=port)
