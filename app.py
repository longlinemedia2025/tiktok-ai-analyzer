from flask import Flask, request, jsonify, render_template
import os
import tempfile
import cv2
import numpy as np
from openai import OpenAI

app = Flask(__name__)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load a lightweight MobileNet model for object detection
# (These files should be added to your repo: "mobilenet_iter_73000.caffemodel" + "deploy.prototxt")
MODEL_PROTO = "deploy.prototxt"
MODEL_WEIGHTS = "mobilenet_iter_73000.caffemodel"

if os.path.exists(MODEL_PROTO) and os.path.exists(MODEL_WEIGHTS):
    net = cv2.dnn.readNetFromCaffe(MODEL_PROTO, MODEL_WEIGHTS)
else:
    net = None  # Skip object detection if not available


# ========= Helper Functions ========= #

def analyze_video_visuals(video_path):
    """
    Analyze visuals + objects in the video to extract:
    - Brightness, colorfulness, hue, motion
    - Objects appearing across frames
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Error: Cannot open video file.")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_interval = max(1, frame_count // 10)

    avg_colors, motion_scores = [], []
    detected_objects = []
    prev_frame = None

    for i in range(0, frame_count, sample_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.resize(frame, (320, 180))
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        avg_color = np.mean(hsv, axis=(0, 1))
        avg_colors.append(avg_color)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_frame is not None:
            diff = cv2.absdiff(prev_frame, gray)
            motion_scores.append(np.mean(diff))
        prev_frame = gray

        # Object detection (if model loaded)
        if net is not None:
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                         0.007843, (300, 300), 127.5)
            net.setInput(blob)
            detections = net.forward()

            class_labels = ["background", "aeroplane", "bicycle", "bird", "boat",
                            "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                            "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                            "sofa", "train", "tvmonitor"]

            for j in range(detections.shape[2]):
                confidence = detections[0, 0, j, 2]
                if confidence > 0.4:  # Confidence threshold
                    idx = int(detections[0, 0, j, 1])
                    if idx < len(class_labels):
                        detected_objects.append(class_labels[idx])

    cap.release()

    avg_color = np.mean(avg_colors, axis=0) if avg_colors else [0, 0, 0]
    avg_motion = np.mean(motion_scores) if motion_scores else 0

    brightness = avg_color[2]
    colorfulness = avg_color[1]
    hue = avg_color[0]

    unique_objects = list(set(detected_objects))

    return {
        "brightness": round(float(brightness), 2),
        "colorfulness": round(float(colorfulness), 2),
        "hue": round(float(hue), 2),
        "motion_intensity": round(float(avg_motion), 2),
        "objects": unique_objects
    }


def generate_ai_analysis(visual_data):
    """
    Ask GPT to classify and analyze the visuals and detected objects.
    """
    object_list = ", ".join(visual_data["objects"]) if visual_data["objects"] else "no major objects detected"

    visual_summary = (
        f"Brightness: {visual_data['brightness']}, Colorfulness: {visual_data['colorfulness']}, "
        f"Hue: {visual_data['hue']}, Motion: {visual_data['motion_intensity']}, "
        f"Objects: {object_list}."
    )

    prompt = f"""
You are a TikTok content analysis AI.
Based on this visual data, predict what niche this video belongs to and generate insights.

Video description data:
{visual_summary}

Respond in this format only:

🎬 TikTok Video Analyzer  
📱 Niche: (type of content)  
💬 Caption: (engaging caption)  
🏷 Hashtags: (5 relevant hashtags)  
⭐ Viral Optimization Score (1–100): (rating)  
💡 Engagement Tip: (1-line tip for creators)  
🔥 Motivation: (motivational line)  
📊 Why this could go viral: (1–2 sentence reason)
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert at understanding TikTok video content and trends."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.8
    )

    return response.choices[0].message.content.strip()


# ========= Routes ========= #

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        if 'video' not in request.files:
            return jsonify({"error": "No video file provided"}), 400

        video = request.files['video']
        if video.filename == '':
            return jsonify({"error": "Empty file name"}), 400

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            video.save(tmp.name)
            video_path = tmp.name

        visual_data = analyze_video_visuals(video_path)
        ai_result = generate_ai_analysis(visual_data)

        os.remove(video_path)
        return jsonify({"result": ai_result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 10000)))
