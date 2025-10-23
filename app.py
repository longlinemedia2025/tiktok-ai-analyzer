from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import cv2
import numpy as np
import tempfile
from openai import OpenAI
import pytesseract
from PIL import Image
import threading
import time

# ==========================
# Flask + OpenAI Setup
# ==========================
app = Flask(__name__, template_folder="templates")
CORS(app)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Shared variable for progress tracking
progress_status = {"stage": "idle", "result": None}

# If tesseract not in PATH, uncomment below and adjust path:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# ==========================
# Helper Functions
# ==========================
def extract_video_frames(video_path, frame_rate=1):
    """Extract one frame per second for analysis"""
    frames = []
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * frame_rate) if fps > 0 else 30
    count = 0
    success, frame = cap.read()
    while success:
        if count % frame_interval == 0:
            frames.append(frame)
        success, frame = cap.read()
        count += 1
    cap.release()
    return frames


def analyze_video(video_path):
    """Extract visual + text information for AI understanding"""
    global progress_status
    progress_status["stage"] = "Extracting frames..."
    frames = extract_video_frames(video_path)
    text_detected = []
    visual_descriptions = []

    for i, frame in enumerate(frames[:5]):  # Limit to 5 frames for efficiency
        progress_status["stage"] = f"Analyzing frame {i+1}/5..."
        # Convert frame to PIL
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # OCR detection
        text = pytesseract.image_to_string(pil_img)
        if text.strip():
            text_detected.append(text.strip())

        # Save temporary frame for CLIP analysis
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_img:
            pil_img.save(temp_img.name)
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are an expert content classifier that describes what’s visually happening in an image."},
                        {"role": "user", "content": [
                            {"type": "text", "text": "Describe this image briefly in terms of its main subject, activity, and mood."},
                            {"type": "image_url", "image_url": f"file://{temp_img.name}"}
                        ]}
                    ]
                )
                desc = response.choices[0].message.content
                if desc:
                    visual_descriptions.append(desc)
            except Exception as e:
                print("Vision model error:", e)
                continue

    combined_visuals = " ".join(visual_descriptions)
    combined_text = " ".join(text_detected)
    return combined_visuals, combined_text


def generate_ai_report(video_path, platform):
    global progress_status
    visuals, extracted_text = analyze_video(video_path)

    progress_status["stage"] = "Generating AI report..."
    prompt = f"""
You are an expert social media analyst. Analyze this video for the chosen platform: {platform}.
The video’s visual content: {visuals}
Text detected in video (OCR): {extracted_text}

Please provide results in the following exact Markdown format:

AI Results

### 🎬 Video Summary
Summarize what’s happening visually and textually.

### 💬 AI-Generated Viral Insights:
1️⃣ **Scroll-Stopping Caption ({platform} only)**  
A creative caption relevant to the video’s niche.

2️⃣ **5 Hashtags ({platform} only)**  
Five platform-optimized hashtags based on the video’s theme.

3️⃣ **Engagement Tip**  
One line of advice for boosting engagement on {platform}.

4️⃣ **Viral Optimization Score (1–100)**  
Numeric score + one-sentence rationale.

5️⃣ **Motivational Tip**  
A short encouraging line related to content creation.

### 🔥 Viral Comparison:
Find and summarize 3 *real viral videos* from {platform} that match this niche.
Include their short URLs and briefly explain why they went viral
and how the user can replicate each example.

### 🧠 Optimization Advice:
Give 3–5 practical suggestions for improving this video to match or outperform
real viral examples in its niche.

🎯 **Detected Niche:** (AI-identified based on visuals and text)
🕓 **Best Time to Post for this Niche ({platform}, today)**:
⏰ Best posting window in EST
💡 Peak engagement hour in EST
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.7,
            messages=[
                {"role": "system", "content": "You are a professional social media growth strategist and video content analyzer."},
                {"role": "user", "content": prompt}
            ]
        )
        result = response.choices[0].message.content
        progress_status["stage"] = "Complete"
        progress_status["result"] = result
        return result
    except Exception as e:
        progress_status["stage"] = f"Error during AI generation: {str(e)}"
        return None


# ==========================
# Flask Routes
# ==========================
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    global progress_status
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    video = request.files['video']
    platform = request.form.get('platform', 'TikTok')
    progress_status = {"stage": "Uploading video...", "result": None}

    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
        video.save(tmp.name)

        def background_task():
            try:
                generate_ai_report(tmp.name, platform)
            except Exception as e:
                progress_status["stage"] = f"Error: {str(e)}"

        threading.Thread(target=background_task).start()

    return jsonify({"message": "Upload successful. Analysis started."})


@app.route('/progress', methods=['GET'])
def get_progress():
    return jsonify(progress_status)


# ==========================
# Run Flask App
# ==========================
if __name__ == "__main__":
    app.run(debug=True)
