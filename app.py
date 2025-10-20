from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import cv2
import numpy as np
from moviepy.editor import VideoFileClip
from datetime import datetime
from openai import OpenAI
import tempfile

app = Flask(__name__, template_folder="templates")
CORS(app)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------- HOME ROUTE ----------
@app.route('/')
def home():
    return '''
    <h2>🎬 Welcome to the Viral Video AI Analyzer</h2>
    <p>Upload a video via <code>/analyze</code> or use the front-end upload form.</p>
    <p>This AI tool works with TikTok, Instagram, YouTube, and Facebook videos.</p>
    '''

# ---------- ANALYZE ROUTE ----------
@app.route('/analyze', methods=['POST'])
def analyze_video():
    if 'video' not in request.files:
        return jsonify({"error": "No video uploaded"}), 400

    video_file = request.files['video']
    platform = request.form.get('platform', 'TikTok')
    filename = video_file.filename

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp:
        video_file.save(temp.name)
        video_path = temp.name

    # Extract video details
    clip = VideoFileClip(video_path)
    duration = clip.duration
    width, height = clip.size
    aspect_ratio = round(width / height, 3)
    frame = clip.get_frame(0)
    brightness = round(np.mean(frame), 2)

    # Simple tone estimation
    tone = "bright" if brightness > 140 else "dark" if brightness < 70 else "neutral or mixed"

    # Heuristic Score
    heuristic_score = "8/10" if brightness > 90 else "6/10"

    # AI prompt for analysis
    prompt = f"""
    Analyze this video and generate a complete viral optimization report for {platform}.

    Video details:
    - Filename: {filename}
    - Duration: {duration:.2f}s
    - Resolution: {width}x{height}
    - Aspect ratio: {aspect_ratio}
    - Brightness: {brightness}
    - Tone: {tone}

    Please return:
    1. Scroll-stopping caption idea
    2. 5 viral hashtags
    3. Actionable engagement improvement tip
    4. Viral Optimization Score (0–100)
    5. Short motivation on how to increase virality
    6. 3 viral video examples from {platform} in the same niche with:
       - Summary of each video
       - What made it go viral
       - How the user can replicate it
    7. Best posting day and time (EST) for this video’s niche on {platform}
    8. Detected niche
    """

    response = client.responses.create(
        model="gpt-4.1",
        input=prompt,
    )

    analysis_text = response.output_text.strip()

    # Generate final formatted output
    result = f"""
🎬 Drag and drop your video file here: "{filename}"
🎥 Running {platform} Viral Optimizer...

🤖 Generating AI-powered analysis, captions, and viral tips...

🔥 Fetching viral video comparisons and strategic insights...

✅ Video Analysis Complete!

🎬 Video: {filename}
📏 Duration: {duration:.2f}s
🖼 Resolution: {width}x{height}
📱 Aspect Ratio: {aspect_ratio}
💡 Brightness: {brightness}
🎨 Tone: {tone}
⭐ Heuristic Score: {heuristic_score}

💬 AI-Generated Viral Insights:
{analysis_text}
"""

    os.remove(video_path)
    return jsonify({"result": result})


# ---------- RUN LOCALLY ----------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
