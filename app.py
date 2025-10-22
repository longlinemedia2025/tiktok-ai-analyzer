from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import cv2
import numpy as np
from moviepy.editor import VideoFileClip
from openai import OpenAI
import tempfile
import datetime

app = Flask(__name__, template_folder="templates")
CORS(app)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        platform = request.form.get("platform")
        video = request.files["video"]

        # Save temporary video
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            video.save(temp_file.name)
            video_path = temp_file.name

        # Extract basic video info
        clip = VideoFileClip(video_path)
        duration = clip.duration
        width, height = clip.size
        fps = clip.fps

        # Capture first frame for analysis
        cap = cv2.VideoCapture(video_path)
        success, frame = cap.read()
        cap.release()

        if not success:
            return jsonify({"error": "Failed to read video frame."}), 400

        # Calculate brightness (average pixel intensity)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = round(np.mean(gray), 2)

        # Estimate tone based on brightness
        tone = "dark and moody" if brightness < 80 else "bright and lively" if brightness > 180 else "balanced"

        # Detect current day
        current_day = datetime.datetime.now().strftime("%A")

        # Send to GPT for analysis
        with open(video_path, "rb") as video_file:
            analysis = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional AI content strategist specializing in social media virality."
                    },
                    {
                        "role": "user",
                        "content": f"""
You are analyzing a {platform} video.

Analyze the uploaded {platform} video and generate a viral optimization report specifically for {platform} only — do not include results for any other platforms.

Determine:
1️⃣ Niche category (e.g. Beauty, Fitness, Gaming, Food, Travel, Comedy, Education, etc.)
2️⃣ What the video is about
3️⃣ What emotion or reaction it triggers
4️⃣ What visual tone it has (use provided tone: {tone})
5️⃣ Which {platform}-specific strategies will make it go viral

Then, provide the results in this exact format (keep structure identical):

### 🎬 Video Summary
📏 Duration: {round(duration, 2)}s | {width}x{height}px | {round(fps)}fps  
💡 Visual Tone: {tone} | Brightness: {brightness}

### 💬 AI-Generated Viral Insights:
1️⃣ **Scroll-Stopping Caption ({platform} only)**
   [one caption for {platform}]

2️⃣ **5 Hashtags ({platform} only)**
   [list exactly 5 hashtags]

3️⃣ **Engagement Tip**
   [one short actionable tip for {platform} creators]

4️⃣ **Viral Optimization Score (1–100)**
   [give a realistic score]

5️⃣ **Motivational Tip**
   [inspire the creator in one line]

### 🔥 Viral Comparison:
Provide **3 real public viral videos on {platform}** in the same niche.
For each:
- URL or unique identifier of the video  
- Summary of what happens  
- What made it go viral  
- How to replicate it in your video

### 🛠 What You Can Do to Rank Higher:
Based on your uploaded video and the above viral examples, give **3–5 specific optimization actions** (editing, hook, sound, length, caption, hashtags, first 2 seconds, etc.) tailored for {platform} and the detected niche.

Finally:
🎯 **Detected Niche:** [insert niche]
🕓 **Best Time to Post for [niche] ({platform}, {current_day}, EST)**:
⏰ [insert best time window]
💡 Peak engagement around [insert specific time].
"""
                    }
                ],
                temperature=0.7,
                max_tokens=1200
            )

        # Extract the AI-generated text
        ai_text = analysis.choices[0].message.content

        return jsonify({"ai_results": ai_text})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
