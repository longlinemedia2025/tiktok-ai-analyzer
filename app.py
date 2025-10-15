from flask import Flask, request, jsonify, render_template
import os
import tempfile
import subprocess
from moviepy.editor import VideoFileClip
from openai import OpenAI
import numpy as np

# ======== CONFIG ========
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
app = Flask(__name__)

# Limit max upload size to ~100MB
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024


# ======== VALIDATION ========
def is_valid_video(video_path):
    """
    Use ffprobe to check if the video is readable and has a duration.
    """
    try:
        cmd = [
            "ffprobe", "-v", "error", "-show_entries",
            "format=duration", "-of",
            "default=noprint_wrappers=1:nokey=1", video_path
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        duration_str = result.stdout.strip()
        if not duration_str:
            return False
        duration = float(duration_str)
        return duration > 0
    except Exception:
        return False


# ======== VIDEO ANALYSIS ========
def analyze_video_properties(video_path):
    try:
        clip = VideoFileClip(video_path)
        duration = round(clip.duration, 2)
        width, height = clip.size
        aspect_ratio = round(width / height, 3)
        fps = round(clip.fps, 2)

        frame = clip.get_frame(0)
        avg_brightness = float(np.mean(frame))
        colorfulness = float(np.std(frame))
        clip.close()

        return {
            "duration": duration,
            "width": width,
            "height": height,
            "aspect_ratio": aspect_ratio,
            "fps": fps,
            "brightness": avg_brightness,
            "colorfulness": colorfulness
        }
    except Exception as e:
        return {"error": f"Video analysis failed: {str(e)}"}


# ======== ROUTES ========

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    if "video" not in request.files:
        return jsonify({"error": "No video uploaded."}), 400

    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    # Save temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        file.save(tmp.name)
        video_path = tmp.name

    try:
        # Validate
        if not is_valid_video(video_path):
            os.remove(video_path)
            return jsonify({"error": "Invalid or corrupted video file. Please re-export or try again."}), 400

        # Analyze visuals
        analysis = analyze_video_properties(video_path)
        if "error" in analysis:
            return jsonify(analysis), 500

        # === ORIGINAL PROMPT FORMAT ===
        prompt = f"""
You are a TikTok algorithm analysis assistant.

Analyze this video based on the following:
- Brightness: {analysis['brightness']}
- Color intensity: {analysis['colorfulness']}
- Duration: {analysis['duration']}s
- Resolution: {analysis['width']}x{analysis['height']}
- Aspect Ratio: {analysis['aspect_ratio']}
- FPS: {analysis['fps']}

Generate a full, detailed response in this **exact format**:

🎬 Drag and drop your TikTok video file here: "{file.filename}"
🎥 Running TikTok Viral Optimizer...

🤖 Generating AI-powered analysis, captions, and viral tips...

🔥 Fetching viral video comparisons and strategic insights...

✅ TikTok Video Analysis Complete!

🎬 Video: {file.filename}
📏 Duration: {analysis['duration']}s
🖼 Resolution: {analysis['width']}x{analysis['height']}
📱 Aspect Ratio: {analysis['aspect_ratio']}
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

        # Call OpenAI API
        ai_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
            max_tokens=900
        )

        ai_text = ai_response.choices[0].message.content.strip()

        return jsonify({
            "success": True,
            "analysis": analysis,
            "ai_results": ai_text
        })

    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

    finally:
        if os.path.exists(video_path):
            os.remove(video_path)


# ======== MAIN ========
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=True)
