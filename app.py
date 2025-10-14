from flask import Flask, request, jsonify
import os
import datetime
import openai
from moviepy.editor import VideoFileClip
from PIL import Image
import io
import base64
import random
import numpy as np
import time

# ========== CONFIG ==========
openai.api_key = os.getenv("OPENAI_API_KEY")
app = Flask(__name__)

# ========== HELPER FUNCTIONS ==========

def analyze_video_properties(video_path):
    """Extracts duration, resolution, brightness, tone, and heuristic score."""
    print("📊 Step 1: Analyzing video properties...")
    clip = VideoFileClip(video_path)
    duration = clip.duration
    width, height = clip.size
    aspect_ratio = width / height if height != 0 else 1

    # Sample brightness
    frame = clip.get_frame(duration / 2)
    brightness = np.mean(frame)
    tone = "bright" if brightness > 150 else "dark" if brightness < 80 else "neutral or mixed"

    # Simple heuristic score
    score = 9 if brightness > 50 and width >= 720 else 7

    clip.close()
    print("✅ Step 1 complete: Basic properties extracted.")
    return duration, width, height, aspect_ratio, brightness, tone, score


def describe_video_content(video_path, sample_count=3):
    """Grabs random frames and encodes them as a minimal visual summary."""
    print("🧩 Step 2: Sampling video frames for content analysis...")
    clip = VideoFileClip(video_path)
    duration = int(clip.duration)
    timestamps = sorted(random.sample(range(duration), min(sample_count, max(1, duration))))
    frame_descriptions = []

    for t in timestamps:
        try:
            frame = clip.get_frame(t)
            img = Image.fromarray(frame)
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG")
            img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
            frame_descriptions.append(f"[Frame {t}s base64 preview: {img_str[:120]}...]")
        except Exception as e:
            frame_descriptions.append(f"[Frame {t}s could not be read: {e}]")

    clip.close()
    print("✅ Step 2 complete: Visual summary generated.")
    return " | ".join(frame_descriptions)


def generate_ai_analysis(video_path, video_name, duration, width, height, aspect_ratio, brightness, tone, score):
    """Generates AI analysis via OpenAI GPT."""
    print("🤖 Step 3: Describing video visuals for niche detection...")
    visual_summary = describe_video_content(video_path)

    print("🧠 Step 4: Sending analysis request to OpenAI...")
    prompt = f"""
You are an expert TikTok strategist and content analyzer.

You are given:
1. Basic video properties (duration, resolution, brightness, etc.)
2. A visual summary of what’s inside the video.
3. The file name: "{video_name}" (which may not always match the actual niche).

⚠️ Important:
Do NOT assume the niche solely from the file name.
Base insights mainly on what is seen and heard in the actual content.

Generate results in this **exact format**:

🎬 Drag and drop your TikTok video file here: "{video_path}"
🎥 Running TikTok Viral Optimizer...

🤖 Generating AI-powered analysis, captions, and viral tips...

🔥 Fetching viral video comparisons and strategic insights...

✅ TikTok Video Analysis Complete!

🎬 Video: {video_name}
📏 Duration: {duration:.2f}s
🖼 Resolution: {width}x{height}
📱 Aspect Ratio: {aspect_ratio:.3f}
💡 Brightness: {brightness:.2f}
🎨 Tone: {tone}
⭐ Heuristic Score: {score}/10

💬 AI-Generated Viral Insights:
### 1. Scroll-Stopping Caption
...

### 2. 5 Viral Hashtags
...

### 3. Actionable Improvement Tip for Engagement
...

### 4. Viral Optimization Score (1–100)
...

### 5. Short Motivation on How to Increase Virality
...

🔥 Viral Comparison Results:
...

📋 Actionable Checklist:
   - Hook viewers in under 2 seconds.
   - Add trending sound if relevant.
   - Post during high activity times (Fri–Sun, 6–10pm).
   - Encourage comments by asking a question.

Now analyze using the visuals below:

Visual Summary:
{visual_summary}
"""

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1400,
        temperature=0.8
    )

    print("✅ Step 4 complete: AI analysis received successfully.")
    return response.choices[0].message.content


# ========== ROUTES ==========

@app.route("/analyze", methods=["POST"])
def analyze_video():
    if "video" not in request.files:
        return jsonify({"error": "No video file uploaded"}), 400

    file = request.files["video"]
    video_path = os.path.join("uploads", file.filename)
    os.makedirs("uploads", exist_ok=True)
    file.save(video_path)

    print(f"🎬 Starting TikTok AI analysis for: {file.filename}")
    start_time = time.time()

    try:
        duration, width, height, aspect_ratio, brightness, tone, score = analyze_video_properties(video_path)
        ai_analysis = generate_ai_analysis(
            video_path, file.filename, duration, width, height, aspect_ratio, brightness, tone, score
        )
        elapsed = time.time() - start_time
        print(f"✅ Full analysis complete in {elapsed:.2f} seconds.")
        return jsonify({"result": ai_analysis, "time_taken": elapsed})
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/")
def home():
    return "✅ TikTok AI Analyzer is running and ready!"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
