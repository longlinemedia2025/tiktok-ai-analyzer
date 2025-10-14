from flask import Flask, request, jsonify, render_template
import os
import datetime
import csv
from moviepy.editor import VideoFileClip
from openai import OpenAI

# ========== CONFIG ==========
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
app = Flask(__name__)

# ========== HELPER FUNCTIONS ==========

def analyze_video_properties(video_path):
    """Extracts video duration, resolution, and frame rate."""
    clip = VideoFileClip(video_path)
    duration = clip.duration
    width, height = clip.size
    fps = clip.fps
    clip.close()

    return {
        "duration_seconds": round(duration, 2),
        "resolution": f"{width}x{height}",
        "frame_rate": round(fps, 2)
    }

def generate_ai_analysis(video_info):
    """Generate captions and hashtags using OpenAI API."""
    prompt = f"""
    The following video has these properties:
    Duration: {video_info['duration_seconds']} seconds
    Resolution: {video_info['resolution']}
    Frame rate: {video_info['frame_rate']} fps

    Generate:
    1. 3 creative TikTok caption ideas
    2. 5 trending hashtags that fit the vibe
    3. A short 1-sentence summary of the video type
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a social media strategist for viral TikToks."},
                {"role": "user", "content": prompt}
            ]
        )

        text = response.choices[0].message.content
        return {"ai_suggestions": text.strip()}

    except Exception as e:
        return {"error": str(e)}

# ========== ROUTES ==========

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze_video():
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    video_file = request.files["video"]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = os.path.join("uploads", f"video_{timestamp}.mp4")

    os.makedirs("uploads", exist_ok=True)
    video_file.save(video_path)

    # Analyze the video
    video_info = analyze_video_properties(video_path)
    ai_results = generate_ai_analysis(video_info)

    # Merge data
    result = {**video_info, **ai_results}

    # Save to CSV log
    log_path = "video_analysis_log.csv"
    file_exists = os.path.isfile(log_path)
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=result.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)

    return jsonify(result)

# ========== RUN LOCAL ==========
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
