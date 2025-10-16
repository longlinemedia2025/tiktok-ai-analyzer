from flask import Flask, request, jsonify, render_template
import os
import datetime
import csv
from moviepy.editor import VideoFileClip
import numpy as np
import openai

# ========== CONFIG ==========
openai.api_key = os.getenv("OPENAI_API_KEY")
app = Flask(__name__, template_folder="templates")

# ========== HELPER FUNCTIONS ==========

def analyze_video_properties(video_path):
    """Extracts duration, resolution, brightness, etc."""
    clip = VideoFileClip(video_path)
    duration = clip.duration
    resolution = clip.size
    aspect_ratio = resolution[0] / resolution[1]
    frame = clip.get_frame(0)
    brightness = np.mean(frame)
    clip.close()
    return {
        "duration_seconds": round(duration, 2),
        "resolution": resolution,
        "aspect_ratio": round(aspect_ratio, 3),
        "mean_brightness_first_frame": round(brightness, 2)
    }


def heuristic_score(data):
    """Simple scoring logic."""
    score = 0
    notes = []

    # Duration heuristic
    if data["duration_seconds"] <= 15:
        score += 3
        notes.append("Short (<=15s): great for high completion rate.")
    elif data["duration_seconds"] <= 30:
        score += 2
        notes.append("Medium (<=30s): acceptable, slightly less retention.")
    else:
        notes.append("Long (>30s): lower retention risk.")

    # Aspect ratio heuristic
    if abs(data["aspect_ratio"] - 9/16) < 0.05:
        score += 3
        notes.append("Vertical (9:16): ideal for short-form mobile content.")
    else:
        notes.append("Non-vertical aspect ratio: may underperform.")

    # Brightness heuristic
    if data["mean_brightness_first_frame"] > 80:
        score += 2
        notes.append("Bright visuals: great for attention.")
    else:
        score += 1
        notes.append("Average brightness: acceptable.")

    return {
        "raw_score": score,
        "normalized_0_10": min(round((score / 8) * 10), 10),
        "notes": notes
    }


def generate_ai_insights(platform, video_meta, keywords, tone, niche):
    """Generate captions/tags/strategy tuned for TikTok or YouTube."""
    try:
        if platform == "tiktok":
            prompt = f"""
Analyze a TikTok video for virality potential in the {niche} niche.
Video stats:
- Duration: {video_meta['duration_seconds']}s
- Aspect Ratio: {video_meta['aspect_ratio']}
- Brightness: {video_meta['mean_brightness_first_frame']}
- Tone: {tone}
- Keywords: {', '.join(keywords)}

Return formatted TikTok-style text with:
1. Scroll-stopping caption
2. 5 viral hashtags
3. Engagement tip
4. Viral optimization score (1–100)
5. Motivation tip
6. 3 example viral TikToks with analysis
7. Takeaway strategy and posting recommendations
"""
        else:
            prompt = f"""
Analyze a YouTube video for virality potential in the {niche} niche.
Video stats:
- Duration: {video_meta['duration_seconds']}s
- Aspect Ratio: {video_meta['aspect_ratio']}
- Brightness: {video_meta['mean_brightness_first_frame']}
- Tone: {tone}
- Keywords: {', '.join(keywords)}

Return formatted YouTube-style text with:
1. Scroll-stopping title
2. 5 viral tags
3. Engagement improvement tip
4. Viral optimization score (1–100)
5. Motivation for creators
6. 3 example viral YouTube videos in same niche with insights
7. Takeaway strategy and upload time recommendations
"""

        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert in social media virality and short-form video optimization."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"(AI generation failed: {e})"


# ========== ROUTES ==========

@app.route('/')
def index():
    """Render the homepage with drag/drop upload UI."""
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze_video():
    data = request.form
    platform = data.get("platform", "tiktok").lower()
    video_file = request.files.get("video")
    csv_file = request.files.get("csv")
    keywords = data.get("keywords", "").split(",")
    tone = data.get("tone", "neutral")
    niche = data.get("niche", "general")

    if not video_file:
        return jsonify({"error": "No video file uploaded."}), 400

    # Save temporary video
    video_path = os.path.join("uploads", video_file.filename)
    os.makedirs("uploads", exist_ok=True)
    video_file.save(video_path)

    # If CSV is uploaded, we can parse it for context (optional future expansion)
    csv_data = None
    if csv_file:
        csv_path = os.path.join("uploads", csv_file.filename)
        csv_file.save(csv_path)
        csv_data = csv_path  # For now, we just confirm it's received

    # Step 1: Analyze video
    meta = analyze_video_properties(video_path)
    heuristics = heuristic_score(meta)
    meta["heuristics"] = heuristics

    # Step 2: Generate AI insights
    ai_output = generate_ai_insights(platform, meta, keywords, tone, niche)

    # Step 3: Build results
    results = {
        "platform": platform,
        "video": video_file.filename,
        "meta": meta,
        "ai_insights": ai_output,
        "csv_used": bool(csv_file),
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    # Step 4: Save CSV output for records
    os.makedirs("output", exist_ok=True)
    csv_filename = f"output/{platform}_ai_results_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
    with open(csv_filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Platform", "Filename", "Duration (s)", "Resolution", "Aspect Ratio", "Brightness", "Score", "AI Insights"])
        writer.writerow([
            platform,
            results["video"],
            meta["duration_seconds"],
            f"{meta['resolution'][0]}x{meta['resolution'][1]}",
            meta["aspect_ratio"],
            meta["mean_brightness_first_frame"],
            heuristics["normalized_0_10"],
            ai_output
        ])

    return jsonify({
        "status": "success",
        "results": results,
        "csv_saved": csv_filename
    })


# ========== MAIN ==========
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
