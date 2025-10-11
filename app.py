from flask import Flask, request, jsonify
import os
import datetime
import csv
from moviepy.editor import VideoFileClip
import numpy as np
import openai

# ========== CONFIG ==========
openai.api_key = os.getenv("OPENAI_API_KEY")  # Make sure this is set
app = Flask(__name__)

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
    """Applies simple scoring logic."""
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
        notes.append("Vertical (9:16): ideal for TikTok mobile full-screen.")
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

def generate_ai_insights(video_meta, keywords, tone, niche):
    """Uses OpenAI to generate caption, hashtags, and posting recommendations."""
    try:
        prompt = f"""
Analyze a TikTok video for virality potential in the {niche} niche.
Video stats:
- Duration: {video_meta['duration_seconds']} seconds
- Aspect Ratio: {video_meta['aspect_ratio']}
- Brightness: {video_meta['mean_brightness_first_frame']}
- Tone: {tone}
- Keywords: {', '.join(keywords)}

Return a short JSON with:
1. caption
2. hashtags (5 viral ones)
3. posting_times (best times to post for this niche)
4. one engagement_tip
"""
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a TikTok virality expert and content strategist."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"(AI generation failed: {e})"

# ========== ROUTE ==========

@app.route('/analyze', methods=['POST'])
def analyze_video():
    """
    POST JSON:
    {
        "video_path": "path/to/video.mp4",
        "keywords": ["haircut", "transformation"],
        "tone": "casual",
        "niche": "barber"
    }
    """
    data = request.get_json()
    video_path = data.get("video_path")
    keywords = data.get("keywords", [])
    tone = data.get("tone", "neutral")
    niche = data.get("niche", "general")

    if not os.path.exists(video_path):
        return jsonify({"error": f"Video file not found: {video_path}"}), 400

    # Step 1: Analyze video
    meta = analyze_video_properties(video_path)
    heuristics = heuristic_score(meta)
    meta["heuristics"] = heuristics

    # Step 2: Generate AI insights
    ai_output = generate_ai_insights(meta, keywords, tone, niche)

    # Step 3: Build results
    results = {
        "video": os.path.basename(video_path),
        "meta": meta,
        "ai_insights": ai_output,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    # Step 4: Save to CSV
    os.makedirs("output", exist_ok=True)
    csv_filename = f"output/tiktok_ai_results_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
    with open(csv_filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Filename", "Duration (s)", "Resolution", "Aspect Ratio", "Brightness", "Score", "AI Insights"])
        writer.writerow([
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
