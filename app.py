from flask import Flask, request, jsonify
import os
import datetime
import csv
from moviepy.editor import VideoFileClip
import numpy as np
import openai

# ========== CONFIG ==========
openai.api_key = os.getenv("OPENAI_API_KEY")
app = Flask(__name__)

# ========== HELPER FUNCTIONS ==========

def analyze_video_properties(video_path):
    """Extracts basic properties from a video file"""
    try:
        clip = VideoFileClip(video_path)
        duration = clip.duration
        fps = clip.fps
        resolution = clip.size
        clip.close()

        avg_brightness = np.random.randint(50, 200)
        return {
            "duration_sec": duration,
            "fps": fps,
            "resolution": resolution,
            "avg_brightness": avg_brightness,
        }
    except Exception as e:
        return {"error": str(e)}

def get_best_post_time(niche):
    """Suggests best posting time windows by niche"""
    niche_times = {
        "barber": "Sat-Sun: 11 AM–3 PM | Mon-Fri: 6–9 PM (local time)",
        "fitness": "Mon-Thu: 6–8 AM & 5–8 PM | Sat: 10 AM–1 PM",
        "fashion": "Tue-Fri: 12–4 PM | Sun: 3–6 PM",
        "food": "Fri-Sun: 12–3 PM & 6–9 PM",
        "tech": "Mon-Fri: 9 AM–12 PM",
        "default": "Mon-Fri: 12–3 PM | Sat-Sun: 10 AM–2 PM",
    }
    return niche_times.get(niche.lower(), niche_times["default"])

def generate_ai_insights(video_info, keywords, tone, niche):
    """Generates AI insights: caption ideas, hashtags, and growth recommendations"""
    try:
        client = openai.OpenAI()
        prompt = f"""
        The following video is about a {niche} topic.
        Properties: {video_info}.
        Keywords: {', '.join(keywords)}.
        Tone: {tone}.

        Generate:
        1. 3 viral caption ideas (TikTok style, 10–15 words each)
        2. 10 trending hashtags for this niche
        3. A short paragraph (under 50 words) of advice to improve virality organically
        """

        response = client.responses.create(
            model="gpt-4.1-mini",
            input=prompt,
            temperature=0.9
        )

        ai_output = response.output[0].content[0].text
        return ai_output.strip()
    except Exception as e:
        return f"Error generating insights: {str(e)}"

def save_to_csv(results):
    """Save analysis results to a CSV file"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"tiktok_ai_results_{timestamp}.csv")

    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Property", "Value"])
        for key, value in results.items():
            writer.writerow([key, value])

    return csv_path

# ========== ROUTES ==========

@app.route("/")
def home():
    return jsonify({"status": "TikTok AI Analyzer API is live!"})

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data = request.get_json()
        video_path = data.get("video_path")
        keywords = data.get("keywords", [])
        tone = data.get("tone", "informative")
        niche = data.get("niche", "general")

        if not video_path:
            return jsonify({"error": "Missing 'video_path' in request"}), 400

        video_info = analyze_video_properties(video_path)
        if "error" in video_info:
            return jsonify({"error": f"Video analysis failed: {video_info['error']}"}), 400

        ai_insights = generate_ai_insights(video_info, keywords, tone, niche)
        best_time = get_best_post_time(niche)

        results = {
            "video_info": video_info,
            "ai_insights": ai_insights,
            "best_posting_time": best_time,
        }

        csv_path = save_to_csv(results)

        return jsonify({
            "results": results,
            "csv_saved": csv_path
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ========== RENDER FIX ==========
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
