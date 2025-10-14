from flask import Flask, request, jsonify, render_template_string
import os
import datetime
from moviepy.editor import VideoFileClip
import numpy as np
from openai import OpenAI

# ========== CONFIG ==========
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
app = Flask(__name__)

# ========== HTML FRONTEND ==========
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>TikTok Viral Optimizer</title>
    <style>
        body {
            background-color: #0d1117;
            color: #c9d1d9;
            font-family: Arial, sans-serif;
            text-align: center;
            padding: 40px;
        }
        h1 { color: #58a6ff; }
        .dropzone {
            margin: 30px auto;
            padding: 50px;
            width: 80%;
            max-width: 600px;
            border: 3px dashed #58a6ff;
            border-radius: 12px;
            background-color: #161b22;
        }
        #output {
            white-space: pre-wrap;
            text-align: left;
            margin-top: 20px;
            background: #161b22;
            border-radius: 12px;
            padding: 20px;
            border: 1px solid #30363d;
        }
    </style>
</head>
<body>
    <h1>🎬 TikTok Viral Optimizer</h1>
    <div class="dropzone" id="dropzone">Drag & Drop Your TikTok Video Here</div>
    <div id="output"></div>

    <script>
        const dropzone = document.getElementById("dropzone");
        const output = document.getElementById("output");

        dropzone.addEventListener("dragover", e => {
            e.preventDefault();
            dropzone.style.borderColor = "#00ff99";
        });

        dropzone.addEventListener("dragleave", () => {
            dropzone.style.borderColor = "#58a6ff";
        });

        dropzone.addEventListener("drop", async e => {
            e.preventDefault();
            dropzone.style.borderColor = "#58a6ff";

            const file = e.dataTransfer.files[0];
            const formData = new FormData();
            formData.append("video", file);

            output.innerText = "🎥 Running TikTok Viral Optimizer...";

            const response = await fetch("/analyze", {
                method: "POST",
                body: formData
            });

            const data = await response.json();
            output.innerText = JSON.stringify(data, null, 2);
        });
    </script>
</body>
</html>
"""

# ========== HELPER FUNCTIONS ==========
def analyze_video_properties(video_path):
    """Extract key metrics (duration, resolution, brightness)."""
    try:
        clip = VideoFileClip(video_path)
        duration = clip.duration
        resolution = f"{clip.w}x{clip.h}"
        fps = clip.fps
        frame = clip.get_frame(0.5)
        brightness = np.mean(frame)
        clip.close()
        return {
            "duration_seconds": round(duration, 2),
            "resolution": resolution,
            "frame_rate": round(fps, 2),
            "brightness": round(brightness, 2)
        }
    except Exception as e:
        return {"error": str(e)}

def generate_niche_analysis(video_metrics, filename):
    """AI-powered analysis with niche inference and viral insights."""
    # Derive simple cues from the filename for better context
    filename_lower = filename.lower()
    possible_niches = {
        "hair": "Barber / Hair Transformation",
        "cut": "Barber / Hairstyle",
        "fade": "Barber / Grooming",
        "gaming": "Gaming / Gameplay",
        "game": "Gaming / Esports",
        "makeup": "Beauty / Makeup Tutorial",
        "cook": "Cooking / Food Content",
        "food": "Cooking / Food",
        "motivation": "Motivational / Self-Improvement",
        "fitness": "Fitness / Gym Content",
        "vlog": "Lifestyle / Daily Vlog",
        "review": "Product Review / Unboxing",
        "dance": "Dance / Performance",
        "pet": "Animals / Pets",
        "dog": "Animals / Pets",
        "cat": "Animals / Pets"
    }

    detected_niche = "General / Undefined"
    for keyword, niche in possible_niches.items():
        if keyword in filename_lower:
            detected_niche = niche
            break

    prompt = f"""
You are TikTok’s top AI viral strategist. 
Analyze this video and generate an advanced, **niche-specific** TikTok optimization report.

🧠 The video details:
- File name: {filename}
- Auto-detected niche from filename: {detected_niche}
- Duration: {video_metrics.get('duration_seconds')}s
- Resolution: {video_metrics.get('resolution')}
- Brightness: {video_metrics.get('brightness')}
- Frame Rate: {video_metrics.get('frame_rate')}

🎯 Your tasks:
1. Determine the **most accurate niche** based on the filename and metrics.
2. Generate the output in the following **exact structured format**:

🎬 Video: {filename}
📏 Duration: {video_metrics.get('duration_seconds')}s
🖼 Resolution: {video_metrics.get('resolution')}
📱 Aspect Ratio: Describe (Vertical, Horizontal, or Square)
💡 Brightness: {video_metrics.get('brightness')}
⭐ Heuristic Score: (1–10, based on clarity & visual quality)

💬 AI-Generated Viral Insights:
### 1. Scroll-Stopping Caption
(Create a short, niche-relevant caption with emojis and emotional hook)

### 2. 5 Viral Hashtags
(Create 5 hashtags perfectly relevant to that niche)

### 3. Actionable Improvement Tip
(Give 1 powerful, niche-specific engagement improvement)

### 4. Viral Optimization Score (1–100)
(Provide score + short reasoning)

### 5. Motivation
(A short motivational paragraph about going viral in that niche)

🔥 Viral Comparison Results:
### Comparison with Viral TikToks in the [Detected Niche] Niche
(Give 3 realistic examples with why they went viral and how to replicate)

📋 Actionable Checklist:
   - Hook viewers in under 2 seconds
   - Add trending sound if relevant
   - Post during peak times
   - Include call-to-action (CTA)
   - Maintain authentic energy

Be **deeply specific to the detected niche** (e.g. for haircut videos, discuss fade techniques or transformations; for gaming, discuss suspense and commentary pacing, etc.)
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert viral strategist for TikTok and short-form video."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating analysis: {e}"

# ========== ROUTES ==========
@app.route("/")
def home():
    return render_template_string(HTML_PAGE)

@app.route("/analyze", methods=["POST"])
def analyze():
    if "video" not in request.files:
        return jsonify({"error": "No video uploaded."}), 400

    video = request.files["video"]
    filename = video.filename
    temp_path = os.path.join("/tmp", filename)
    video.save(temp_path)

    video_metrics = analyze_video_properties(temp_path)
    ai_output = generate_niche_analysis(video_metrics, filename)

    return jsonify({
        **video_metrics,
        "niche_analysis": ai_output
    })

# ========== MAIN ==========
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
