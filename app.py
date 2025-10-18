from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import cv2
import numpy as np
from moviepy.editor import VideoFileClip
from openai import OpenAI
import tempfile
import re
import datetime
import random

app = Flask(__name__, template_folder="templates")
CORS(app)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def analyze_video_properties(video_path):
    """Extract basic visual properties from video."""
    clip = VideoFileClip(video_path)
    duration = round(clip.duration, 2)
    frame = clip.get_frame(clip.duration / 2)
    height, width, _ = frame.shape
    aspect_ratio = round(width / height, 3)
    brightness = np.mean(cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY))
    tone = "bright and energetic" if brightness > 100 else "dark and moody"

    return {
        "duration": duration,
        "resolution": f"{width}x{height}",
        "aspect_ratio": aspect_ratio,
        "brightness": round(brightness, 2),
        "tone": tone,
    }


def generate_best_post_time(platform, niche):
    """Generate a realistic best posting time based on niche, platform, and day."""
    now = datetime.datetime.now()
    day = now.strftime("%a")

    # Base posting time patterns by platform
    post_patterns = {
        "tiktok": ("6–10 PM EST", "8:{:02d} PM EST".format(random.randint(10, 55))),
        "youtube": ("7–10 PM EST", "8:{:02d} PM EST".format(random.randint(15, 59))),
        "instagram": ("5–8 PM EST", "6:{:02d} PM EST".format(random.randint(20, 50))),
        "facebook": ("12–5 PM EST", "3:{:02d} PM EST".format(random.randint(10, 50))),
    }

    # Adjust by niche keywords — now also for Facebook
    niche_lower = niche.lower() if niche else ""
    if any(k in niche_lower for k in ["fitness", "motivation", "lifestyle"]):
        post_window, peak = ("6–9 AM EST", "7:{:02d} AM EST".format(random.randint(0, 50)))
    elif any(k in niche_lower for k in ["business", "education", "career"]):
        post_window, peak = ("12–3 PM EST", "1:{:02d} PM EST".format(random.randint(5, 55)))
    elif any(k in niche_lower for k in ["gaming", "music", "entertainment"]):
        post_window, peak = ("7–10 PM EST", "8:{:02d} PM EST".format(random.randint(10, 59)))
    elif any(k in niche_lower for k in ["news", "community", "discussion", "awareness"]):
        post_window, peak = ("9 AM–1 PM EST", "11:{:02d} AM EST".format(random.randint(0, 59)))
    else:
        post_window, peak = post_patterns.get(platform, ("6–9 PM EST", "7:{:02d} PM EST".format(random.randint(10, 59))))

    # Facebook weekday effect
    if platform == "facebook":
        weekday_boosts = {
            "Mon": "Professional / Educational reach highest mid-day",
            "Wed": "Community engagement spikes late morning",
            "Fri": "Entertainment & lifestyle perform best around 2–4 PM",
            "Sun": "Emotional or story-driven posts trend at noon",
        }
        extra_tip = weekday_boosts.get(day, "Consistent engagement expected during early afternoon.")
    else:
        extra_tip = "Optimize for your target region’s active hours."

    return f"⏰ {day} {post_window}\n💡 Peak engagement around {peak}\n📊 Tip: {extra_tip}"


def generate_ai_analysis(video_props, platform, video_name):
    """Generate AI analysis using OpenAI API."""
    platform_name = platform.capitalize()

    platform_color = {
        "tiktok": "#000000",
        "youtube": "#FF0000",
        "instagram": "#E1306C",
        "facebook": "#1877F2",
    }.get(platform, "#FFFFFF")

    prompt = f"""
You are an expert social media strategist. Analyze this {platform_name} video and generate viral optimization insights.
Use a tone and algorithmic strategy relevant to {platform_name}.
Platform color theme: {platform_color}

Video: {video_name}
Duration: {video_props['duration']}s
Resolution: {video_props['resolution']}
Aspect Ratio: {video_props['aspect_ratio']}
Brightness: {video_props['brightness']}
Tone: {video_props['tone']}

Provide a complete analysis in this exact structured format:

🎬 Drag and drop your {platform_name} video file here: "{video_name}"
🎥 Running {platform_name} Viral Optimizer...

🤖 Generating AI-powered analysis, captions, and viral tips...

🔥 Fetching viral video comparisons and strategic insights...

✅ {platform_name} Video Analysis Complete!

🎬 Video: {video_name}
📏 Duration: {video_props['duration']}s
🖼 Resolution: {video_props['resolution']}
📱 Aspect Ratio: {video_props['aspect_ratio']}
💡 Brightness: {video_props['brightness']}
🎨 Tone: {video_props['tone']}
⭐ Heuristic Score: 8/10 (High brightness and clear resolution contribute to visual appeal.)

🎯 Detected Attributes:
- Niche: (based on tone and content)
- Tone: {video_props['tone']}
- Keywords: (based on likely subject matter)

💬 AI-Generated Viral Insights:
### 1. Scroll-Stopping Caption
(Give a catchy, emotional caption)

### 2. 5 Viral Tags
(Give 5 tags relevant to {platform_name} niche)

### 3. Actionable Improvement Tip for Engagement
(Give one improvement idea)

### 4. Viral Optimization Score (1–100)
(Give a score and short explanation)

### 5. Motivation to Increase Virality
(Give an encouraging tip)

🔥 Viral Comparison Results:
### Comparison with Viral {platform_name} Videos in the Same Niche
#### Viral Example 1
- **Video Concept Summary:** ...
- **What Made It Go Viral:** ...
- **How to Replicate Success:** ...

#### Viral Example 2
- **Video Concept Summary:** ...
- **What Made It Go Viral:** ...
- **How to Replicate Success:** ...

#### Viral Example 3
- **Video Concept Summary:** ...
- **What Made It Go Viral:** ...
- **How to Replicate Success:** ...

### Takeaway Strategy
(Summarize actionable insights for {platform_name} creators)

📋 Actionable Checklist:
- Hook viewers in the first 2 seconds.
- Use trending audio and relevant captions.
- Encourage saves and shares with call-to-actions.
- Maintain visual consistency across uploads.

🎯 **Detected Niche:** (detected niche)
🕓 **Best Time to Post for that Niche ({platform_name})**:
⏰ (Day + time range in EST)
💡 Peak engagement around (specific time in EST)
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.9,
    )
    return response.choices[0].message.content.strip()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        platform = request.form.get("platform", "tiktok")
        video = request.files.get("video")

        if not video:
            return jsonify({"error": "No video uploaded"}), 400

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp:
            video.save(temp.name)
            video_path = temp.name

        props = analyze_video_properties(video_path)
        ai_text = generate_ai_analysis(props, platform, video.filename)

        niche_match = re.search(r"Niche:\s*(.+)", ai_text)
        niche = niche_match.group(1).strip() if niche_match else "General"

        best_time_text = generate_best_post_time(platform, niche)

        score_match = re.search(r"(\d{1,3})/100", ai_text)
        score = score_match.group(1) if score_match else "N/A"

        final_output = f"""
AI Results
🎬 Video Analyzed: "{video.filename}"

⭐ Viral Optimization Score: {score}/100

{ai_text}

🕓 **Best Time to Post for {niche} ({platform.capitalize()})**:
{best_time_text}
"""

        final_output = re.sub(r"===JSON===.*", "", final_output, flags=re.DOTALL)
        return jsonify({"result": final_output})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
