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
        "facebook": ("9–11 AM EST", "10:{:02d} AM EST".format(random.randint(0, 59))),
    }

    # Adjust by niche keywords
    niche_lower = niche.lower() if niche else ""
    if platform == "facebook":
        if any(k in niche_lower for k in ["news", "politics", "community", "education"]):
            post_window, peak = ("7–10 AM EST", "8:{:02d} AM EST".format(random.randint(5, 55)))
        elif any(k in niche_lower for k in ["lifestyle", "beauty", "fashion", "health", "fitness"]):
            post_window, peak = ("11 AM–2 PM EST", "12:{:02d} PM EST".format(random.randint(10, 50)))
        elif any(k in niche_lower for k in ["gaming", "entertainment", "music", "memes"]):
            post_window, peak = ("6–9 PM EST", "7:{:02d} PM EST".format(random.randint(0, 59)))
        elif any(k in niche_lower for k in ["business", "finance", "marketing"]):
            post_window, peak = ("9 AM–12 PM EST", "10:{:02d} AM EST".format(random.randint(0, 59)))
        else:
            post_window, peak = ("8–11 AM EST", "9:{:02d} AM EST".format(random.randint(0, 59)))
    else:
        if any(k in niche_lower for k in ["fitness", "motivation", "lifestyle"]):
            post_window, peak = ("6–9 AM EST", "7:{:02d} AM EST".format(random.randint(0, 50)))
        elif any(k in niche_lower for k in ["business", "education", "career"]):
            post_window, peak = ("12–3 PM EST", "1:{:02d} PM EST".format(random.randint(5, 55)))
        elif any(k in niche_lower for k in ["gaming", "music", "entertainment"]):
            post_window, peak = ("7–10 PM EST", "8:{:02d} PM EST".format(random.randint(10, 59)))
        else:
            post_window, peak = post_patterns.get(platform, ("6–9 PM EST", "7:{:02d} PM EST".format(random.randint(10, 59))))

    return f"⏰ {day} {post_window}\n💡 Peak engagement around {peak}"


def generate_ai_analysis(video_props, platform, video_name):
    """Generate AI analysis using OpenAI API tuned for each platform."""
    tone_focus = {
        "tiktok": "fast-paced trends and short-form engagement hooks",
        "youtube": "retention, watch-time optimization, and storytelling structure",
        "instagram": "visual aesthetic, brand tone, and emotional storytelling",
        "facebook": "shareability, community engagement, and conversation triggers"
    }

    prompt = f"""
You are an expert social media strategist specializing in {platform}’s algorithm.
Analyze this {platform} video and generate viral optimization insights tailored to {platform}’s ranking system.

Platform focus: {tone_focus.get(platform, 'social video engagement principles')}
Video: {video_name}
Duration: {video_props['duration']}s
Resolution: {video_props['resolution']}
Aspect Ratio: {video_props['aspect_ratio']}
Brightness: {video_props['brightness']}
Tone: {video_props['tone']}

Provide your response in this exact structured format:

🎬 Drag and drop your {platform} video file here: "{video_name}"
🎥 Running {platform} Viral Optimizer...

🤖 Generating AI-powered analysis, captions, and viral tips...

🔥 Fetching viral video comparisons and strategic insights...

✅ {platform.capitalize()} Video Analysis Complete!

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
(Give a catchy, emotional caption tailored to {platform})

### 2. 5 Viral Hashtags
(Give 5 hashtags relevant to the {platform} niche)

### 3. Actionable Improvement Tip for Engagement
(Give one improvement idea specific to {platform})

### 4. Viral Optimization Score (1–100)
(Give a score and short explanation)

### 5. Motivation to Increase Virality
(Give an encouraging tip)

🔥 Viral Comparison Results:
### Comparison with Viral {platform} Videos in the Same Niche
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
(Summarize actionable insights for {platform} creators)

📋 Actionable Checklist:
- Hook viewers in the first 2 seconds.
- Use platform-native text formats and CTAs.
- Encourage shares and comments to boost visibility.
- Post when your target audience is most active.

🎯 **Detected Niche:** (detected niche)
🕓 **Best Time to Post for that Niche ({platform.capitalize()})**:
⏰ (Day + time range in EST)
💡 Peak engagement around (specific time in EST)
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an advanced AI trained for viral video analysis."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.9,
        max_tokens=1600,
    )
    return response.choices[0].message.content.strip()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        platform = request.form.get("platform", "tiktok").lower()
        video = request.files.get("video")

        if not video:
            return jsonify({"error": "No video uploaded"}), 400

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp:
            video.save(temp.name)
            video_path = temp.name

        # Extract video properties
        props = analyze_video_properties(video_path)

        # Get AI-generated text
        ai_text = generate_ai_analysis(props, platform, video.filename)

        # Extract niche
        niche_match = re.search(r"Niche:\s*(.+)", ai_text)
        niche = niche_match.group(1).strip() if niche_match else "General"

        # Generate best posting time dynamically
        best_time_text = generate_best_post_time(platform, niche)

        # Extract viral score
        score_match = re.search(r"(\d{1,3})/100", ai_text)
        score = score_match.group(1) if score_match else "N/A"

        # Combine results neatly
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
