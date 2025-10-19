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
import json

app = Flask(__name__, template_folder="templates")
CORS(app)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def analyze_video_properties(video_path):
    """Extract basic visual properties from video."""
    clip = VideoFileClip(video_path)
    duration = round(clip.duration, 2)
    sample_time = min(max(0.1, clip.duration / 2), clip.duration - 0.1)
    frame = clip.get_frame(sample_time)
    try:
        frame_arr = (frame * 255).astype(np.uint8)
    except Exception:
        frame_arr = frame.astype(np.uint8)
    height, width = frame_arr.shape[0], frame_arr.shape[1]
    aspect_ratio = round(width / height, 3) if height != 0 else 0
    try:
        gray = cv2.cvtColor(frame_arr, cv2.COLOR_RGB2GRAY)
    except Exception:
        gray = cv2.cvtColor(frame_arr, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    tone = "bright and energetic" if brightness > 100 else "dark and moody"

    return {
        "duration": duration,
        "resolution": f"{width}x{height}",
        "aspect_ratio": aspect_ratio,
        "brightness": round(brightness, 2),
        "tone": tone,
    }


def generate_best_post_time(platform, niche):
    """Generate realistic best posting time dynamically using niche, platform, and day/time behavior."""
    now = datetime.datetime.now()
    day = now.strftime("%a")
    hour = now.hour

    post_patterns = {
        "TikTok": ("6–10 PM EST", f"8:{random.randint(10,55):02d} PM EST"),
        "YouTube": ("7–10 PM EST", f"8:{random.randint(15,59):02d} PM EST"),
        "Instagram": ("5–8 PM EST", f"6:{random.randint(20,50):02d} PM EST"),
        "Facebook": ("11 AM–2 PM EST", f"12:{random.randint(5,55):02d} PM EST"),
    }

    niche_lower = (niche or "").lower()
    dynamic_time = ""

    if "gaming" in niche_lower:
        dynamic_time = ("Fri" if day not in ["Fri", "Sat"] else day, "7–10 PM EST", f"8:{random.randint(15,55):02d} PM EST")
    elif "business" in niche_lower or "education" in niche_lower:
        dynamic_time = (day, "12–3 PM EST", f"1:{random.randint(10,50):02d} PM EST")
    elif "fitness" in niche_lower or "motivation" in niche_lower:
        dynamic_time = (day, "6–9 AM EST", f"7:{random.randint(0,45):02d} AM EST")
    elif "lifestyle" in niche_lower or "fashion" in niche_lower:
        dynamic_time = (day, "9 AM–12 PM EST", f"10:{random.randint(0,50):02d} AM EST")
    elif "news" in niche_lower or "current events" in niche_lower:
        dynamic_time = (day, "8–11 AM EST", f"9:{random.randint(10,55):02d} AM EST")
    else:
        dynamic_time = (day, *post_patterns.get(platform, ("6–9 PM EST", f"7:{random.randint(10,59):02d} PM EST")))

    day, window, peak = dynamic_time
    return f"⏰ {day} {window}\n💡 Peak engagement around {peak}"


def generate_ai_analysis(video_props, platform, video_name):
    """Generate AI analysis using OpenAI API tuned per platform (including Facebook)."""
    platform_label = platform
    if platform_label.lower() == "tiktok":
        algo_focus = (
            "Analyze this TikTok video using TikTok’s algorithmic preferences: "
            "short loops, early engagement, strong hook within first 2 seconds, trending sounds, and native features."
        )
        tag_label = "Hashtags"
        insights_label = "AI-Generated Viral Insights"
    elif platform_label.lower() == "youtube":
        algo_focus = (
            "Analyze this YouTube video using YouTube’s algorithmic priorities: "
            "click-through rate, watch time, audience retention, SEO title/description, and suggested feed optimization."
        )
        tag_label = "Tags"
        insights_label = "AI-Generated Viral Insights"
    elif platform_label.lower() == "instagram":
        algo_focus = (
            "Analyze this Instagram Reel using Instagram's algorithmic signals: "
            "early saves/shares, Reels completion rate, trending audio, and Explore/Hashtag discoverability."
        )
        tag_label = "Hashtags"
        insights_label = "Reels-Focused Viral Insights"
    elif platform_label.lower() == "facebook":
        algo_focus = (
            "Analyze this Facebook Reel/Post according to Facebook's algorithm: "
            "prioritize native video completion, early reactions & shares, group/community amplification, "
            "and content that sparks conversation. Include timing suggestions using recent engagement behavior "
            "for the detected niche and day of week."
        )
        tag_label = "Tags/Hashtags"
        insights_label = "Facebook-Specific Viral Insights"
    else:
        algo_focus = "Analyze this video for general social media virality."
        tag_label = "Tags"
        insights_label = "AI-Generated Viral Insights"

    prompt = f"""
{algo_focus}

Create a viral video analysis for {platform_label} with this structure EXACTLY:

🎬 Drag and drop your {platform_label} video file here: "{video_name}"
🎥 Running {platform_label} Viral Optimizer...

🤖 Generating AI-powered analysis, captions, and viral tips...

🔥 Fetching viral video comparisons and strategic insights...

✅ {platform_label} Video Analysis Complete!

🎬 Video: {video_name}
📏 Duration: {video_props['duration']}s
🖼 Resolution: {video_props['resolution']}
📱 Aspect Ratio: {video_props['aspect_ratio']}
💡 Brightness: {video_props['brightness']}
🎨 Tone: {video_props['tone']}
⭐ Heuristic Score: 8/10 (High brightness and clear resolution contribute to visual appeal.)

🎯 Detected Attributes:
- Niche: (detect from content)
- Tone: {video_props['tone']}
- Keywords: (detect likely keywords from content)

💬 {insights_label}:
### 1. Scroll-Stopping Caption
(Provide one creative, high-performing caption idea tailored to {platform_label}.)

### 2. 5 Viral {tag_label}
(Provide 5 {platform_label}-relevant {tag_label.lower()} based on content and niche.)

### 3. Actionable Improvement Tip for Engagement
(Provide a short, actionable suggestion tuned for {platform_label}’s algorithm.)

### 4. Viral Optimization Score (1–100)
(Give a numeric score and short explanation.)

### 5. Motivation to Increase Virality
(Give platform-specific encouragement or strategy tip.)

🔥 Viral Comparison Results:
### Comparison with Viral {platform_label} Videos in the Same Niche
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
(Summarize actionable insights for {platform_label} creators)

📋 Actionable Checklist:
- Hook viewers in the first 2 seconds.
- Use trending audio and relevant captions.
- Encourage saves and shares with call-to-actions.
- Maintain visual consistency across posts.

🎯 **Detected Niche:** (detected niche)
🕓 **Best Time to Post for {platform_label} (by niche)**:
⏰ (Day + time range in EST)
💡 Peak engagement around (specific time in EST)
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.9,
            max_tokens=1200,
        )
        text = response.choices[0].message.content.strip()
    except Exception as e:
        text = f"(AI generation failed: {str(e)})\n\nPlease retry."

    return text


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        platform = request.form.get("platform", "TikTok")
        video = request.files.get("video")

        if not video:
            return jsonify({"error": "No video uploaded"}), 400

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp:
            video.save(temp.name)
            video_path = temp.name

        props = analyze_video_properties(video_path)
        ai_text = generate_ai_analysis(props, platform, video.filename)

        niche_match = re.search(r"Detected Niche:\s*(.+)", ai_text)
        niche = (niche_match.group(1).strip() if niche_match else "General")

        best_time_text = generate_best_post_time(platform, niche)

        hashtags_list = re.findall(r"#\w+", ai_text)
        score_match = re.search(r"(\d{1,3})\s*(?:/100|\%)", ai_text)
        score = score_match.group(1) if score_match else None

        final_output = f"""
AI Results
🎬 Drag and drop your {platform} video file here: "{video.filename}"
🎥 Running {platform} Viral Optimizer...

🤖 Generating AI-powered analysis, captions, and viral tips...

🔥 Fetching viral video comparisons and strategic insights...

✅ {platform} Video Analysis Complete!

🎬 Video: {video.filename}
📏 Duration: {props['duration']}s
🖼 Resolution: {props['resolution']}
📱 Aspect Ratio: {props['aspect_ratio']}
💡 Brightness: {props['brightness']}
🎨 Tone: {props['tone']}
⭐ Heuristic Score: {score or '8'}/10

💬 AI-Generated Viral Insights:
{ai_text.split('💬',1)[-1] if '💬' in ai_text else ai_text}

🕓 **Best Time to Post for {niche} ({platform})**:
{best_time_text}
"""

        final_output = re.sub(r"===JSON===.*", "", final_output, flags=re.DOTALL).strip()

        return jsonify({"result": final_output})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
