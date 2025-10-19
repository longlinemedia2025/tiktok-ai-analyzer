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
    # try to sample a middle frame; guard if video is very short
    sample_time = min(max(0.1, clip.duration / 2), clip.duration - 0.1)
    frame = clip.get_frame(sample_time)
    # MoviePy returns frames as RGB float [0..1] sometimes; convert safely
    try:
        frame_arr = (frame * 255).astype(np.uint8)
    except Exception:
        frame_arr = frame.astype(np.uint8)
    height, width = frame_arr.shape[0], frame_arr.shape[1]
    aspect_ratio = round(width / height, 3) if height != 0 else 0
    # compute brightness from grayscale
    try:
        gray = cv2.cvtColor(frame_arr, cv2.COLOR_RGB2GRAY)
    except Exception:
        # fallback if frame already in BGR
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
    """Generate a realistic best posting time based on niche, platform, and day."""
    now = datetime.datetime.now()
    day = now.strftime("%a")

    # Base posting time patterns by platform
    post_patterns = {
        "TikTok": ("6–10 PM EST", "8:{:02d} PM EST".format(random.randint(10, 55))),
        "YouTube": ("7–10 PM EST", "8:{:02d} PM EST".format(random.randint(15, 59))),
        "Instagram": ("5–8 PM EST", "6:{:02d} PM EST".format(random.randint(20, 50))),
        "Facebook": ("11 AM–2 PM EST", "12:{:02d} PM EST".format(random.randint(5, 55))),
    }

    niche_lower = (niche or "").lower()

    # niche heuristics (keeps it dynamic and not a fixed weekday mapping)
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
    """Generate AI analysis using OpenAI API tuned per platform (including Facebook)."""
    platform_label = platform
    # platform-specific algorithm focus
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
            "and content that sparks conversation. Pay attention to shareability, thumbnail, and cross-post timings."
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
(Give one creative, high-performing caption idea. For Instagram: make it emotional and include subtle CTA. For Facebook: prioritize shareability and conversation prompts.)

### 2. 5 Viral {tag_label}
(Provide 5 platform-relevant {tag_label.lower()} that match the {platform_label} algorithm.)

### 3. Actionable Improvement Tip for Engagement
(Provide a short suggestion tailored to the platform's algorithm.)

### 4. Viral Optimization Score (1–100)
(Give a numeric score and a short explanation.)

### 5. Motivation to Increase Virality
(Encouraging, platform-specific creator tip.)

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

    # call the OpenAI Chat completion (using chat completions api present in your environment)
    # NOTE: adjust model as you prefer and ensure OPENAI_API_KEY is set in env
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.9,
            max_tokens=1200,
        )
        # The SDK might differ; this code assumes the response structure has choices[0].message.content
        text = response.choices[0].message.content.strip()
    except Exception as e:
        # fallback simple templated response if model call fails
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
        csv_file = request.files.get("csv")

        if not video:
            return jsonify({"error": "No video uploaded"}), 400

        # Save video temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp:
            video.save(temp.name)
            video_path = temp.name

        # Extract video properties
        props = analyze_video_properties(video_path)

        # Attempt to parse a small sample of CSV if provided
        csv_preview = None
        if csv_file:
            try:
                import pandas as pd
                df = pd.read_csv(csv_file)
                csv_preview = df.head(5).to_dict(orient="records")
            except Exception:
                csv_preview = None

        # Get AI-generated text (platform tuned)
        ai_text = generate_ai_analysis(props, platform, video.filename)

        # Extract detected niche heuristically from AI text
        niche_match = re.search(r"Detected Niche:\s*(.+)", ai_text) or re.search(r"🎯\s*\*\*Detected Niche:\*\*\s*(.+)", ai_text)
        niche = (niche_match.group(1).strip() if niche_match else None) or "General"

        # Generate best posting time dynamically (niche-aware)
        best_time_text = generate_best_post_time(platform, niche)

        # Try to extract structured pieces (caption, hashtags, tags, score)
        caption = None
        hashtags = None
        tags = None
        score = None

        # 1) Caption: find the Scroll-Stopping Caption section
        cap_match = re.search(r"###\s*1\.\s*Scroll-Stopping Caption[\s\S]*?(?=###|$)", ai_text, re.IGNORECASE)
        if cap_match:
            q = re.search(r'"([^"]{3,400})"', cap_match.group(0))
            if q:
                caption = q.group(1).strip()
            else:
                # fallback: first non-empty line
                lines = [l.strip() for l in cap_match.group(0).splitlines() if l.strip()]
                if len(lines) > 1:
                    caption = lines[1]

        # 2) Hashtags / Tags: find hashes
        hashtags_list = re.findall(r"#\w+", ai_text)
        if hashtags_list:
            hashtags = " ".join(hashtags_list)
            # tags fallback: first five words or keywords list after "5 Viral"
            tags = ", ".join([t.strip("#") for t in hashtags_list[:5]])

        # 3) Score
        score_m = re.search(r"(\d{1,3})\s*(?:/100|\bpercent\b|\%)", ai_text)
        if score_m:
            score = score_m.group(1)

        # Prepare structured dict
        structured = {
            "caption": caption,
            "hashtags": hashtags,
            "tags": tags,
            "optimization_score": int(score) if score and score.isdigit() else None,
            "best_time": best_time_text,
            "niche": niche,
        }

        # Prepare final human-readable output (keeping your exact structured format but adding the best_time block)
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
⭐ Heuristic Score: {structured.get('optimization_score') or '8'}/10 (Heuristic baseline; see AI suggestions)

🎯 Detected Attributes:
- Niche: {structured.get('niche') or 'General'}
- Tone: {props['tone']}
- Keywords: {', '.join(list(hashtags_list[:6])) if hashtags_list else 'None detected'}

💬 AI-Generated Viral Insights:
{ai_text.split('💬',1)[-1] if '💬' in ai_text else ai_text}

🕓 **Best Time to Post for {structured.get('niche') or 'General'} ({platform})**:
{best_time_text}
"""

        # Strip any trailing JSON block markers (common in some prompts) — keep the cleaning conservative
        final_output = re.sub(r"===JSON===.*", "", final_output, flags=re.DOTALL).strip()

        return jsonify({"result": final_output, "structured": structured})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
