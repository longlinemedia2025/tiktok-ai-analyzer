from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import cv2
import numpy as np
from moviepy.editor import VideoFileClip
from openai import OpenAI
import tempfile
import re

app = Flask(__name__, template_folder="templates")
CORS(app)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    platform = request.form.get("platform", "tiktok").lower()
    niche = request.form.get("niche", "General")

    video = request.files.get("video")
    csv_file = request.files.get("csv")

    video_path = None
    csv_path = None

    if video:
        temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        video.save(temp_video.name)
        video_path = temp_video.name

    if csv_file:
        temp_csv = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        csv_file.save(temp_csv.name)
        csv_path = temp_csv.name

    # --- Extract video metadata ---
    video_info = ""
    if video_path:
        try:
            clip = VideoFileClip(video_path)
            duration = clip.duration
            frame = clip.get_frame(0)
            height, width, _ = frame.shape
            aspect_ratio_val = width / height
            video_info = (
                f"📏 Duration: {duration:.2f}s\n"
                f"🖼 Resolution: {width}x{height}\n"
                f"📱 Aspect Ratio: {aspect_ratio_val:.3f}"
            )
            clip.close()
        except Exception as e:
            video_info = f"Error analyzing video: {e}"

    # --- Prompt for AI ---
    prompt = f"""
🎬 Analyze this {platform} video for viral potential in the {niche} niche.

Include:
1. A scroll-stopping caption idea.
2. 5 viral hashtags.
3. One actionable tip for engagement.
4. A numeric viral optimization score (0–100) with explanation.
5. A short motivational takeaway.
6. A comparison with 3 viral {platform} examples in the same niche.
7. A concise takeaway strategy.
8. A 4-point actionable checklist.
9. The best time to post for this niche in EST.

Use clear emojis and structured Markdown format exactly like a social media strategist report.
"""

    ai_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a creative viral strategist and social platform optimization expert."},
            {"role": "user", "content": prompt},
        ],
    )

    ai_text = ai_response.choices[0].message.content

    # --- Remove any duplicate Best Time section ---
    ai_text = re.sub(r"🕓\s*\*\*Best Time to Post.*?(?:\n💡.*)?", "", ai_text, flags=re.DOTALL)

    # --- Default viral examples by platform ---
    viral_examples_by_platform = {
        "tiktok": """
🔥 Viral Comparison Results:
### Comparison with Viral TikToks in the Same Niche
#### Viral Example 1
- **Video Concept Summary:** A hairstylist showcases a dramatic hair color change with before-and-after shots.
- **What Made It Go Viral:** Quick cuts and an upbeat trending sound enhanced the transformation.
- **How to Replicate Success:** Use rapid transitions and a catchy trending sound that aligns with your theme.

#### Viral Example 2
- **Video Concept Summary:** A barbershop highlights client transformations in a fun montage.
- **What Made It Go Viral:** High-energy edits and viewer challenges encouraged engagement.
- **How to Replicate Success:** Ask viewers to comment their favorite transformation.

#### Viral Example 3
- **Video Concept Summary:** A barber educates viewers while performing a fade.
- **What Made It Go Viral:** Blending education with entertainment increased shareability.
- **How to Replicate Success:** Add quick tutorials or tips in your videos.
""",
        "youtube": """
🔥 Viral Comparison Results:
### Comparison with Viral YouTube Videos in the Same Niche
#### Viral Example 1
- **Video Concept Summary:** A creator explains a complex topic using humor and visual storytelling.
- **What Made It Go Viral:** High retention from a captivating hook and clear narrative pacing.
- **How to Replicate Success:** Start with a bold question or problem, then resolve it by the end.

#### Viral Example 2
- **Video Concept Summary:** A creator posts a cinematic vlog with music and emotion-driven editing.
- **What Made It Go Viral:** Emotional resonance combined with visually striking shots.
- **How to Replicate Success:** Focus on emotional storytelling and pacing.

#### Viral Example 3
- **Video Concept Summary:** A tutorial that solves a common problem in under 5 minutes.
- **What Made It Go Viral:** Short, actionable, and valuable—optimized for algorithmic promotion.
- **How to Replicate Success:** Deliver immediate value early and cut all fluff.
""",
        "instagram": """
🔥 Viral Comparison Results:
### Comparison with Viral Instagram Reels in the Same Niche
#### Viral Example 1
- **Video Concept Summary:** A short beauty reel showcasing a quick morning routine.
- **What Made It Go Viral:** Visually satisfying transitions and aesthetic color grading.
- **How to Replicate Success:** Maintain color consistency and sync edits to the beat.

#### Viral Example 2
- **Video Concept Summary:** A lifestyle influencer shares an emotional message with trending audio.
- **What Made It Go Viral:** Authentic emotion paired with a relatable caption.
- **How to Replicate Success:** Write captions that evoke emotion or vulnerability.

#### Viral Example 3
- **Video Concept Summary:** A fitness coach demonstrates a 15-second challenge.
- **What Made It Go Viral:** Fast-paced action with clear on-screen text.
- **How to Replicate Success:** Use on-screen text to highlight key takeaways.
""",
        "facebook": """
🔥 Viral Comparison Results:
### Comparison with Viral Facebook Videos in the Same Niche
#### Viral Example 1
- **Video Concept Summary:** A community member shares a heartwarming story of kindness.
- **What Made It Go Viral:** Emotional storytelling and strong community connection.
- **How to Replicate Success:** Emphasize relatable human experiences and local relevance.

#### Viral Example 2
- **Video Concept Summary:** A small business shares a behind-the-scenes video of their process.
- **What Made It Go Viral:** Transparency and authenticity attracted engagement.
- **How to Replicate Success:** Show your process—people love “how it’s made” content.

#### Viral Example 3
- **Video Concept Summary:** A funny meme video with commentary on current events.
- **What Made It Go Viral:** Humor and timely posting created high shareability.
- **How to Replicate Success:** Post fast on trending topics while adding your own twist.
"""
    }

    # --- Insert platform-appropriate viral examples if missing ---
    if "Viral Example 1" not in ai_text:
        viral_block = viral_examples_by_platform.get(platform, viral_examples_by_platform["tiktok"])
        if "### Takeaway" in ai_text:
            ai_text = ai_text.replace("### Takeaway", viral_block + "\n\n### Takeaway")
        else:
            ai_text += viral_block

    # --- Extract best time if missing ---
    best_time_match = re.search(r"(⏰ .*?EST[^\n]*)", ai_text)
    best_time_text = best_time_match.group(1) if best_time_match else "⏰ 6–10 PM EST"
    peak_match = re.search(r"(💡 .*?EST[^\n]*)", ai_text)
    peak_text = peak_match.group(1) if peak_match else "💡 Peak engagement around 8 PM EST."

    ai_text = re.sub(r"\n{3,}", "\n\n", ai_text).strip()

    # --- Final formatted output ---
    final_output = f"""
🎬 Drag and drop your {platform.capitalize()} video file here: "{video.filename if video else 'N/A'}"
🎥 Running {platform.capitalize()} Viral Optimizer...

🤖 Generating AI-powered analysis, captions, and viral tips...

🔥 Fetching viral video comparisons and strategic insights...

✅ {platform.capitalize()} Video Analysis Complete!

🎬 Video: {video.filename if video else 'N/A'}
{video_info}

{ai_text}

🎯 **Detected Niche:** {niche.capitalize()}
🕓 **Best Time to Post for {niche.capitalize()} ({platform.capitalize()})**:
{best_time_text}
{peak_text}
"""

    return jsonify({"result": final_output})

if __name__ == "__main__":
    app.run(debug=True)
