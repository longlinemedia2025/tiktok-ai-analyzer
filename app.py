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
    # safe sample frame
    sample_time = min(max(0.05, clip.duration / 2), max(0.05, clip.duration - 0.05))
    frame = clip.get_frame(sample_time)
    # normalize frame to uint8 RGB/BGR
    try:
        frame_arr = (frame * 255).astype(np.uint8)
    except Exception:
        frame_arr = frame.astype(np.uint8)
    # guess shape order
    h, w = frame_arr.shape[0], frame_arr.shape[1]
    aspect_ratio = round(w / h, 3) if h != 0 else 0
    # convert to grayscale robustly
    try:
        gray = cv2.cvtColor(frame_arr, cv2.COLOR_RGB2GRAY)
    except Exception:
        gray = cv2.cvtColor(frame_arr, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    # tone heuristic
    tone = "bright and energetic" if brightness > 110 else "neutral or mixed" if 60 <= brightness <= 110 else "dark and moody"

    clip.reader.close()
    if hasattr(clip, "audio") and clip.audio:
        try:
            clip.audio.reader.close_proc()
        except Exception:
            pass

    return {
        "duration": duration,
        "resolution": f"{w}x{h}",
        "aspect_ratio": aspect_ratio,
        "brightness": round(brightness, 2),
        "tone": tone,
    }


def best_time_for(platform, niche):
    """
    Heuristic best-post-time generator based on platform, niche, and current day.
    Returns a tuple: (day_name, time_window_str, peak_time_str)
    """
    now = datetime.datetime.now()
    day = now.strftime("%a")  # Mon, Tue, ...
    niche_l = (niche or "").lower()

    # Default windows by platform
    platform_windows = {
        "tiktok": ("6–10 PM EST", "8:{:02d} PM EST"),
        "instagram": ("5–9 PM EST", "7:{:02d} PM EST"),
        "youtube": ("7–10 PM EST", "8:{:02d} PM EST"),
        "facebook": ("11 AM–2 PM EST", "12:{:02d} PM EST"),
    }
    window, peak_fmt = platform_windows.get(platform.lower(), ("6–10 PM EST", "8:{:02d} PM EST"))

    # Niche-driven adjustments
    if any(k in niche_l for k in ["gaming", "music", "entertainment", "memes"]):
        if platform.lower() in ["tiktok", "youtube"]:
            window, peak_fmt = ("7–11 PM EST", "9:{:02d} PM EST")
        elif platform.lower() == "instagram":
            window, peak_fmt = ("6–10 PM EST", "8:{:02d} PM EST")
        else:
            window, peak_fmt = ("6–9 PM EST", "7:{:02d} PM EST")
    elif any(k in niche_l for k in ["business", "finance", "education", "marketing"]):
        window, peak_fmt = ("9 AM–12 PM EST", "10:{:02d} AM EST")
    elif any(k in niche_l for k in ["beauty", "fashion", "lifestyle", "health"]):
        if platform.lower() == "facebook":
            window, peak_fmt = ("11 AM–3 PM EST", "12:{:02d} PM EST")
        else:
            window, peak_fmt = ("4–7 PM EST", "5:{:02d} PM EST")
    elif any(k in niche_l for k in ["food", "cooking"]):
        window, peak_fmt = ("11 AM–2 PM EST", "12:{:02d} PM EST")

    # Prefer weekends for lifestyle/entertainment, weekdays for business/education
    weekday = now.weekday()  # 0=Mon
    day_name = now.strftime("%a")
    # small stochastic shift so repeated runs don't always show same minute
    minute_rand = random.randint(0, 59)
    peak_time = peak_fmt.format(minute_rand)

    return day_name, window, peak_time


def generate_ai_prompt(video_name, video_props, platform):
    """
    Build an instruction-prompt that asks for:
    - the exact human-readable emoji format (as you required)
    - a compact JSON object at the end labeled ===STRUCTURED_JSON===
    """
    platform_label = platform.capitalize()
    # instruct the model to produce 3 real relevant viral examples, plus per-platform caption/title & hashtag suggestions
    prompt = f"""
You are an expert social media strategist and content analyst that emulates how each platform
(TikTok, Instagram Reels, YouTube Shorts, Facebook Reels) analyzes uploaded videos.

Produce a complete, actionable viral optimization analysis for the following uploaded video. MUST follow this exact human-readable structure (including emojis and headings). After the human-readable text, append a compact JSON block labeled ===STRUCTURED_JSON=== containing specific fields (caption/title suggestions for each platform, hashtags list, tags list, optimization_score, detected_niche, best_time_day, best_time_window, best_time_peak, and 'viral_examples' array with 3 items where each item has summary, why_it_went_viral, replicate_instructions).

Video file name: "{video_name}"
Duration: {video_props['duration']}s
Resolution: {video_props['resolution']}
Aspect Ratio: {video_props['aspect_ratio']}
Brightness: {video_props['brightness']}
Tone: {video_props['tone']}

Use the exact formatted output as below (replace bracketed content with analysis):
---
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

💬 AI-Generated Viral Insights:
### 1. Scroll-Stopping Caption
(Provide a single platform-agnostic caption + then platform-specific short variations labeled TikTok/Instagram/YouTube/Facebook. Keep the first line quoted exactly like a caption.)

### 2. 5 Viral Hashtags
(List five hashtags optimized for the detected niche; each hashtag prefixed with # separated by spaces.)

### 3. Actionable Improvement Tip for Engagement
(Short, concrete suggestion tuned to the chosen platform.)

### 4. Viral Optimization Score (1–100)
(Provide numeric score and 1-2 sentence explanation of strengths/weaknesses.)

### 5. Short Motivation on How to Increase Virality
(A brief encouraging tip.)

🔥 Viral Comparison Results:
### Comparison with Viral {platform_label} Videos in the Same Niche
(Provide exactly 3 viral examples. For each example include:)
#### Viral Example 1
- **Video Concept Summary:** short summary (1-2 lines)
- **What Made It Go Viral:** 1-2 lines
- **How to Replicate Success:** 1-2 lines

#### Viral Example 2
- **Video Concept Summary:** ...
- **What Made It Go Viral:** ...
- **How to Replicate Success:** ...

#### Viral Example 3
- **Video Concept Summary:** ...
- **What Made It Go Viral:** ...
- **How to Replicate Success:** ...

### Takeaway Strategy
(Concise actionable summary)

📋 Actionable Checklist:
- 4 short checklist items

🎯 **Detected Niche:** (detect the niche from visual/audio clues)
🕓 **Best Time to Post for {platform_label} (by niche and current day)**:
⏰ (Day + time window in EST)
💡 Peak engagement around (specific time in EST)

--- (end human-readable)

IMPORTANT: After the human-readable section, output a compact JSON block starting on a new line EXACTLY like:
===STRUCTURED_JSON===
{{"caption_suggestion":"...","platform_variations":{{"tiktok":"...","instagram":"...","youtube":"...","facebook":"..."}}, "hashtags":["h1","h2","h3","h4","h5"], "tags":["tag1","tag2"], "optimization_score":85, "detected_niche":"Beauty","best_time_day":"Thu","best_time_window":"4–7 PM EST","best_time_peak":"8:43 PM EST","viral_examples":[{{"summary":"...","why_it_went_viral":"...","how_to_replicate":"..."}},...]}}

Do NOT include any other JSON anywhere else. The JSON must be valid and parsable.

Be concise in human-readable fields but keep the structure exactly as requested.
"""
    return prompt


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        platform = request.form.get("platform", "tiktok").strip()
        video = request.files.get("video")

        if not video:
            return jsonify({"error": "No video uploaded"}), 400

        # save video
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            video.save(tmp.name)
            video_path = tmp.name

        # analyze visual properties
        props = analyze_video_properties(video_path)

        # build and send prompt to OpenAI
        prompt = generate_ai_prompt(video.filename, props, platform)

        # call model
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.85,
                max_tokens=1600,
            )
            ai_text = response.choices[0].message.content
            if isinstance(ai_text, dict):
                ai_text = ai_text.get("content", "")  # safety if SDK returns different shape
            ai_text = ai_text.strip()
        except Exception as e:
            # graceful fallback text if model fails
            ai_text = f"(AI generation failed: {str(e)})\n\nPlease retry."

        # try to extract structured JSON block labeled ===STRUCTURED_JSON===
        structured = {}
        json_block = None
        m = re.search(r"===STRUCTURED_JSON===\s*(\{[\s\S]*\})\s*$", ai_text, flags=re.MULTILINE)
        if m:
            json_block = m.group(1)
            try:
                structured = json.loads(json_block)
            except Exception:
                # attempt to sanitize single quotes or trailing commas then parse
                try:
                    cleaned = re.sub(r",\s*}", "}", json_block)
                    cleaned = re.sub(r",\s*\]", "]", cleaned)
                    structured = json.loads(cleaned)
                except Exception:
                    structured = {}
            # remove the JSON block from ai_text for the human-readable output to avoid duplication
            ai_text = ai_text[:m.start()].strip()
        else:
            # If JSON block not present, try extracting essential items heuristically
            # extract first quoted caption
            cap_m = re.search(r'###\s*1\.\s*Scroll-Stopping Caption[\s\S]*?"([^"]{3,400})"', ai_text)
            caption_guess = cap_m.group(1).strip() if cap_m else None
            hashtags = re.findall(r"#\w+", ai_text)[:5]
            score_m = re.search(r"(\d{1,3})\s*(?:/100|\bpercent\b|\%)", ai_text)
            optimization_score = int(score_m.group(1)) if score_m else None
            # detected niche heuristic
            niche_m = re.search(r"🎯\s*\*\*Detected Niche:\*\*\s*(.+)", ai_text) or re.search(r"\*\*Detected Niche\*\*[:\-–]?\s*(.+)", ai_text)
            detected_niche = (niche_m.group(1).strip() if niche_m else None) or "General"

            # compute best time heuristically
            day_name, window, peak = best_time_for(platform, detected_niche)
            structured = {
                "caption_suggestion": caption_guess,
                "platform_variations": {
                    "tiktok": caption_guess or "",
                    "instagram": caption_guess or "",
                    "youtube": caption_guess or "",
                    "facebook": caption_guess or ""
                },
                "hashtags": [h.lstrip("#") for h in hashtags],
                "tags": [],
                "optimization_score": optimization_score or None,
                "detected_niche": detected_niche,
                "best_time_day": day_name,
                "best_time_window": window,
                "best_time_peak": peak,
                "viral_examples": []
            }

        # If structured missing best_time, fill with heuristic
        if not structured.get("best_time_day") or not structured.get("best_time_window"):
            day_name, window, peak = best_time_for(platform, structured.get("detected_niche", "General"))
            structured.setdefault("best_time_day", day_name)
            structured.setdefault("best_time_window", window)
            structured.setdefault("best_time_peak", peak)

        # Build final human-readable output (ensuring format requested, and no duplicate best-time block)
        # We'll include the AI human text and then append the best time block from our structured fields
        # If ai_text already contains the human-readable block (likely), keep it intact but append best-time if missing
        final_text = ai_text

        # Ensure the top header (drag and drop) appears — if AI didn't include it, construct it
        if not final_text.startswith("🎬"):
            platform_label = platform.capitalize()
            header = (
                f"🎬 Drag and drop your {platform_label} video file here: \"{video.filename}\"\n"
                f"🎥 Running {platform_label} Viral Optimizer...\n\n"
                "🤖 Generating AI-powered analysis, captions, and viral tips...\n\n"
                "🔥 Fetching viral video comparisons and strategic insights...\n\n"
                f"✅ {platform_label} Video Analysis Complete!\n\n"
                f"🎬 Video: {video.filename}\n"
                f"📏 Duration: {props['duration']}s\n"
                f"🖼 Resolution: {props['resolution']}\n"
                f"📱 Aspect Ratio: {props['aspect_ratio']}\n"
                f"💡 Brightness: {props['brightness']}\n"
                f"🎨 Tone: {props['tone']}\n"
                f"⭐ Heuristic Score: {structured.get('optimization_score') or 8}/10 (Heuristic baseline; see AI suggestions)\n\n"
            )
            final_text = header + final_text

        # Append the single "Best Time to Post" block (from structured) — don't duplicate
        bt_day = structured.get("best_time_day")
        bt_window = structured.get("best_time_window")
        bt_peak = structured.get("best_time_peak")
        if bt_day and bt_window:
            best_time_block = (
                f"\n🎯 **Detected Niche:** {structured.get('detected_niche', 'General')}\n"
                f"🕓 **Best Time to Post for {structured.get('detected_niche', 'General')} ({platform.capitalize()})**:\n"
                f"⏰ {bt_day} {bt_window}\n"
                f"💡 Peak engagement around {bt_peak}\n"
            )
            # remove any old duplicated lines that look like a best time block
            final_text = re.sub(r"🕓\s*\*\*Best Time to Post[\s\S]*?(?=\n\n|$)", "", final_text)
            final_text = final_text.strip() + "\n\n" + best_time_block

        # Return human-readable final_text plus structured JSON separately
        return jsonify({"result": final_text.strip(), "structured": structured})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # run on 0.0.0.0:5000 in dev mode by default
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
