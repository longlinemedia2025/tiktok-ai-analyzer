from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import cv2
import numpy as np
from moviepy.editor import VideoFileClip
from openai import OpenAI
import tempfile
import pandas as pd
import re
from datetime import datetime, timedelta
import requests
import math

app = Flask(__name__, template_folder="templates")
CORS(app)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Optional: external "best time" service (set in Render or env)
BEST_TIME_API_URL = os.getenv("BEST_TIME_API_URL")  # e.g. https://example.com/best_time
BEST_TIME_API_KEY = os.getenv("BEST_TIME_API_KEY")


def analyze_video_features(video_path):
    """Extracts key video metrics like brightness, duration, and resolution."""
    clip = VideoFileClip(video_path)
    duration = clip.duration
    frame = clip.get_frame(0)
    height, width, _ = frame.shape
    brightness = np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 2])
    aspect_ratio = round(width / height, 3)
    return {
        "duration": round(duration, 2),
        "resolution": f"{width}x{height}",
        "aspect_ratio": aspect_ratio,
        "brightness": round(float(brightness), 2),
    }


def extract_audio_features(video_path):
    """Extracts average volume and audio presence using moviepy."""
    try:
        clip = VideoFileClip(video_path)
        audio = clip.audio
        if audio is None:
            return {"audio_volume": 0, "has_audio": False}
        samples = audio.to_soundarray(fps=22000)
        volume = np.mean(np.abs(samples))
        return {"audio_volume": round(float(volume), 4), "has_audio": True}
    except Exception:
        return {"audio_volume": 0, "has_audio": False}


def detect_niche_tone_keywords(video_name, csv_data=None, video_insights=None):
    """Uses AI to detect niche, tone, and keywords with fallback logic."""
    base_prompt = f"""
    Analyze this TikTok/YouTube/Instagram/Facebook video to detect its niche, tone, and most relevant keywords.

    Video Name: {video_name}
    Video Insights: {video_insights if video_insights else 'No media insights available'}
    CSV Sample Data: {csv_data if csv_data else 'No CSV provided'}

    Return only JSON with keys: niche, tone, keywords.
    """
    response = client.responses.create(
        model="gpt-4.1-mini",
        input=base_prompt
    )
    # model output parsing (robust-ish)
    text = ""
    try:
        text = response.output[0].content[0].text
    except Exception:
        text = str(response)

    match = re.findall(r'"?(\w+)"?\s*:\s*"([^"]+)"', text)
    result = {k: v for k, v in match}
    return {
        "niche": result.get("niche", "General"),
        "tone": result.get("tone", "Neutral"),
        "keywords": result.get("keywords", "None detected")
    }


def call_external_best_time_api(platform, niche):
    """
    Call external BEST_TIME_API_URL if provided.
    Expected to return JSON with keys: best_day, start_time, end_time, peak_time
    Example response:
      { "best_day": "Fri", "start_time": "19:00", "end_time": "22:00", "peak_time": "20:22" }
    """
    if not BEST_TIME_API_URL:
        return None

    try:
        headers = {}
        if BEST_TIME_API_KEY:
            headers["Authorization"] = f"Bearer {BEST_TIME_API_KEY}"
        params = {"platform": platform, "niche": niche}
        resp = requests.get(BEST_TIME_API_URL, params=params, headers=headers, timeout=8)
        if resp.status_code == 200:
            data = resp.json()
            # basic validation
            if isinstance(data, dict) and any(k in data for k in ("best_day", "peak_time", "start_time", "end_time")):
                return {
                    "best_day": data.get("best_day"),
                    "start_time": data.get("start_time"),
                    "end_time": data.get("end_time"),
                    "peak_time": data.get("peak_time")
                }
    except Exception as e:
        # external call failed; fallback will be used
        print("BEST_TIME_API call failed:", e)
    return None


def heuristic_best_time(platform, niche):
    """
    Simple but data-driven heuristic for best post time by niche & platform.
    Uses current date/time to propose a best day, a 3-hour window and a peak time.
    """
    # base mapping: niche -> preferred day type and hour center (local heuristic)
    niche = (niche or "General").lower()
    now = datetime.utcnow()  # UTC baseline; frontend/js can translate if needed
    weekday = now.weekday()  # 0=Mon .. 6=Sun

    # typical hour centers for niche per platform (in 24h UTC offsets approximate)
    # These are heuristics — you can tweak them if you want different behavior per region/timezone later.
    mapping = {
        "gaming": {"day_offset": 4, "hour": 20},            # Fri evening
        "beauty": {"day_offset": 3, "hour": 19},            # Thu evening
        "education": {"day_offset": 1, "hour": 18},         # Tue evening
        "lifestyle": {"day_offset": 6, "hour": 16},         # Sun afternoon
        "fitness": {"day_offset": 5, "hour": 7},            # Sat morning
        "technology": {"day_offset": 2, "hour": 12},        # Wed midday
        "food": {"day_offset": 5, "hour": 12},              # Sat midday
        "music": {"day_offset": 4, "hour": 21},             # Fri night
        "gaming - horror": {"day_offset": 4, "hour": 20},   # keep gaming mapping
        "general": {"day_offset": 4, "hour": 19}
    }

    # find closest mapping key
    matched = None
    for k in mapping:
        if k in niche:
            matched = mapping[k]
            break
    if matched is None:
        matched = mapping.get("general")

    # platform-specific adjustments (Facebook tends to show midday and early evening)
    platform = (platform or "general").lower()
    platform_adj = {"facebook": -1, "tiktok": 0, "youtube": 0, "instagram": 0}
    pad = platform_adj.get(platform, 0)
    center_hour = (matched["hour"] + pad) % 24

    # compute best day relative to today (use day_offset as weekday index 0..6)
    # we allow the mapping to return a preferred weekday name like 'Fri' if possible
    preferred_weekday_index = matched["day_offset"]  # 0..6
    # pick the next preferred weekday from now
    days_ahead = (preferred_weekday_index - weekday) % 7
    if days_ahead == 0:
        days_ahead = 0  # today is ok
    best_day_dt = now + timedelta(days=days_ahead)
    best_day_name = best_day_dt.strftime("%a")  # e.g. "Fri"

    # create a 3-hour window centered on center_hour
    start_hour = (center_hour - 1) % 24
    end_hour = (center_hour + 2) % 24
    # Get approximate peak minute (use niche hash to vary minute)
    minute = int((abs(hash(niche)) % 60))
    peak_hour = center_hour
    peak_time = f"{peak_hour:02d}:{minute:02d}"

    start_time = f"{start_hour:02d}:00"
    end_time = f"{end_hour:02d}:00"

    # Present as friendly local-ish times: we'll return the values as strings; the UI can adjust timezones if you want later.
    return {
        "best_day": best_day_name,
        "start_time": start_time,
        "end_time": end_time,
        "peak_time": peak_time
    }


def get_best_posting_window(platform, niche):
    """
    Returns a dict with keys best_day, start_time, end_time, peak_time.
    First tries external API (if configured) then falls back to heuristic.
    """
    # try external API
    ext = call_external_best_time_api(platform, niche)
    if ext:
        return ext
    # fallback heuristic
    return heuristic_best_time(platform, niche)


def sanitize_model_output(text):
    """
    Clean up the model response to remove any trailing '===JSON===' blocks
    or stray appended captions that were duplicated.
    Returns cleaned text.
    """
    if not text:
        return text
    # remove anything after "===JSON===" token if present
    split_token = "===JSON==="
    if split_token in text:
        text = text.split(split_token)[0].strip()

    # remove stray trailing lines that start with "Caption:" after the main body
    # find last occurrence of a major section marker like "Takeaway Strategy" or "📋 Actionable Checklist"
    cut_markers = ["📋 Actionable Checklist", "### Takeaway Strategy", "🔥 Viral Comparison Results", "✅"]
    cut_index = None
    for m in cut_markers:
        idx = text.rfind(m)
        if idx != -1:
            cut_index = idx
            break
    # If we found a marker, keep everything from start to end (don't chop). Otherwise attempt to remove lines starting with "Caption:"
    # (This logic is conservative to avoid cutting important content.)
    if cut_index is None:
        # Remove trailing lines that look like duplicate short metadata lines such as:
        # Caption: ...
        # Hashtags: ...
        # Tags: ...
        # Score: ...
        lines = text.splitlines()
        cleaned_lines = []
        for line in lines:
            if re.match(r'^\s*(Caption|Hashtags|Tags|Score)\s*:', line):
                # skip these duplicates at the end if they occur after a long output (heuristic)
                # we only skip these if text is long (to avoid removing intended small outputs)
                if len(lines) > 12:
                    continue
            cleaned_lines.append(line)
        text = "\n".join(cleaned_lines).strip()

    return text


def generate_ai_analysis(platform, video_name, metrics, detected, csv_data=None):
    """Generates the AI-enhanced viral analysis with your exact format."""
    duration = metrics["duration"]
    resolution = metrics["resolution"]
    aspect_ratio = metrics["aspect_ratio"]
    brightness = metrics["brightness"]

    niche = detected["niche"]
    tone = detected["tone"]
    keywords = detected["keywords"]

    # Platform-specific instructions
    if platform == "TikTok":
        algo_focus = (
            "Analyze this TikTok video using TikTok’s algorithmic preferences: "
            "short watch loops, high early engagement, strong hook within first 2 seconds, trending sounds, "
            "and authentic visual appeal. Focus on captions, hashtags, and timing that improve For You Page visibility."
        )
        tag_label = "Hashtags"
        insights_label = "AI-Generated Viral Insights"
    elif platform == "YouTube":
        algo_focus = (
            "Analyze this YouTube video using YouTube’s algorithm: "
            "click-through rate, watch time, audience retention, SEO tags, and consistency in content topic. "
            "Focus on title optimization, tags, and viewer engagement strategies."
        )
        tag_label = "Tags"
        insights_label = "AI-Generated Viral Insights"
    elif platform == "Instagram":
        algo_focus = (
            "Analyze this Instagram Reel or post according to Instagram’s algorithm: "
            "prioritize early saves, shares, comments, Reels completion rate, trending audio use, "
            "and strong visual storytelling. Focus on boosting discoverability via Explore and Reels tabs, "
            "and tailoring captions, sounds, and timing for Instagram’s content behavior."
        )
        tag_label = "Hashtags"
        insights_label = "Reels-Focused Viral Insights"
    elif platform == "Facebook":
        algo_focus = (
            "Analyze this Facebook video/post with Facebook's algorithm in mind: "
            "engagement signals (likes, comments, shares) and retention matter; Live and native video perform well; "
            "formatting that drives shares and comments (caption prompts, meaningful CTAs) is crucial. "
            "Prioritize audience affinity and community retention patterns when recommending post timing and CTAs."
        )
        tag_label = "Tags"
        insights_label = "Facebook-Focused Viral Insights"
    else:
        algo_focus = "Analyze this video for general social media virality."
        tag_label = "Tags"
        insights_label = "AI-Generated Viral Insights"

    # Determine best post timing (real-time or heuristic)
    timing = get_best_posting_window(platform, niche)
    # Format best time text for insertion into the prompt (keep same friendly style)
    best_day = timing.get("best_day", "Thu")
    start_time = timing.get("start_time", "16:00")
    end_time = timing.get("end_time", "19:00")
    peak_time = timing.get("peak_time", "20:43")

    best_time_text = f"{best_day} {start_time}–{end_time} — peak around {peak_time}."

    prompt = f"""
    {algo_focus}

    Create a viral video analysis for {platform} with this structure EXACTLY:

    🎬 Drag and drop your {platform} video file here: "{video_name}"
    🎥 Running {platform} Viral Optimizer...

    🤖 Generating AI-powered analysis, captions, and viral tips...

    🔥 Fetching viral video comparisons and strategic insights...

    ✅ {platform} Video Analysis Complete!

    🎬 Video: {video_name}
    📏 Duration: {duration}s
    🖼 Resolution: {resolution}
    📱 Aspect Ratio: {aspect_ratio}
    💡 Brightness: {brightness}
    🎨 Tone: {tone}
    ⭐ Heuristic Score: 8/10 (High brightness and clear resolution contribute to visual appeal.)

    🎯 Detected Attributes:
    - Niche: {niche}
    - Tone: {tone}
    - Keywords: {keywords}

    💬 {insights_label}:
    ### 1. Scroll-Stopping Caption
    (Give one creative, high-performing caption idea. 
    For Instagram: make it emotional, story-driven, and include a subtle CTA to save/share.)

    ### 2. 5 Viral {tag_label}
    (Provide 5 platform-relevant {tag_label.lower()} that match the {platform} algorithm 
    — for Instagram, include niche + trending Reels hashtags like #reelitfeelit, #instagrowth.)

    ### 3. Actionable Improvement Tip for Engagement
    (Provide a short suggestion for improving engagement on {platform}. 
    For Instagram, prioritize “save-worthy” content, visually cohesive style, and authentic storytelling.)

    ### 4. Viral Optimization Score (1–100)
    (Give a score with an explanation tailored to the {platform} algorithm.)

    ### 5. Motivation to Increase Virality
    (Add an encouraging, platform-specific creator tip. 
    For Instagram: mention Reels consistency, audio trends, and storytelling over perfection.)

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
    (Summarize actionable insights for {platform} creators.)

    📋 Actionable Checklist:
    - Hook viewers in the first 2 seconds.
    - Use trending audio and relevant captions.
    - Encourage saves and shares with call-to-actions.
    - Maintain visual consistency across Reels.

    🎯 **Detected Niche:** {niche}
    🕓 **Best Time to Post for {niche} ({platform})**:
    ⏰ {best_time_text}
    """

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )

    # sanitize & return cleaned text
    try:
        raw_text = response.output[0].content[0].text
    except Exception:
        raw_text = str(response)
    # remove any trailing JSON blocks or duplicated caption snippets
    clean_text = sanitize_model_output(raw_text)
    return clean_text


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    platform = request.form.get("platform", "TikTok")
    video = request.files.get("video")
    csv_file = request.files.get("csv")

    if not video:
        return jsonify({"error": "No video uploaded."}), 400

    # Save video temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        video.save(tmp.name)
        metrics = analyze_video_features(tmp.name)
        audio_features = extract_audio_features(tmp.name)

    csv_data = None
    if csv_file:
        try:
            df = pd.read_csv(csv_file)
            csv_data = df.head(5).to_dict()
        except Exception as e:
            csv_data = None

    # First try AI media analysis summary for context
    video_insights = None
    try:
        clip = VideoFileClip(tmp.name)
        video_insights = f"Duration: {clip.duration}s, Avg Brightness: {metrics['brightness']}, Audio Volume: {audio_features['audio_volume']}, Has Audio: {audio_features['has_audio']}"
    except Exception as e:
        print("Media analysis failed, falling back:", e)

    detected = detect_niche_tone_keywords(video.filename, csv_data, video_insights)
    result_text = generate_ai_analysis(platform, video.filename, metrics, detected, csv_data)

    return jsonify({"result": result_text})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
