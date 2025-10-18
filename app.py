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
from datetime import datetime

app = Flask(__name__, template_folder="templates")
CORS(app)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


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
        "brightness": round(brightness, 2),
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
    Analyze this TikTok or YouTube video to detect its niche, tone, and most relevant keywords.

    Video Name: {video_name}
    Video Insights: {video_insights if video_insights else 'No media insights available'}
    CSV Sample Data: {csv_data if csv_data else 'No CSV provided'}

    Return only JSON with keys: niche, tone, keywords.
    """

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=base_prompt
    )

    text = response.output[0].content[0].text
    match = re.findall(r'"?(\w+)"?:\s*"([^"]+)"', text)
    result = {k: v for k, v in match}
    return {
        "niche": result.get("niche", "General"),
        "tone": result.get("tone", "Neutral"),
        "keywords": result.get("keywords", "None detected")
    }


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
    else:
        algo_focus = "Analyze this video for general social media virality."
        tag_label = "Tags"
        insights_label = "AI-Generated Viral Insights"

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
    ⏰ 4–7 PM EST
    💡 Peak engagement around 8:43 PM EST.
    """

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )

    return response.output[0].content[0].text.strip()


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
        df = pd.read_csv(csv_file)
        csv_data = df.head(5).to_dict()

    # First try AI media analysis
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
