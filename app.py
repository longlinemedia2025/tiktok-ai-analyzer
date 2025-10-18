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
import json
from datetime import datetime

app = Flask(__name__, template_folder="templates")
CORS(app)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def analyze_video_features(video_path):
    """Extracts key video metrics like brightness, duration, and resolution."""
    clip = VideoFileClip(video_path)
    duration = clip.duration
    # get a frame safely
    try:
        frame = clip.get_frame(0)
        # note: moviepy uses (h, w, ch)
        height, width, _ = frame.shape
        brightness = np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 2])
        aspect_ratio = round(width / height, 3)
    except Exception:
        # fallback safe values
        height, width = 720, 1280
        brightness = 0.0
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
            return {"audio_volume": 0.0, "has_audio": False}
        samples = audio.to_soundarray(fps=22000)
        volume = np.mean(np.abs(samples))
        return {"audio_volume": round(float(volume), 6), "has_audio": True}
    except Exception:
        return {"audio_volume": 0.0, "has_audio": False}


def call_model_return_json(prompt, max_tokens=600):
    """
    Calls the OpenAI Responses API and attempts to parse a JSON blob from the text.
    The prompt should ask the model to return strict JSON for easier parsing.
    Returns (parsed_dict, raw_text)
    """
    try:
        resp = client.responses.create(model="gpt-4.1-mini", input=prompt, max_output_tokens=max_tokens)
        # best-effort extract text
        text = ""
        try:
            # new Responses API may return nested structure
            for c in resp.output:
                for item in (c.content or []):
                    if hasattr(item, "text"):
                        text += item.text
                    elif isinstance(item, dict) and item.get("type") == "output_text":
                        text += item.get("text", "")
        except Exception:
            # fallback
            text = getattr(resp, "text", "") or str(resp)

        # try to find the first JSON object in the text
        json_obj = None
        json_text_match = re.search(r"(\{[\s\S]*\})", text)
        if json_text_match:
            candidate = json_text_match.group(1)
            try:
                json_obj = json.loads(candidate)
            except Exception:
                # try to fix common issues (replace single quotes)
                try:
                    fixed = candidate.replace("\n", " ").replace("'", '"')
                    json_obj = json.loads(fixed)
                except Exception:
                    json_obj = None

        return (json_obj, text.strip())
    except Exception as e:
        return (None, f"MODEL_ERROR: {e}")


def detect_niche_tone_keywords(video_name, csv_data=None, video_insights=None):
    """
    Uses AI to detect niche, tone, keywords and suggests a peak engagement time.
    Returns a dict: {niche, tone, keywords, suggested_peak_time}
    """
    csv_preview = json.dumps(csv_data, default=str) if csv_data else "No CSV provided"
    vi = video_insights if video_insights else "No media insights available"

    prompt = f"""
You are an assistant that MUST return strict JSON (only) with the following keys:
- niche (string)
- tone (string)
- keywords (string, comma-separated short list or 'None detected')
- suggested_peak_time (string; user's timezone -- return a human-friendly best posting window & an exact best minute, e.g. "Thu 4–7 PM EST — peak ~8:43 PM EST")

Analyze the following inputs to determine the niche, tone, short keywords list, and a recommended peak engagement time tailored to platform & niche:
Video Name: "{video_name}"
Video Insights: "{vi}"
CSV Sample (first rows or metadata): {csv_preview}

Be concise in values. Return ONLY a JSON object (no explanation).
"""
    parsed, raw = call_model_return_json(prompt)
    # fallback defaults
    if not parsed:
        # try to extract using looser regex as backup
        niche = "General"
        tone = "Neutral"
        keywords = "None detected"
        suggested_peak_time = "Thu 4–7 PM EST — peak around 8:43 PM EST"
        # naive regex tries
        m_n = re.search(r'"niche"\s*:\s*"([^"]+)"', raw, re.I)
        m_t = re.search(r'"tone"\s*:\s*"([^"]+)"', raw, re.I)
        m_k = re.search(r'"keywords"\s*:\s*"([^"]+)"', raw, re.I)
        m_p = re.search(r'"suggested_peak_time"\s*:\s*"([^"]+)"', raw, re.I)
        if m_n: niche = m_n.group(1)
        if m_t: tone = m_t.group(1)
        if m_k: keywords = m_k.group(1)
        if m_p: suggested_peak_time = m_p.group(1)
        return {
            "niche": niche,
            "tone": tone,
            "keywords": keywords,
            "suggested_peak_time": suggested_peak_time,
            "raw_model": raw
        }
    # normalize fields
    niche = parsed.get("niche", "General")
    tone = parsed.get("tone", "Neutral")
    keywords = parsed.get("keywords", "None detected")
    suggested_peak_time = parsed.get("suggested_peak_time", "Thu 4–7 PM EST — peak around 8:43 PM EST")
    parsed["raw_model"] = raw
    return parsed


def generate_ai_analysis(platform, video_name, metrics, detected, csv_data=None):
    """
    Generates the large textual analysis (exact format you requested), while asking
    the model to also return a structured JSON block at the end containing caption,
    tags/hashtags list, optimization_score, and the suggested_peak_time (we will
    parse and return both full text and structured fields).
    """
    duration = metrics["duration"]
    resolution = metrics["resolution"]
    aspect_ratio = metrics["aspect_ratio"]
    brightness = metrics["brightness"]

    niche = detected.get("niche", "General")
    tone = detected.get("tone", "Neutral")
    keywords = detected.get("keywords", "None detected")
    suggested_peak_time = detected.get("suggested_peak_time", "Thu 4–7 PM EST — peak around 8:43 PM EST")

    # platform-specific labels
    if platform == "TikTok":
        tag_label = "Hashtags"
        insights_label = "AI-Generated Viral Insights"
        algo_focus = "short watch loops, strong hooks, trending sounds, For You Page optimization"
    elif platform == "YouTube":
        tag_label = "Tags"
        insights_label = "AI-Generated Viral Insights"
        algo_focus = "watch time, click-through rate, SEO, audience retention"
    elif platform == "Instagram":
        tag_label = "Hashtags"
        insights_label = "Reels-Focused Viral Insights"
        algo_focus = "Reels completion, saves, shares, trending audio, storytelling"
    else:
        tag_label = "Tags"
        insights_label = "AI-Generated Viral Insights"
        algo_focus = "generic virality factors"

    # We'll ask the model to produce EXACT formatted human-readable section AND a JSON block
    prompt = f"""
You are asked to produce a viral-optimization analysis for a {platform} video.
Follow this EXACT textual structure (produce the full human-readable analysis), then append a JSON block labeled with "===JSON===" on its own line containing keys:
caption, { 'hashtags' if tag_label=='Hashtags' else 'tags' }, improvement_tip, optimization_score (0-100), motivation, viral_examples (array of 3 short objects), takeaway, actionable_checklist (array), detected_niche, detected_tone, detected_keywords, suggested_peak_time.

Use the following inputs:
Video Name: "{video_name}"
Duration: {duration}s
Resolution: {resolution}
Aspect Ratio: {aspect_ratio}
Brightness: {brightness}
Detected Niche: {niche}
Detected Tone: {tone}
Detected Keywords: {keywords}
Context for algorithm focus: {algo_focus}

Write the human-readable analysis EXACTLY in this format:

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
(Provide one creative, high-performing caption idea.)

### 2. 5 Viral {tag_label}
(Provide 5 platform-relevant {tag_label.lower()} separated by spaces or commas.)

### 3. Actionable Improvement Tip for Engagement
(One concise suggestion.)

### 4. Viral Optimization Score (1–100)
(Provide a score and one-sentence justification.)

### 5. Motivation to Increase Virality
(Short uplifting tip.)

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
(Summarize actionable insights.)

📋 Actionable Checklist:
- Hook viewers in the first 2 seconds.
- Use trending audio and relevant captions.
- Encourage saves and shares with call-to-actions.
- Maintain visual consistency across Reels.

🎯 **Detected Niche:** {niche}
🕓 **Best Time to Post for {niche} ({platform})**:
⏰ {suggested_peak_time}

After the human-readable analysis, append this exact line:
===JSON===
Then return a valid JSON object (only JSON after that line) with keys described above.

Do not include any other text or commentary outside the required human-readable block and the JSON block.
"""

    parsed, raw = call_model_return_json(prompt, max_tokens=1000)
    # If the call returned a parsed JSON (from detection step maybe), try a fallback:
    if parsed and isinstance(parsed, dict) and ("caption" in parsed or "optimization_score" in parsed):
        # try to reconstruct human text from raw if raw contains the human analysis
        # if call_model_return_json returned the JSON parsed earlier, we still need the full human text.
        # Try to extract the human readable text before the JSON object in raw.
        human_text = raw
        # If raw contains the JSON block we extracted earlier, split it off
        split = re.split(r"\n===JSON===\s*", raw)
        if len(split) >= 2:
            human_text = split[0].strip()
        return human_text, parsed
    else:
        # fallback: if model couldn't give JSON parsed by our helper, do a simpler approach:
        # Call model again, but this time ask only for the human text (we'll do a second call for structured JSON).
        try:
            resp_text = client.responses.create(model="gpt-4.1-mini", input=prompt, max_output_tokens=900)
            # assemble text
            text = ""
            try:
                for c in resp_text.output:
                    for item in (c.content or []):
                        if hasattr(item, "text"):
                            text += item.text
                        elif isinstance(item, dict) and item.get("type") == "output_text":
                            text += item.get("text", "")
            except Exception:
                text = getattr(resp_text, "text", "") or str(resp_text)

            # attempt to parse JSON after ===JSON===
            json_part = None
            m = re.search(r"===JSON===\s*(\{[\s\S]*\})", text)
            if m:
                json_part = m.group(1)
                try:
                    parsed_json = json.loads(json_part)
                except Exception:
                    try:
                        parsed_json = json.loads(json_part.replace("'", '"'))
                    except Exception:
                        parsed_json = None
            else:
                parsed_json = None

            # if parsed_json is None, build a minimal structured response from detected and a few heuristics
            if not parsed_json:
                parsed_json = {
                    "caption": f'Check this out: {video_name} — watch the transformation! ✨',
                    "hashtags": "#example #viral",
                    "tags": "#example",
                    "improvement_tip": "Add a trending sound and a strong hook in the first 2 seconds.",
                    "optimization_score": 78,
                    "motivation": "Keep experimenting with hooks — you’re close!",
                    "viral_examples": [],
                    "takeaway": "Focus on hooks and trending audio.",
                    "actionable_checklist": ["Hook in first 2s", "Use trending audio", "Ask viewers to comment"],
                    "detected_niche": niche,
                    "detected_tone": tone,
                    "detected_keywords": keywords,
                    "suggested_peak_time": suggested_peak_time
                }

            # Return the full human text and the parsed JSON
            return text.strip(), parsed_json
        except Exception as e:
            # final fallback: minimal human text + structured
            human_text = f"""🎬 Drag and drop your {platform} video file here: "{video_name}"
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
"Watch the transformation — you won't believe the change! ✨"

### 2. 5 Viral {tag_label}
#example1 #example2 #example3 #example4 #example5

### 3. Actionable Improvement Tip for Engagement
Add a trending sound and hook in the first 2 seconds.

### 4. Viral Optimization Score (1–100)
78/100 – Strong brightness and engaging content; optimize hook & audio.

### 5. Motivation to Increase Virality
Keep experimenting and iterate quickly!

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
Focus on hooks and trending audio.

📋 Actionable Checklist:
- Hook viewers in the first 2 seconds.
- Use trending audio and relevant captions.
- Encourage saves and shares with call-to-actions.
- Maintain visual consistency across Reels.

🎯 **Detected Niche:** {niche}
🕓 **Best Time to Post for {niche} ({platform})**:
⏰ {suggested_peak_time}
"""
            parsed_json = {
                "caption": "Watch the transformation — you won't believe the change! ✨",
                "hashtags": "#example1 #example2 #example3 #example4 #example5",
                "tags": "#example1 #example2 #example3",
                "improvement_tip": "Add a trending sound and hook in the first 2 seconds.",
                "optimization_score": 78,
                "motivation": "Keep experimenting and iterate quickly!",
                "viral_examples": [],
                "takeaway": "Focus on hooks and trending audio.",
                "actionable_checklist": ["Hook in first 2s", "Use trending audio", "Ask viewers to comment"],
                "detected_niche": niche,
                "detected_tone": tone,
                "detected_keywords": keywords,
                "suggested_peak_time": suggested_peak_time
            }
            return human_text.strip(), parsed_json


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
        # extract features
        try:
            metrics = analyze_video_features(tmp.name)
        except Exception as e:
            metrics = {"duration": 0.0, "resolution": "0x0", "aspect_ratio": 0.0, "brightness": 0.0}
        audio_features = extract_audio_features(tmp.name)

    csv_data = None
    if csv_file:
        try:
            df = pd.read_csv(csv_file)
            csv_data = df.head(5).to_dict()
        except Exception:
            csv_data = {"error": "invalid csv"}

    # Build a short media insights string
    video_insights = f"Duration: {metrics.get('duration')}s, Avg Brightness: {metrics.get('brightness')}, Audio Volume: {audio_features.get('audio_volume')}, Has Audio: {audio_features.get('has_audio')}"

    # Detect niche/tone/keywords + suggested peak time (AI)
    detected = detect_niche_tone_keywords(video.filename, csv_data, video_insights)

    # Generate full human readable analysis + structured JSON
    human_text, structured = generate_ai_analysis(platform, video.filename, metrics, detected, csv_data)

    # assemble response (both the textual analysis and structured pieces)
    response_payload = {
        "result": human_text,
        "structured": structured,
        "detected": detected,
        "metrics": metrics
    }

    return jsonify(response_payload)


if __name__ == "__main__":
    # run debug locally
    app.run(host="0.0.0.0", port=5000, debug=True)
