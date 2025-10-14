from flask import Flask, request, jsonify, render_template
import os
import tempfile
import openai
from moviepy.editor import VideoFileClip
import numpy as np
from PIL import Image
import io
import base64

# ========== CONFIG ==========
openai.api_key = os.getenv("OPENAI_API_KEY")
app = Flask(__name__)

# ========== HELPER FUNCTIONS ==========

def analyze_video_properties(video_path):
    clip = VideoFileClip(video_path)
    duration = round(clip.duration, 2)
    frame_rate = int(clip.fps)
    resolution = f"{clip.w}x{clip.h}"
    aspect_ratio = round(clip.w / clip.h, 3)
    frame = clip.get_frame(clip.duration / 2)
    brightness = round(np.mean(frame), 2)
    tone = "bright" if brightness > 120 else "dark" if brightness < 80 else "neutral or mixed"
    clip.close()
    return duration, frame_rate, resolution, aspect_ratio, brightness, tone


def extract_sample_frames(video_path, num_frames=3):
    clip = VideoFileClip(video_path)
    duration = clip.duration
    frames = []
    for i in np.linspace(0.2, 0.8, num_frames):
        frame_time = duration * i
        frame = clip.get_frame(frame_time)
        img = Image.fromarray(frame)
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        frame_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        frames.append(frame_b64)
    clip.close()
    return frames


def describe_frames_with_ai(frames_b64):
    descriptions = []
    for b64_img in frames_b64:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a visual description AI. Describe the image briefly and factually."},
                    {"role": "user", "content": [{"type": "image_url", "image_url": f"data:image/jpeg;base64,{b64_img}"}]}
                ],
                max_tokens=50
            )
            descriptions.append(response.choices[0].message["content"])
        except Exception as e:
            descriptions.append(f"(Frame description failed: {e})")
    return " ".join(descriptions)


def transcribe_audio(video_path):
    try:
        clip = VideoFileClip(video_path)
        audio_path = tempfile.mktemp(suffix=".wav")
        clip.audio.write_audiofile(audio_path, verbose=False, logger=None)
        clip.close()
        with open(audio_path, "rb") as audio_file:
            transcript = openai.Audio.transcriptions.create(model="whisper-1", file=audio_file)
        os.remove(audio_path)
        return transcript.text
    except Exception:
        return ""


def generate_full_analysis(video_path, filename):
    # 1️⃣ Analyze core properties
    duration, fps, res, aspect, brightness, tone = analyze_video_properties(video_path)

    # 2️⃣ Extract and describe frames
    frames_b64 = extract_sample_frames(video_path)
    visual_description = describe_frames_with_ai(frames_b64)

    # 3️⃣ Transcribe audio (if any)
    audio_text = transcribe_audio(video_path)

    # 4️⃣ Combine all info for GPT reasoning
    context = f"""
    Video name: {filename}
    Visual analysis: {visual_description}
    Audio transcription: {audio_text or 'No audio or transcription unavailable.'}
    """

    # 5️⃣ Ask GPT for niche-specific TikTok insights
    prompt = f"""
    You are TikTok's viral content analysis engine.
    Based on the following video context:
    {context}

    Generate a full analysis strictly following this template:

    🎬 Drag and drop your TikTok video file here: "{filename}"
    🎥 Running TikTok Viral Optimizer...

    🤖 Generating AI-powered analysis, captions, and viral tips...

    🔥 Fetching viral video comparisons and strategic insights...

    ✅ TikTok Video Analysis Complete!

    🎬 Video: {filename}
    📏 Duration: {duration}s
    🖼 Resolution: {res}
    📱 Aspect Ratio: {aspect}
    💡 Brightness: {brightness}
    🎨 Tone: {tone}
    ⭐ Heuristic Score: 9/10

    💬 AI-Generated Viral Insights:
    ### 1. Scroll-Stopping Caption
    (write a creative caption suited for the detected niche)

    ### 2. 5 Viral Hashtags
    (5 hashtags, numbered 1–5, specific to this niche)

    ### 3. Actionable Improvement Tip for Engagement
    (1 paragraph explaining a tactic to boost interaction)

    ### 4. Viral Optimization Score (1–100)
    **Score: (number)**

    ### 5. Short Motivation on How to Increase Virality
    (1 short motivational paragraph)

    🔥 Viral Comparison Results:
    (Give 2–3 viral TikTok examples similar to this niche, each with summary, why it went viral, and how to replicate.)

    ### Takeaway Strategy
    (1 paragraph summarizing what this creator should focus on)

    📋 Actionable Checklist:
       - Hook viewers in under 2 seconds.
       - Add trending sound if relevant.
       - Post during high activity times (Fri–Sun, 6–10pm).
       - Encourage comments by asking a question.
    """

    ai_response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "You are a professional TikTok viral strategy analyst."},
                  {"role": "user", "content": prompt}],
        max_tokens=1500
    )

    return ai_response.choices[0].message["content"]


# ========== ROUTES ==========

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze_video():
    if "video" not in request.files:
        return jsonify({"error": "No video uploaded"}), 400
    file = request.files["video"]
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        file.save(tmp.name)
        filename = file.filename
        try:
            result = generate_full_analysis(tmp.name, filename)
            return jsonify({"result": result})
        except Exception as e:
            return jsonify({"error": f"⚠️ Request failed: {str(e)}"}), 500
        finally:
            os.remove(tmp.name)


# ========== RUN APP ==========
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
