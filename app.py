from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import cv2
import base64
import numpy as np
import tempfile
import threading
from openai import OpenAI
import pytesseract
from PIL import Image
from pydub import AudioSegment
import json
import datetime

app = Flask(__name__, template_folder="templates")
CORS(app)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

progress_status = {"step": "idle", "percent": 0}

def update_progress(step, percent):
    global progress_status
    progress_status["step"] = step
    progress_status["percent"] = percent
    print(f"[Progress] {step} ({percent}%)")

def extract_video_frames(video_path, frame_rate=1):
    update_progress("Extracting video frames...", 10)
    frames = []
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_interval = int(fps * frame_rate)
    count = 0
    success, frame = cap.read()
    while success:
        if count % frame_interval == 0:
            frames.append(frame)
        success, frame = cap.read()
        count += 1
    cap.release()
    update_progress(f"Extracted {len(frames)} frames", 20)
    return frames

def extract_audio_text(video_path):
    """Extract spoken words or key audio cues from the video using Whisper."""
    try:
        update_progress("Extracting and analyzing audio...", 25)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
            audio = AudioSegment.from_file(video_path)
            audio.export(tmp_audio.name, format="wav")
            tmp_audio_path = tmp_audio.name

        with open(tmp_audio_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="gpt-4o-mini-transcribe",
                file=audio_file
            )
        os.remove(tmp_audio_path)
        print("🎧 Audio detected successfully.")
        return transcription.text
    except Exception as e:
        print("⚠ Audio extraction failed:", e)
        return None

def analyze_video(video_path):
    frames = extract_video_frames(video_path)
    text_detected, visual_descriptions = [], []
    total_steps = min(5, len(frames))

    for i, frame in enumerate(frames[:5]):
        step_pct = 20 + int((i + 1) / total_steps * 60)
        update_progress(f"Analyzing frame {i + 1}/{total_steps}...", step_pct)
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        try:
            text = pytesseract.image_to_string(pil_img)
            if text.strip():
                text_detected.append(text.strip())
        except Exception as e:
            print("⚠ OCR skipped:", e)

        try:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_img:
                pil_img.save(temp_img.name)
                with open(temp_img.name, "rb") as f:
                    img_b64 = base64.b64encode(f.read()).decode("utf-8")

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a social media vision analyzer."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe this image briefly for algorithmic analysis."},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                        ],
                    },
                ],
            )
            desc = response.choices[0].message.content
            if desc:
                visual_descriptions.append(desc)
        except Exception as e:
            print("⚠ Vision model error:", e)
            continue

    if not visual_descriptions:
        update_progress("Fallback visual check...", 80)
        try:
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            cap.release()
            if ret:
                _, buffer = cv2.imencode(".jpg", frame)
                img_b64 = base64.b64encode(buffer).decode("utf-8")
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a visual recognition assistant."},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Describe what is happening in this single frame from a TikTok-style video."},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                            ],
                        },
                    ],
                )
                visual_descriptions.append(response.choices[0].message.content)
        except Exception as e:
            print("⚠ Final fallback failed:", e)

    update_progress("Finished frame analysis", 85)
    return " ".join(visual_descriptions), " ".join(text_detected)

def generate_ai_report(video_path, platform, csv_data=None):
    update_progress("Generating AI report...", 90)
    visuals, extracted_text = analyze_video(video_path)
    audio_text = extract_audio_text(video_path) or "No audio transcription available."

    historical_context = ""
    if csv_data:
        historical_context = f"\nHistorical performance data:\n{csv_data}\n"

    today = datetime.datetime.now().strftime("%A, %B %d, %Y")

    prompt = f"""
You are an expert social media analyst. Analyze this {platform} video.

Date: {today}
Visual analysis: {visuals or 'No visual data.'}
Text detected: {extracted_text or 'No text detected.'}
Audio content: {audio_text}
{historical_context}

Return in Markdown format:

# AI Results

### 🎬 Video Summary
Summarize what the algorithm will perceive in the visuals and audio.

### 💬 AI-Generated Viral Insights:
1️⃣ Scroll-Stopping Caption ({platform})
2️⃣ 5 Hashtags ({platform})
3️⃣ Engagement Tip
4️⃣ Viral Optimization Score (1–100)
5️⃣ Motivational Tip

### 🔥 Viral Comparison:
Use **CURRENT** viral examples from the same detected niche (past few months). For each:
- 🎥 **Summary**
- 💡 **Why it went viral**
- 🔁 **How to replicate**

### 🧠 Optimization Advice:
Give 3–5 specific suggestions to improve performance.

🎯 Detected Niche
🕓 Best Time to Post ({platform}, {today})
⏰ Peak EST posting window
💡 Peak engagement hour (based on niche + current trend data)
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            temperature=0.7,
            messages=[
                {"role": "system", "content": "You are a professional viral strategist that uses real 2025 trend data."},
                {"role": "user", "content": prompt},
            ],
        )
        update_progress("Analysis complete!", 100)
        return response.choices[0].message.content
    except Exception as e:
        print("⚠ AI generation error:", e)
        update_progress("AI generation failed", 100)
        return None

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/progress")
def get_progress():
    return jsonify(progress_status)

@app.route("/analyze", methods=["POST"])
def analyze():
    if "video" not in request.files:
        return jsonify({"error": "No video provided"}), 400

    video = request.files["video"]
    platform = request.form.get("platform", "TikTok")
    csv_file = request.files.get("csv")
    csv_data = None
    if csv_file:
        csv_data = csv_file.read().decode("utf-8")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        video.save(tmp.name)
        video_path = tmp.name

    def background_analysis():
        try:
            result = generate_ai_report(video_path, platform, csv_data)
            progress_status.update(
                {"step": "Analysis complete!", "percent": 100, "result": result or "⚠ No output."}
            )
        except Exception as e:
            print("❌ Background thread error:", e)
            progress_status.update({"step": "Error occurred", "percent": 100, "result": str(e)})
        finally:
            try:
                os.remove(video_path)
            except:
                pass

    threading.Thread(target=background_analysis).start()
    return jsonify({"status": "Processing started"})

if __name__ == "__main__":
    app.run(debug=True)
