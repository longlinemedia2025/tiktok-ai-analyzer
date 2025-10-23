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
import datetime
import csv
import io

app = Flask(__name__, template_folder="templates")
CORS(app)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

progress_status = {"step": "idle", "percent": 0}


# ✅ Automatic Intent Detection (unchanged)
def detect_video_intent(text_analysis, audio_analysis, visual_analysis):
    """Automatically infer video intent based on text, audio, and visual cues."""
    try:
        combined = f"{text_analysis} {audio_analysis} {visual_analysis}".lower()

        if any(word in combined for word in ["listen", "stream now", "new track", "music video", "rapper", "beat", "producer", "song out", "album", "spotify"]):
            return "Music Promotion"
        if any(word in combined for word in ["sale", "shop now", "discount", "product", "offer", "brand", "business", "marketing", "client", "store", "buy"]):
            return "Business Promotion"
        if any(word in combined for word in ["tutorial", "how to", "guide", "tips", "learn", "educational", "explaining", "course"]):
            return "Educational Content"
        if any(word in combined for word in ["motivation", "inspiration", "mindset", "success", "growth", "positive", "grind", "discipline"]):
            return "Motivational Content"
        if any(word in combined for word in ["fashion", "outfit", "style", "ootd", "lookbook", "model", "streetwear", "aesthetic"]):
            return "Lifestyle / Fashion"
        if any(word in combined for word in ["funny", "comedy", "skit", "laugh", "relatable", "joke", "prank"]):
            return "Entertainment / Skit"

        return "General / Other"

    except Exception as e:
        print("⚠ Error during intent detection:", e)
        return "General / Other"


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


def extract_audio_from_video(video_path):
    """Extract audio for speech/music analysis."""
    update_progress("Extracting and analyzing audio...", 25)
    try:
        audio_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        os.system(f'ffmpeg -i "{video_path}" -vn -acodec pcm_s16le -ar 16000 -ac 1 "{audio_path}" -y')

        with open(audio_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1", file=audio_file
            )
        os.remove(audio_path)
        text_output = transcript.text if hasattr(transcript, "text") else "No transcription detected."
        update_progress("Audio analysis complete", 35)
        return text_output
    except Exception as e:
        print("⚠ Audio extraction failed:", e)
        return "No audio detected."


def analyze_video(video_path):
    frames = extract_video_frames(video_path)
    text_detected, visual_descriptions = [], []
    total_steps = min(5, len(frames))

    for i, frame in enumerate(frames[:5]):
        step_pct = 35 + int((i + 1) / total_steps * 45)
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
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}]
                    }
                ]
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
                                {"type": "text", "text": "Describe what is happening in this single frame from a social video."},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}]
                        }
                    ]
                )
                visual_descriptions.append(response.choices[0].message.content)
        except Exception as e:
            print("⚠ Final fallback failed:", e)

    update_progress("Finished frame analysis", 85)
    return " ".join(visual_descriptions), " ".join(text_detected)


def analyze_csv(csv_file):
    """Read CSV for personalization — use trends or performance data."""
    try:
        content = csv_file.read().decode("utf-8")
        reader = csv.DictReader(io.StringIO(content))
        data = [row for row in reader]
        return data
    except Exception as e:
        print("⚠ CSV read error:", e)
        return []


def generate_ai_report(video_path, platform, csv_data=None):
    update_progress("Generating AI report...", 90)
    visuals, extracted_text = analyze_video(video_path)
    audio_text = extract_audio_from_video(video_path)

    # ✅ Detect intent dynamically
    detected_intent = detect_video_intent(extracted_text, audio_text, visuals)
    print(f"[Intent Detection] → {detected_intent}")

    today = datetime.datetime.now().strftime("%A")
    csv_context = f"The CSV provided contains {len(csv_data)} past posts for insight." if csv_data else "No CSV data provided."

    prompt_context = f"""
Detected Intent: {detected_intent}
Day: {today}
Platform: {platform}
{csv_context}

Please tailor the viral analysis, captioning, and timing based on the intent and niche.
"""

    prompt = f"""
{prompt_context}

You are an expert social media analyst for {platform}. Use the following inputs:

Visuals: {visuals or 'No visuals found.'}
Text: {extracted_text or 'No text detected.'}
Audio transcript: {audio_text or 'No audio detected.'}

Generate a full viral analysis in Markdown format:

# AI Results

### 🎬 Video Summary
Summarize what the video communicates visually and audibly.

### 💬 AI-Generated Viral Insights:
1️⃣ Scroll-Stopping Caption ({platform})
2️⃣ 5 Hashtags ({platform})
3️⃣ Engagement Tip
4️⃣ Viral Optimization Score (1–100)
5️⃣ Motivational Tip

### 🔥 Viral Comparison (Recent Examples):
Provide 3 **recent (within the last 3 months)** viral examples from the same niche.
For each:
- 🎥 **Summary**
- 💡 **Why it went viral**
- 🔁 **How to replicate**

### 🧠 Optimization Advice:
Give 3–5 precise improvement tips.

🎯 **Detected Niche**
🎯 **Detected Intent** — briefly describe what the video is trying to achieve (e.g., promote, entertain, educate, inspire).
🕓 **Best Time to Post ({platform}, {today})** — based on niche and current trend data.
💡 **Peak Engagement Hour** — estimate from similar posts or CSV history.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            temperature=0.7,
            messages=[
                {"role": "system", "content": "You are a professional viral strategist that incorporates real-time data and recent examples."},
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
    csv_data = None

    if "csv" in request.files and request.files["csv"].filename:
        csv_file = request.files["csv"]
        csv_data = analyze_csv(csv_file)

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
