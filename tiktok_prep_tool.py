import os
import cv2
import numpy as np
import csv
from datetime import datetime
from moviepy.editor import VideoFileClip
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# === Detect dominant tone from visual content ===
def detect_video_tone(video_path):
    cap = cv2.VideoCapture(video_path)
    total_brightness, colorfulness, frame_count = 0, 0, 0

    while True:
        ret, frame = cap.read()
        if not ret or frame_count >= 10:
            break
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        total_brightness += np.mean(hsv[:, :, 2])
        colorfulness += np.std(frame)
        frame_count += 1

    cap.release()

    avg_brightness = total_brightness / max(frame_count, 1)
    avg_colorfulness = colorfulness / max(frame_count, 1)

    if avg_brightness < 40:
        tone = "dark or suspenseful"
    elif avg_colorfulness > 60 and avg_brightness > 120:
        tone = "energetic or vibrant"
    elif avg_brightness > 80 and avg_colorfulness < 40:
        tone = "calm or aesthetic"
    else:
        tone = "neutral or mixed"

    return tone, avg_brightness, avg_colorfulness


# === AI Caption + Hashtag + Viral Optimization ===
def generate_ai_insights(video_path, duration, resolution, brightness, aspect_ratio, tone):
    try:
        video_name = os.path.basename(video_path)

        prompt = f"""
        You are a professional TikTok strategist who specializes in creating viral content.

        Analyze the video stats and provide:
        1. A scroll-stopping caption
        2. 5 viral hashtags suited for its niche and tone
        3. One actionable improvement tip for engagement
        4. A Viral Optimization Score (1–100)
        5. A short motivation on how to increase virality

        Context:
        - Title: {video_name}
        - Duration: {duration:.2f} seconds
        - Resolution: {resolution[0]}x{resolution[1]}
        - Brightness: {brightness:.2f}
        - Aspect Ratio: {aspect_ratio:.3f}
        - Tone: {tone}

        If it's gaming, make it energetic and hype.
        If it's haircut/self-care, focus on transformation.
        If it's horror, emphasize tension.
        If it's storytelling, lean into emotion.
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert TikTok strategist trained to optimize content for virality."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.85,
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"(AI generation failed: {str(e)})"


# === NEW: Viral Comparison Engine ===
def generate_viral_comparison(video_path, tone):
    try:
        video_name = os.path.basename(video_path)

        prompt = f"""
        You are an advanced TikTok algorithm analyst.
        Compare the video titled "{video_name}" with 3 real-world viral TikToks in the same niche and tone: {tone}.

        For each viral example, provide:
        - Video concept summary (one sentence)
        - What made it go viral (hook, pacing, emotion, or style)
        - How the user can replicate its success organically

        End with a short summary called **"Takeaway Strategy"** that describes
        what the user can learn from these viral examples and how to apply it.

        Keep it practical and motivating.
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a viral trend researcher for TikTok content."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.9,
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"(Viral comparison failed: {str(e)})"


# === Video Stats & Heuristic Analysis ===
def analyze_video(video_path):
    clip = VideoFileClip(video_path)
    duration = clip.duration
    resolution = clip.size
    aspect_ratio = resolution[0] / resolution[1]

    tone, avg_brightness, colorfulness = detect_video_tone(video_path)

    score = 10
    if duration > 60:
        score -= 2
    if avg_brightness < 30:
        score -= 1
    if resolution[0] < 720:
        score -= 1

    score = max(0, min(score, 10))
    return duration, resolution, aspect_ratio, avg_brightness, tone, score


# === Main Processing ===
def process_video(video_path):
    print("🎥 Running TikTok Viral Optimizer...\n")

    duration, resolution, aspect_ratio, brightness, tone, score = analyze_video(video_path)

    print("🤖 Generating AI-powered analysis, captions, and viral tips...\n")
    ai_output = generate_ai_insights(video_path, duration, resolution, brightness, aspect_ratio, tone)

    print("🔥 Fetching viral video comparisons and strategic insights...\n")
    comparison_output = generate_viral_comparison(video_path, tone)

    print("✅ TikTok Video Analysis Complete!\n")
    print(f"🎬 Video: {os.path.basename(video_path)}")
    print(f"📏 Duration: {duration:.2f}s")
    print(f"🖼 Resolution: {resolution[0]}x{resolution[1]}")
    print(f"📱 Aspect Ratio: {aspect_ratio:.3f}")
    print(f"💡 Brightness: {brightness:.2f}")
    print(f"🎨 Tone: {tone}")
    print(f"⭐ Heuristic Score: {score}/10\n")

    print("💬 AI-Generated Viral Insights:")
    print(ai_output)
    print("\n🔥 Viral Comparison Results:")
    print(comparison_output)
    print()

    print("📋 Actionable Checklist:")
    print("   - Hook viewers in under 2 seconds.")
    print("   - Add trending sound if relevant.")
    print("   - Post during high activity times (Fri–Sun, 6–10pm).")
    print("   - Encourage comments by asking a question.\n")

    # Save results to CSV
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"tiktok_video_analysis_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv")

    with open(output_file, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file, delimiter="\t")
        writer.writerow([
            "Filename", "Duration (s)", "Resolution", "Aspect Ratio",
            "Brightness", "Tone", "Score (0-10)", "AI Insights", "Viral Comparisons", "Checklist"
        ])
        writer.writerow([
            os.path.basename(video_path),
            round(duration, 2),
            f"{resolution[0]}x{resolution[1]}",
            round(aspect_ratio, 3),
            round(brightness, 2),
            tone,
            score,
            ai_output,
            comparison_output,
            "Hook fast | Use trending sound | Post at peak hours"
        ])

    print(f"📁 Results saved to: {output_file}")


# === Run Script ===
if __name__ == "__main__":
    video_path = input("🎬 Drag and drop your TikTok video file here: ").strip().strip('"')
    process_video(video_path)
