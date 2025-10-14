from flask import Flask, request, jsonify, render_template_string
import os
import datetime
import numpy as np
from moviepy.editor import VideoFileClip
import openai

# ========== CONFIG ==========
openai.api_key = os.getenv("OPENAI_API_KEY")
app = Flask(__name__)

# ========== FRONTEND HTML (Dark Mode + Spinner) ==========
HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>TikTok AI Analyzer</title>
<style>
    body {
        font-family: 'Segoe UI', sans-serif;
        background-color: #0d1117;
        color: #e6edf3;
        text-align: center;
        margin-top: 60px;
    }
    h1 {
        color: #39ff14;
        font-weight: 600;
    }
    #drop-zone {
        border: 3px dashed #39ff14;
        border-radius: 15px;
        padding: 60px;
        width: 80%;
        margin: 30px auto;
        background: #161b22;
        cursor: pointer;
        transition: 0.3s;
    }
    #drop-zone:hover {
        background: #21262d;
    }
    #output {
        text-align: left;
        width: 80%;
        margin: 40px auto;
        white-space: pre-wrap;
        background: #161b22;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 0 10px rgba(57, 255, 20, 0.3);
        border: 1px solid #30363d;
    }
    .hidden { display: none; }

    /* Spinner styles */
    .spinner {
        border: 4px solid rgba(255, 255, 255, 0.1);
        border-top: 4px solid #39ff14;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        animation: spin 1s linear infinite;
        margin: 30px auto;
    }
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
</style>
</head>
<body>

<h1>🎬 TikTok AI Analyzer</h1>
<p>Drag and drop your TikTok video below or click to upload</p>
<div id="drop-zone">⬇️ Drop video here or click to upload ⬇️</div>

<div id="spinner" class="spinner hidden"></div>
<div id="output" class="hidden"></div>

<script>
const dropZone = document.getElementById('drop-zone');
const output = document.getElementById('output');
const spinner = document.getElementById('spinner');

dropZone.addEventListener('click', () => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = 'video/*';
    input.onchange = () => uploadFile(input.files[0]);
    input.click();
});

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.style.background = '#20272e';
});

dropZone.addEventListener('dragleave', () => {
    dropZone.style.background = '#161b22';
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.style.background = '#161b22';
    const file = e.dataTransfer.files[0];
    if (file) uploadFile(file);
});

async function uploadFile(file) {
    output.classList.remove('hidden');
    spinner.classList.remove('hidden');
    output.textContent = '🎥 Running TikTok Viral Optimizer...\\n\\n🤖 Generating AI-powered analysis... please wait...';
    const formData = new FormData();
    formData.append('video', file);

    try {
        const res = await fetch('/analyze', { method: 'POST', body: formData });
        const data = await res.json();
        spinner.classList.add('hidden');
        output.textContent = data.result || '⚠️ Something went wrong.';
    } catch (err) {
        spinner.classList.add('hidden');
        output.textContent = '❌ Error: ' + err.message;
    }
}
</script>
</body>
</html>
"""

# ========== HELPER FUNCTIONS ==========
def analyze_video_properties(video_path):
    try:
        clip = VideoFileClip(video_path)
        duration = clip.duration
        resolution = clip.size
        frame = clip.get_frame(0)
        brightness = np.mean(frame)
        aspect_ratio = resolution[0] / resolution[1]
        tone = "bright" if brightness > 150 else "dark" if brightness < 70 else "neutral or mixed"
        heuristic_score = round(np.clip((brightness / 255) * 10, 1, 10), 1)
        clip.close()
        return duration, resolution, brightness, aspect_ratio, tone, heuristic_score
    except Exception as e:
        return 0, (0, 0), 0, 0, "unknown", 0

# ========== ROUTES ==========
@app.route('/')
def home():
    return render_template_string(HTML_PAGE)

@app.route('/analyze', methods=['POST'])
def analyze_video():
    try:
        file = request.files['video']
        temp_path = f"temp_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.mp4"
        file.save(temp_path)

        duration, resolution, brightness, aspect_ratio, tone, heuristic_score = analyze_video_properties(temp_path)
        filename = file.filename

        # Stronger contextual prompt
        prompt = f"""
You are a TikTok marketing expert. Analyze this TikTok video based on its visuals and content details:

File name: {filename}
Duration: {duration:.2f}s
Resolution: {resolution}
Brightness: {brightness:.2f}
Aspect Ratio: {aspect_ratio:.3f}
Tone: {tone}
Heuristic Score: {heuristic_score}/10

Identify the niche from visual and filename context (not just the title).
Then return results formatted exactly as follows (do not add extra text):

🎬 Drag and drop your TikTok video file here: "{filename}"
🎥 Running TikTok Viral Optimizer...

🤖 Generating AI-powered analysis, captions, and viral tips...

🔥 Fetching viral video comparisons and strategic insights...

✅ TikTok Video Analysis Complete!

🎬 Video: {filename}
📏 Duration: {duration:.2f}s
🖼 Resolution: {resolution[0]}x{resolution[1]}
📱 Aspect Ratio: {aspect_ratio:.3f}
💡 Brightness: {brightness:.2f}
🎨 Tone: {tone}
⭐ Heuristic Score: {heuristic_score}/10

💬 AI-Generated Viral Insights:
### 1. Scroll-Stopping Caption
(Generate a caption suitable for the niche)

### 2. 5 Viral Hashtags
(Generate 5 hashtags related to the specific niche)

### 3. Actionable Improvement Tip for Engagement
(Give a tactical improvement)

### 4. Viral Optimization Score (1–100)
(Assign a realistic score and explain briefly)

### 5. Short Motivation on How to Increase Virality
(Make it niche-specific)

🔥 Viral Comparison Results:
### Comparison with Viral TikToks in the Same Niche
(Provide 3 relevant viral examples with why they worked and how to replicate)

### Takeaway Strategy
(Provide actionable summary for creator)

📋 Actionable Checklist:
   - Hook viewers in under 2 seconds.
   - Add trending sound if relevant.
   - Post during high activity times (Fri–Sun, 6–10pm).
   - Encourage comments by asking a question.
"""

        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        result_text = response.choices[0].message.content.strip()

        os.remove(temp_path)
        return jsonify({"result": result_text})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ========== MAIN ==========
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
