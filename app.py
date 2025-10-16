<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>TikTok & YouTube Virality Analyzer</title>
  <style>
    body {
      font-family: 'Inter', sans-serif;
      background-color: #0e0e10;
      color: white;
      text-align: center;
      margin: 0;
      padding: 0;
    }

    h1 {
      font-size: 1.8rem;
      margin-top: 30px;
    }

    .container {
      width: 90%;
      max-width: 650px;
      margin: 40px auto;
      padding: 25px;
      background: #1b1b1f;
      border-radius: 16px;
      box-shadow: 0 0 25px rgba(255,255,255,0.05);
    }

    .upload-zone {
      border: 2px dashed #555;
      border-radius: 12px;
      padding: 25px;
      margin-bottom: 20px;
      transition: 0.3s ease;
    }

    .upload-zone:hover {
      border-color: #00ff99;
      background-color: rgba(0,255,153,0.05);
    }

    input[type="file"] {
      display: none;
    }

    label {
      background: #00ff99;
      color: black;
      padding: 10px 20px;
      border-radius: 8px;
      cursor: pointer;
      font-weight: bold;
      margin: 5px;
      display: inline-block;
    }

    select {
      padding: 10px;
      border-radius: 8px;
      border: none;
      background: #2a2a2d;
      color: white;
      margin-bottom: 20px;
      font-size: 1rem;
    }

    button {
      background: #00ff99;
      color: black;
      padding: 12px 30px;
      font-weight: bold;
      border: none;
      border-radius: 8px;
      cursor: pointer;
      font-size: 1rem;
      transition: 0.2s ease;
    }

    button:hover {
      background: #00e68a;
    }

    pre {
      text-align: left;
      background: #161618;
      padding: 15px;
      border-radius: 10px;
      color: #c7c7c7;
      overflow-x: auto;
      white-space: pre-wrap;
      word-break: break-word;
    }

    .file-info {
      font-size: 0.9rem;
      color: #00ff99;
      margin-top: 5px;
    }

    .loading {
      margin-top: 20px;
      color: #00ff99;
      font-weight: bold;
      font-size: 1.1rem;
    }
  </style>
</head>
<body>
  <div class="container">
    <h1>🎬 TikTok & YouTube Virality Analyzer</h1>

    <select id="platform">
      <option value="tiktok">TikTok</option>
      <option value="youtube">YouTube</option>
    </select>

    <div class="upload-zone">
      <h3>🎥 Upload Video</h3>
      <label for="videoInput">Choose Video</label>
      <input type="file" id="videoInput" accept="video/*" />
      <div id="videoInfo" class="file-info"></div>
    </div>

    <div class="upload-zone">
      <h3>📈 Upload Performance CSV (optional)</h3>
      <label for="csvInput">Choose CSV</label>
      <input type="file" id="csvInput" accept=".csv" />
      <div id="csvInfo" class="file-info"></div>
    </div>

    <button id="analyzeBtn">Analyze</button>

    <div id="loading" class="loading" style="display:none;">🔍 Analyzing video, please wait...</div>
    <pre id="results"></pre>
  </div>

  <script>
    const analyzeBtn = document.getElementById("analyzeBtn");
    const videoInput = document.getElementById("videoInput");
    const csvInput = document.getElementById("csvInput");
    const resultsEl = document.getElementById("results");
    const loadingEl = document.getElementById("loading");
    const videoInfo = document.getElementById("videoInfo");
    const csvInfo = document.getElementById("csvInfo");
    const platformSelect = document.getElementById("platform");

    videoInput.addEventListener("change", () => {
      if (videoInput.files.length > 0) {
        videoInfo.textContent = `🎬 Uploaded: ${videoInput.files[0].name}`;
      } else {
        videoInfo.textContent = "";
      }
    });

    csvInput.addEventListener("change", () => {
      if (csvInput.files.length > 0) {
        csvInfo.textContent = `📊 Uploaded: ${csvInput.files[0].name}`;
      } else {
        csvInfo.textContent = "";
      }
    });

    analyzeBtn.addEventListener("click", async () => {
      const platform = platformSelect.value;
      const videoFile = videoInput.files[0];
      const csvFile = csvInput.files[0];

      if (!videoFile) {
        alert("Please upload a video file first.");
        return;
      }

      const formData = new FormData();
      formData.append("video", videoFile);
      if (csvFile) formData.append("csv", csvFile);

      loadingEl.style.display = "block";
      resultsEl.textContent = "";

      try {
        const response = await fetch(`/analyze_${platform}`, {
          method: "POST",
          body: formData,
        });
        const data = await response.json();

        if (data.error) {
          resultsEl.textContent = `❌ Error: ${data.error}`;
        } else {
          resultsEl.textContent = data.results || "No results returned.";
        }
      } catch (err) {
        resultsEl.textContent = `⚠️ Request failed: ${err.message}`;
      } finally {
        loadingEl.style.display = "none";
      }
    });
  </script>
</body>
</html>
