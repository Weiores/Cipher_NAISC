# START SERVERS - QUICK REFERENCE

## Telegram Alert Demo

The Telegram operator alert service lives in `ui-layer/`.

```bash
cd NAISC/ui-layer
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
uvicorn app.main:app --reload --port 8010
```

Use `POST /telegram/alert` with a `ReasoningOutput` payload or the sample file in `ui-layer/samples/reasoning_alert.json`.

### Terminal 1: Launch all 3 servers
```bash
cd Cipher_NAISC_ver1\perception-layer\security-perception-layer
python start_all_servers.py
```

**Expected output**:
```
======================================================================
ALL SERVERS STARTED SUCCESSFULLY
======================================================================

ENDPOINT SUMMARY:

  Dashboard UI              → http://localhost:8000          (Main UI - Select 'Video Testing')
  Video API                 → http://localhost:8001          (Batch processing (legacy))
  Real-Time API             → http://localhost:8002          (Live frame analysis (RECOMMENDED))
  Ollama API                → http://localhost:11434         (Cloud reasoning (LLM))
```

---

## Access Your Dashboard 
This dashboard is just for visualising the detection result

### Dashboard URL
```
http://localhost:8000/dashboard
```

### API Health Checks
```
Main API:  http://localhost:8000/health
Video API: http://localhost:8001/api/health
```

---

## Test It Works

### In Browser
1. Open http://localhost:8000/dashboard
2. Click "Test Scenarios" tab
3. Click "Armed Intruder" button
4. Should show CRITICAL threat → "immediate_alert" action

### Or Use Command Line
```bash
# Check Main API
curl http://localhost:8000/health

# Check Video API
curl http://localhost:8001/api/health
```

---

## Add Your Videos

### Step 1: Create videos folder (if doesn't exist)
```bash
cd Cipher_NAISC_ver1\perception-layer\security-perception-layer
mkdir videos
```

### Step 2: Copy videos there
```
videos/
├── my_video.mp4
├── test.avi
└── camera_feed.mov
```

### Step 3: Process in Dashboard
1. Go to "Video Testing" tab
2. Click "Refresh Video List"
3. Select video
4. Click "Process Video"

---

## Stop Servers

Close the terminal windows (Ctrl+C in each)

---

## What Each Server Does

| Server | Port | Purpose | Status |
|--------|------|---------|--------|
| Main API | 8000 | Dashboard, perception, reasoning | FastAPI |
| Video API | 8001 | Video processing, frame analysis | HTTP |

Both must be running for full functionality

---

