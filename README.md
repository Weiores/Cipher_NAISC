# Cipher_NAISC – AI Security Surveillance System

An AI-powered surveillance system that combines YOLOv8 weapon detection,
face emotion analysis, audio tone classification, and LLM reasoning (via
**Groq API**) to detect threats, alert operators via Telegram, and learn
from officer feedback.



## Architecture Overview

```
Video Source (webcam / file / RTSP)
          │
          ▼
┌─────────────────────┐
│   Video Processor   │  src/video_processor.py
│  (frame sampling)   │
└──────────┬──────────┘
           │ frames
           ▼
┌─────────────────────┐
│  Perception Layer   │  perception-layer/perception_layer.py
│  ┌───────────────┐  │
│  │ Weapon (YOLO) │  │  → weapon_detector.py
│  │ Emotion (FER) │  │  → emotion_detector.py
│  │ Tone (librosa)│  │  → tone_detector.py
│  │ Uniform (YOLO)│  │  → uniform_detector.py
│  └───────────────┘  │
└──────────┬──────────┘
           │ PerceptionResult
           ▼
    Danger? ──NO──▶  log "clear", continue loop
           │
          YES
           │
           ▼
┌─────────────────────┐
│  Telegram Alert #1  │  src/alert_manager.py
│  (initial detection)│
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Learning Agent     │  learning-layer/learning_agent.py
│  (similar past      │  TF-IDF cosine similarity
│   incidents)        │
└──────────┬──────────┘
           │ historical context
           ▼
┌─────────────────────┐
│  Reasoning Agent    │  reasoning-layer/reasoning_agent.py
│  Groq llama-3.3-70b │  → summarise + determine action
└──────────┬──────────┘
           │ ReasoningResult
           ▼
┌─────────────────────┐
│  Incident Database  │  src/incident_database.py (SQLite)
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Telegram Alert #2  │  src/alert_manager.py
│  (summary + action) │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐       ┌──────────────────────┐
│  Officer Response   │◀─────▶│  React Dashboard     │
│  API (FastAPI :8000)│       │  (frontend/ :5173)   │
└──────────┬──────────┘       └──────────────────────┘
           │
           ▼
    Feedback loop → Incident Database → Learning Agent
```

---

## Installation

### Python backend

```bash
# 1. Clone the repo
git clone https://github.com/your-org/Cipher_NAISC.git
cd Cipher_NAISC

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy environment config
cp .env.example .env
# Edit .env with your keys (see table below)
```

### React frontend

```bash
cd frontend
npm install
cp .env.example .env   # set VITE_API_URL if needed
```

---


Use the example .env file to enter API key before renaming it to .env



1. Start Backend
cd C:\Users\lerattoo\Downloads\Cipher_NAISC-main\Cipher_NAISC-main
venv\Scripts\activate
python src/main.py

2. Start Front end
cd C:\Users\lerattoo\Downloads\Cipher_NAISC-main\Cipher_NAISC-main\frontend
npm run dev

3. Enter Password in http://localhost:3000/login
User: Cipher@test.com
PW : Test123

---

## Set Up Telegram Bot

1. Open Telegram and message **@BotFather**
2. Send `/newbot` and follow the prompts — you'll receive a **bot token**
3. Add the bot to a group or start a private chat with it
4. Get your **chat ID** by visiting:
   `https://api.telegram.org/bot<TOKEN>/getUpdates`
   Look for `"chat":{"id": ...}` in the JSON response
5. Add to `.env`:
   ```
   TELEGRAM_BOT_TOKEN=123456:ABC-...
   TELEGRAM_CHAT_ID=-100123456789
   ```

---

## Demo Mode (Pre-recorded Video)

The system has two operating modes selected automatically from `VIDEO_SOURCE`:

| Mode | `VIDEO_SOURCE` | Behaviour |
|---|---|---|
| **Demo** | file path | YOLOv8 runs on every frame at full speed, video loops forever, no Groq Vision calls |
| **Live** | `0` (webcam) | User must click "Activate" in the dashboard; Groq Vision used only in the uncertain confidence zone (0.15–0.50), rate-limited to 1 call / 10 s |

### Quick demo setup

1. Download a weapon/threat detection test video (e.g. from a public dataset or record your own)
2. Edit `.env`:
   ```
   VIDEO_SOURCE=C:\path\to\demo.mp4
   ```
3. Start the backend:
   ```bash
   python src/main.py
   ```
4. The video loops continuously — detections trigger the full alert pipeline automatically.
5. Switch back to webcam mode at any time: set `VIDEO_SOURCE=0` and restart.

> The dashboard camera panel shows STANDBY in webcam mode until the user clicks "Click to activate live feed". In demo/file mode the feed starts immediately.

---

## Running the System

### Full system (video + API + alerts)

```bash
python src/main.py
```

### Video processor only (CLI)

```bash
python src/video_processor.py --source path/to/video.mp4 --fps 2
python src/video_processor.py --source 0                          # webcam
python src/video_processor.py --source rtsp://192.168.1.100:554/stream
```

### Officer response API only

```bash
python src/officer_response_api.py
```

### React dashboard

```bash
cd frontend
npm run dev
# Open http://localhost:5173
```

---

## Dashboard Tabs

| Tab | Description |
|-----|-------------|
| **Live Monitor** | Real-time detection panels, camera grid, alert feed |
| **Incidents** | Full incident list with officer response form |
| **Analytics** | Stats cards, daily incident chart, action distribution |
| **Simulation** | Upload video for offline analysis, export CSV |

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | — | Groq API key (required for AI reasoning) |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq model to use |
| `TELEGRAM_BOT_TOKEN` | — | Telegram bot token from @BotFather |
| `TELEGRAM_CHAT_ID` | — | Target chat/group ID for alerts |
| `VIDEO_SOURCE` | `0` | Webcam index, file path, or RTSP URL |
| `SAMPLE_FPS` | `2.0` | Frames per second to sample |
| `DANGER_WEAPON_THRESHOLD` | `0.6` | Minimum weapon confidence to trigger alert |
| `DANGER_EMOTION_THRESHOLD` | `0.7` | Minimum emotion confidence for threat combo |
| `WEAPON_MODEL_PATH` | *(bundled)* | Custom YOLOv8 weapon weights path |
| `UNIFORM_MODEL_PATH` | — | YOLOv8 uniform classifier weights path |
| `API_PORT` | `8000` | FastAPI server port |
| `DASHBOARD_URL` | `http://localhost:5173` | Dashboard URL in Telegram Alert #2 |
| `LOG_LEVEL` | `INFO` | Python logging level |
| `GROQ_VISION_ENABLED` | `true` | Enable Groq Vision fallback (webcam mode only) |
| `GROQ_VISION_MIN_INTERVAL` | `10` | Minimum seconds between Groq Vision API calls |

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `GET` | `/incidents?limit=50` | Recent incidents list |
| `GET` | `/incident/{id}` | Single incident detail |
| `POST` | `/incident/{id}/response` | Submit officer response |
| `GET` | `/analytics` | Aggregate statistics |

---

## Project Structure

```
Cipher_NAISC/
├── src/
│   ├── main.py                  # System orchestrator
│   ├── video_processor.py       # Frame decoder + loop
│   ├── alert_manager.py         # Telegram alert sender
│   ├── incident_database.py     # SQLite incident store
│   └── officer_response_api.py  # FastAPI REST API
├── perception-layer/
│   ├── perception_layer.py      # Orchestrator + danger logic
│   ├── weapon_detector.py       # YOLOv8 weapon detection
│   ├── emotion_detector.py      # FER face emotion detection
│   ├── tone_detector.py         # librosa audio tone analysis
│   ├── uniform_detector.py      # Uniform/civilian classifier
│   └── security-perception-layer/  # Full existing pipeline (advanced)
├── reasoning-layer/
│   └── reasoning_agent.py       # Groq-powered reasoning agent
├── learning-layer/
│   └── learning_agent.py        # TF-IDF similarity + stats
├── frontend/                    # React + TypeScript dashboard
│   └── src/pages/
│       ├── DashboardPage.tsx    # 4-tab layout
│       ├── IncidentsTab.tsx     # Incidents + officer response
│       ├── AnalyticsTab.tsx     # Charts + stats
│       └── SimulationTab.tsx    # Video upload + CSV export
├── ui-layer/                    # Existing Telegram alert service
├── data/                        # SQLite database (auto-created)
├── .env.example
└── requirements.txt
```

---

## Existing Components (already implemented)

The repo ships with a production-grade perception pipeline in
`perception-layer/security-perception-layer/` including:

- Full YOLOv8 weapon/knife/gun detector with bounding box visualisation
- FER + OpenCV emotion detector
- librosa + ffmpeg audio tone analyser
- Uniform/civilian classifier
- FastAPI video processing server with real-time WebSocket streaming
- Cloud + local reasoning agents
- Telegram alert service (ui-layer/)
- WebSocket-based React dashboard panels (frontend/)

The `src/` layer provides a simplified integration layer that connects all
components into a single `python src/main.py` command.
