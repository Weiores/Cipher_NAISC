# Security Perception Layer

FastAPI scaffold for a multimodal perception layer that accepts CCTV or bodycam inputs, runs three perception adapters, and produces a fused event summary.

## Included services

- Weapon detection adapter intended for a fine-tuned YOLO model
- Video emotion detection adapter for face/emotion inference
- Bodycam audio adapter for tone, stress, and acoustic event detection
- Trait fusion service that normalizes model outputs into a single event package

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/docs`.

## Endpoints

- `GET /health`
- `GET /perception/models`
- `POST /perception/infer`

## Notes

This scaffold uses deterministic placeholder inference so the API is runnable without shipping large model weights.

Recommended production backends:

- Weapon: Ultralytics YOLO fine-tuned for CCTV and bodycam weapon classes
- Emotion: face detector + emotion classifier
- Audio: VAD + Whisper transcription + speech emotion + acoustic event classifier

## Real emotion model

The main app can use the `fer` package for facial emotion detection from images or sampled video frames.

Supported normalized outputs:

- `angry`
- `fearful`
- `distressed`
- `neutral`
- `unknown`

## Real audio analysis

The current audio path now performs real waveform analysis on audio streams or embedded video audio using `ffmpeg` extraction plus signal features.

Current outputs are based on:

- decoded mono audio at 16 kHz
- frame energy
- zero-crossing rate
- peak intensity

This supports real `speech_present`, `tone`, and basic `acoustic_events`.

If Whisper is installed, the audio path also attempts transcription and returns:

- `transcript`
- `keyword_flags`

## Default YOLO weapon model

Set `WEAPON_MODEL_PATH` to a trained Ultralytics model such as `best.pt`. When `video.uri` points to a local image or video file, the weapon adapter will use YOLO directly instead of the placeholder path.

```powershell
$env:WEAPON_MODEL_PATH='C:\models\weapons\best.pt'
.\.venv\Scripts\python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Example request using a local frame or clip:

```json
{
  "source_id": "cctv_lobby_01",
  "video": {
    "stream_type": "cctv",
    "uri": "C:\\evidence\\frame_001.jpg",
    "camera_id": "lobby_01",
    "frame_sample_fps": 2.0
  }
}
```

The weapon output now includes a `backend` field so you can see whether the result came from `ultralytics_yolo`, `tf2_saved_model`, or the placeholder fallback.

Default local YOLO model:

- `<perception-layer>\Weapons-and-Knives-Detector-with-YOLOv8\runs\detect\Normal_Compressed\weights\best.pt`
