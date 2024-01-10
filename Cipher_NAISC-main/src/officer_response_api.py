"""
FastAPI officer response API for Cipher_NAISC.

Officers log their actions via this REST API.  The database is updated and
the learning agent is notified on each new response.

Endpoints:
  POST /incident/{incident_id}/response
  GET  /incident/{incident_id}
  GET  /incidents?limit=50
  GET  /analytics
  GET  /health
  WS   /ws   — live SentinelMessage stream for the dashboard
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Allow imports from project root when running standalone
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "learning-layer"))
sys.path.insert(0, str(_ROOT / "reasoning-layer"))
sys.path.insert(0, str(_ROOT / "perception-layer"))

from fastapi import FastAPI, HTTPException, Response, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from incident_database import IncidentDatabase

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# .env loader
# ---------------------------------------------------------------------------

def _load_env() -> None:
    env_path = _ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_env()

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class OfficerResponseBody(BaseModel):
    """Request body for submitting an officer response."""

    officer_action: str = Field(..., description="Action taken by officer")
    final_outcome: str = Field(..., description="Outcome of the incident")
    feedback: str = Field(default="", description="Optional feedback notes")
    is_false_positive: bool = Field(default=False)


# ---------------------------------------------------------------------------
# Shared state (initialised at module load; safe to reference from lifespan)
# ---------------------------------------------------------------------------

_db = IncidentDatabase()
_learning_agent: Any = None
_ml_model: Any = None


def _get_learning_agent() -> Any:
    global _learning_agent
    if _learning_agent is None:
        try:
            from learning_agent import LearningAgent
            _learning_agent = LearningAgent(db=_db)
        except ImportError:
            pass
    return _learning_agent


def _get_ml_model() -> Any:
    global _ml_model
    if _ml_model is None:
        try:
            from ml_model import CipherMLModel
            _ml_model = CipherMLModel()
        except ImportError:
            pass
    return _ml_model


# ---------------------------------------------------------------------------
# Lifespan — start Telegram feedback listener on startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    feedback_task: asyncio.Task | None = None
    try:
        from alert_manager import run_feedback_listener
        agent = _get_learning_agent()
        feedback_task = asyncio.create_task(
            run_feedback_listener(_db, agent),
            name="telegram-feedback-listener",
        )
        logger.info("[API] Telegram feedback listener started")
    except Exception as exc:
        logger.warning("[API] Could not start feedback listener: %s", exc)

    yield

    if feedback_task is not None:
        feedback_task.cancel()
        try:
            await feedback_task
        except asyncio.CancelledError:
            pass
        logger.info("[API] Telegram feedback listener stopped")


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Cipher_NAISC Officer Response API",
    version="1.0.0",
    description="Log officer actions and retrieve incident data.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict[str, str]:
    """Basic liveness check."""
    return {"status": "ok", "service": "officer-response-api"}


@app.get("/debug/detect-now")
def debug_detect_now() -> dict[str, Any]:
    """Bypasses all pipeline threading: opens webcam directly, grabs one frame,
    runs weapon_detector, saves annotated JPEG, returns raw detection result.
    Use this to verify detection works independently of the stream loop."""
    import cv2
    import numpy as np
    import sys as _sys

    _sys.path.insert(0, str(_ROOT / "perception-layer"))

    result: dict[str, Any] = {"status": "error", "detail": "unknown"}

    try:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            return {"status": "error", "detail": "Could not open webcam index 0"}

        time.sleep(1.0)  # let sensor stabilise
        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            return {"status": "error", "detail": "cap.read() returned no frame"}

        from weapon_detector import WeaponDetector  # type: ignore[import]
        detector = WeaponDetector()
        detection = detector.detect(frame)

        # Save annotated frame so /stream/frame picks it up immediately
        import video_processor as _vp
        _vp._store_frame_jpeg(frame)
        _vp._has_annotated_frame = True

        result = {
            "status": "ok",
            "frame_shape": list(frame.shape),
            "label": detection.label,
            "confidence": round(detection.confidence, 4),
            "confirmed": detection.confirmed,
            "bbox": detection.bbox,
        }
        logger.info(f"[DEBUG-DETECT] Result: {result}")

    except Exception as exc:
        logger.error(f"[DEBUG-DETECT] Failed: {exc}")
        result = {"status": "error", "detail": str(exc)}

    return result


@app.post("/incident/{incident_id}/response")
def submit_response(incident_id: str, body: OfficerResponseBody) -> dict[str, Any]:
    """Log an officer's response to an incident.

    Updates the incident record and notifies the learning agent.
    """
    incident = _db.get_incident(incident_id)
    if incident is None:
        raise HTTPException(status_code=404, detail=f"Incident {incident_id!r} not found")

    updated = _db.update_officer_response(
        incident_id=incident_id,
        officer_action=body.officer_action,
        outcome=body.final_outcome,
        feedback=body.feedback,
        is_false_positive=body.is_false_positive,
    )
    if not updated:
        raise HTTPException(status_code=500, detail="Failed to update incident")

    logger.info(
        "[API] Officer response recorded: incident=%s action=%s fp=%s",
        incident_id, body.officer_action, body.is_false_positive,
    )

    # Notify learning agent (fire-and-forget – no blocking)
    agent = _get_learning_agent()
    if agent is not None:
        try:
            # Refresh the agent's internal data by reloading stats
            agent.get_recommendation_stats()
        except Exception as exc:
            logger.debug("[API] Learning agent refresh failed: %s", exc)

    return {
        "status": "ok",
        "incident_id": incident_id,
        "officer_action": body.officer_action,
        "is_false_positive": body.is_false_positive,
    }


@app.get("/incident/{incident_id}")
def get_incident(incident_id: str) -> dict[str, Any]:
    """Retrieve a single incident by ID."""
    incident = _db.get_incident(incident_id)
    if incident is None:
        raise HTTPException(status_code=404, detail=f"Incident {incident_id!r} not found")
    return incident


@app.get("/incidents")
def list_incidents(limit: int = 50) -> list[dict[str, Any]]:
    """Return recent incidents, newest first."""
    limit = max(1, min(limit, 500))
    return _db.get_recent_incidents(limit=limit)


@app.get("/incident/{incident_id}/agent-reports")
def get_agent_reports(incident_id: str) -> dict[str, Any]:
    """Return the raw agent_reports JSON blob for a swarm-analysed incident."""
    incident = _db.get_incident(incident_id)
    if incident is None:
        raise HTTPException(status_code=404, detail=f"Incident {incident_id!r} not found")
    reports = incident.get("agent_reports")
    if not reports:
        raise HTTPException(status_code=404, detail="No agent reports for this incident")
    return reports if isinstance(reports, dict) else {}


@app.get("/analytics")
def analytics() -> dict[str, Any]:
    """Return aggregate analytics statistics."""
    db_stats = _db.get_analytics()

    # Build learning stats — prefer live agent data, fall back to DB-computed values
    learning_stats: dict[str, Any] = {
        "total_incidents":        db_stats.get("total_incidents", 0),
        "responded":              db_stats.get("responded_count", 0),
        "recommendation_accuracy": db_stats.get("recommendation_accuracy", 0.0),
    }
    agent = _get_learning_agent()
    if agent is not None:
        try:
            agent_stats = agent.get_recommendation_stats()
            # Only override with agent data when it has real values
            if agent_stats.get("recommendation_accuracy", 0) > 0:
                learning_stats.update(agent_stats)
        except Exception as exc:
            logger.debug("[API] Learning stats failed: %s", exc)

    # ML model stats for the analytics dashboard
    ml = _get_ml_model()
    ml_dashboard: dict[str, Any] = {
        "accuracy": 0.0,
        "samples_trained": 0,
        "last_updated": None,
        "is_active": False,
    }
    false_positives_prevented = 0
    if ml is not None:
        try:
            mst = ml.get_stats()
            ml_dashboard = {
                "accuracy":        mst.get("accuracy", 0.0),
                "samples_trained": mst.get("samples_seen", 0),
                "last_updated":    mst.get("last_updated"),
                "is_active":       mst.get("is_fitted", False),
            }
            fb = _db.get_feedback_summary()
            # Estimate false positives the model can now correctly flag
            false_positives_prevented = round(
                fb.get("false_alarm", 0) * mst.get("accuracy", 0.0)
            )
        except Exception as exc:
            logger.debug("[API] ML stats for analytics failed: %s", exc)

    return {
        **db_stats,
        "learning": learning_stats,
        "ml_stats": ml_dashboard,
        "ml_adjusted_alerts": ml_dashboard["samples_trained"],
        "false_positives_prevented": false_positives_prevented,
    }


@app.get("/ml-stats")
def ml_stats() -> dict[str, Any]:
    """Return CipherMLModel statistics including accuracy history."""
    ml = _get_ml_model()
    if ml is None:
        return {"error": "ML model unavailable (scikit-learn not installed)",
                "sklearn_available": False, "is_fitted": False,
                "samples_seen": 0, "accuracy": 0.0, "last_updated": None,
                "accuracy_history": []}
    stats = ml.get_stats()
    stats["accuracy_history"] = _db.get_ml_accuracy_history()
    return stats


@app.post("/ml-retrain")
async def ml_retrain() -> dict[str, Any]:
    """Manually trigger full retrain from all feedback data."""
    t0 = time.perf_counter()
    ml = _get_ml_model()
    if ml is None:
        raise HTTPException(status_code=503, detail="ML model unavailable")

    training_data = _db.get_training_data(min_samples=1)
    if not training_data:
        return {"samples_used": 0, "accuracy": 0.0,
                "duration_seconds": 0.0, "message": "No labelled feedback yet"}

    result = ml.train_initial(training_data)
    acc = result.get("accuracy", 0.0)
    _db.log_ml_accuracy(acc, ml._samples_seen)

    duration = round(time.perf_counter() - t0, 3)
    logger.info("[API] Manual retrain: samples=%d acc=%.3f in %.2fs",
                result.get("samples", 0), acc, duration)
    return {
        "samples_used":     result.get("samples", 0),
        "accuracy":         acc,
        "duration_seconds": duration,
    }


@app.get("/feedback-summary")
def feedback_summary() -> dict[str, Any]:
    """Return counts and rates of all Telegram feedback received."""
    return _db.get_feedback_summary()


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Video simulation — upload a file, process frames, poll for results
# ---------------------------------------------------------------------------

_jobs: dict[str, dict[str, Any]] = {}


def _process_video(video_id: str, file_path: str) -> None:
    """Background worker: sample up to 100 frames from a video and run perception."""
    import cv2

    _jobs[video_id]["status"] = "processing"

    perception = None
    try:
        from perception_layer import PerceptionLayer  # type: ignore[import]
        perception = PerceptionLayer()
        logger.info("[SIM] PerceptionLayer loaded for job %s", video_id[:8])
    except Exception as exc:
        logger.warning("[SIM] PerceptionLayer unavailable (%s) — returning placeholder results", exc)

    cap: Any = None
    try:
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            _jobs[video_id].update({"status": "error", "error": "Could not open video file"})
            return

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        fps   = cap.get(cv2.CAP_PROP_FPS) or 25
        max_frames = 100

        # Pick up to max_frames evenly-spaced frame indices
        if total <= max_frames:
            indices = list(range(total))
        else:
            step = total / max_frames
            indices = [int(i * step) for i in range(max_frames)]

        detections: list[dict[str, Any]] = []

        for proc_idx, frame_idx in enumerate(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            ts         = datetime.now(timezone.utc).isoformat()
            frame_secs = round(frame_idx / fps, 2)

            if perception is not None:
                try:
                    result = perception.process_frame(frame)
                    weapon = result.get("weapon", {}) or {}
                    emotion = result.get("emotion", {}) or {}
                    tone   = result.get("tone", {}) or {}
                    detections.append({
                        "frame_id":          frame_idx,
                        "timestamp":         ts,
                        "frame_time_seconds": frame_secs,
                        "is_danger":         bool(result.get("is_danger", False)),
                        "danger_reasons":    result.get("danger_reasons", []),
                        "weapon_detected":   weapon.get("label", "unarmed") not in ("unarmed", ""),
                        "weapon_confidence": round(float(weapon.get("confidence", 0)), 3),
                        "weapon_label":      weapon.get("label", "unarmed"),
                        "emotion":           emotion.get("label", "—"),
                        "tone":              tone.get("label", tone.get("tone", "—")),
                        "detections":        [result],
                    })
                except Exception as exc:
                    logger.debug("[SIM] Frame %d perception error: %s", frame_idx, exc)
                    detections.append({
                        "frame_id": frame_idx, "timestamp": ts,
                        "frame_time_seconds": frame_secs,
                        "is_danger": False, "danger_reasons": [],
                        "weapon_detected": False, "weapon_confidence": 0,
                        "weapon_label": "unarmed", "emotion": "—", "tone": "—",
                        "detections": [],
                    })
            else:
                # No perception layer — return safe placeholder so UI still shows data
                detections.append({
                    "frame_id": frame_idx, "timestamp": ts,
                    "frame_time_seconds": frame_secs,
                    "is_danger": False, "danger_reasons": [],
                    "weapon_detected": False, "weapon_confidence": 0.0,
                    "weapon_label": "unarmed", "emotion": "neutral", "tone": "calm",
                    "detections": [{"note": "perception-layer not available in this environment"}],
                })

            _jobs[video_id]["progress_pct"]    = int((proc_idx + 1) / len(indices) * 100)
            _jobs[video_id]["processed_frames"] = proc_idx + 1

        _jobs[video_id].update({
            "status":           "completed",
            "detections":       detections,
            "processed_frames": len(detections),
            "progress_pct":     100,
        })
        logger.info("[SIM] Job %s complete: %d frames", video_id[:8], len(detections))

    except Exception as exc:
        logger.error("[SIM] Job %s failed: %s", video_id[:8], exc)
        _jobs[video_id].update({"status": "error", "error": str(exc)})
    finally:
        if cap is not None:
            cap.release()
        try:
            Path(file_path).unlink(missing_ok=True)
            Path(file_path).parent.rmdir()
        except Exception:
            pass


@app.post("/upload-video")
async def upload_video(file: UploadFile) -> dict[str, Any]:
    """Accept a video file and start background frame analysis.

    Returns immediately with a ``video_id``; poll ``GET /results/{video_id}``
    to track progress and retrieve results.
    """
    video_id = str(uuid.uuid4())
    suffix   = Path(file.filename or "video.mp4").suffix or ".mp4"
    tmp_dir  = Path(tempfile.mkdtemp())
    tmp_path = tmp_dir / f"{video_id}{suffix}"

    content = await file.read()
    tmp_path.write_bytes(content)

    _jobs[video_id] = {
        "status":           "queued",
        "progress_pct":     0,
        "processed_frames": 0,
        "detections":       [],
        "error":            None,
    }

    loop = asyncio.get_running_loop()
    loop.run_in_executor(None, _process_video, video_id, str(tmp_path))

    logger.info("[SIM] Job %s queued for '%s' (%.2f MB)",
                video_id[:8], file.filename, len(content) / 1_048_576)
    return {"video_id": video_id, "id": video_id, "status": "queued"}


@app.get("/results/{video_id}")
def get_results(video_id: str) -> dict[str, Any]:
    """Return current status and results for a simulation job."""
    job = _jobs.get(video_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Simulation job {video_id!r} not found")
    return job


# Test-alert endpoint — manually fires the full detection → swarm → Telegram pipeline
# ---------------------------------------------------------------------------

@app.get("/test-alert")
async def test_alert() -> dict[str, Any]:
    """Trigger a fake weapon detection to test the full alert pipeline.

    Creates a synthetic high-confidence gun detection, runs it through the
    swarm reasoning agent, saves it to the database, and sends both Telegram
    alerts.  Does NOT require a camera.
    """
    from datetime import datetime, timezone

    fake_perception: dict[str, Any] = {
        "frame_id": 0,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "is_danger": True,
        "danger_reasons": ["weapon:gun:0.95:confirmed", "weapon_civilian_combo:gun:0.95"],
        "detections": [{
            "weapon": {"label": "gun", "confidence": 0.95, "bbox": [100, 100, 300, 280], "confirmed": True},
            "emotion": {"label": "angry", "confidence": 0.85, "face_count": 1},
            "tone": {"label": "aggressive", "confidence": 0.75, "speech_present": True, "acoustic_events": ["high_energy_speech"]},
            "uniform": {"present": False, "confidence": 0.92, "label": "civilian"},
            "is_danger": True,
            "danger_reasons": ["weapon:gun:0.95:confirmed", "weapon_civilian_combo:gun:0.95"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }],
        "recommended_action": "DISPATCH_OFFICERS",
    }
    perception_data = fake_perception["detections"][0]

    # ── Lazy imports (avoid startup cost when not used) ───────────────────
    from alert_manager import AlertManager
    alert_mgr = AlertManager()

    # ── Alert 1 ──────────────────────────────────────────────────────────
    logger.info("[TEST-ALERT] Sending alert 1…")
    await alert_mgr.send_alert_1(perception_data, frame=None)

    # ── Swarm reasoning ──────────────────────────────────────────────────
    reasoning_result: dict[str, Any] = {}
    reasoning_obj: Any = None
    try:
        from swarm_reasoning_agent import SwarmReasoningAgent
        swarm = SwarmReasoningAgent()
        logger.info("[TEST-ALERT] Running swarm reasoning…")
        swarm_result = swarm.analyse(perception_data, learning_context="")
        reasoning_result = swarm_result.to_dict()
        reasoning_obj = swarm_result
        logger.info(
            "[TEST-ALERT] Swarm done: threat=%s action=%s conf=%.2f",
            swarm_result.overall_threat_level,
            swarm_result.final_action,
            swarm_result.confidence,
        )
    except Exception as exc:
        logger.warning("[TEST-ALERT] Swarm unavailable: %s", exc)

    # ── Save to DB ───────────────────────────────────────────────────────
    incident_id = _db.create_incident(
        perception_result={**fake_perception, "summary": reasoning_result.get("summary", "Test alert")},
        reasoning_result=reasoning_result,
        agent_reports=reasoning_result.get("agent_reports"),
    )
    logger.info("[TEST-ALERT] Incident saved: %s", incident_id)

    # ── Alert 2 ──────────────────────────────────────────────────────────
    logger.info("[TEST-ALERT] Sending alert 2…")
    await alert_mgr.send_alert_2(reasoning_obj or incident_id)

    return {
        "status": "ok",
        "incident_id": incident_id,
        "swarm_ran": reasoning_obj is not None,
        "threat_level": reasoning_result.get("overall_threat_level", "N/A"),
        "action": reasoning_result.get("course_of_action", "N/A"),
        "telegram_enabled": alert_mgr._enabled,
    }


# ---------------------------------------------------------------------------
# WebSocket — live dashboard stream
# ---------------------------------------------------------------------------

class _WSManager:
    """Tracks active WebSocket clients and broadcasts messages to all of them."""

    def __init__(self) -> None:
        self._clients: set[WebSocket] = set()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._clients.add(ws)
        logger.info("[WS] Client connected — total=%d", len(self._clients))

    def disconnect(self, ws: WebSocket) -> None:
        self._clients.discard(ws)
        logger.info("[WS] Client disconnected — total=%d", len(self._clients))

    async def broadcast(self, payload: dict) -> None:
        import json
        data = json.dumps(payload, default=str)
        dead: list[WebSocket] = []
        for ws in list(self._clients):
            try:
                await ws.send_text(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._clients.discard(ws)


_ws_manager = _WSManager()


def _utc_now_z() -> str:
    """Return current UTC time as ISO-8601 with 'Z' suffix (Zod-compatible)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _build_sentinel_message() -> dict[str, Any]:
    """Build a SentinelMessage payload from live DB state."""
    now = _utc_now_z()

    # Pull recent incidents for alert list
    recent = _db.get_recent_incidents(limit=20)
    analytics_data = _db.get_analytics()

    # Derive system mode from most recent unresolved incidents
    mode = "cloud"
    if recent:
        latest = recent[0]
        action = (latest.get("recommended_action") or "").upper()
        if "LOCKDOWN" in action or "DISPATCH" in action or "EMERGENCY" in action:
            mode = "incident"
        elif latest.get("officer_action") is None:
            mode = "cloud"

    # Convert DB incidents → Alert objects matching the frontend schema
    alerts: list[dict] = []
    for inc in recent[:10]:
        action = inc.get("recommended_action") or "MONITOR_ONLY"
        level = "clear"
        if "LOCKDOWN" in action or "DISPATCH" in action or "EMERGENCY" in action:
            level = "critical"
        elif "SURVEILLANCE" in action or "WARNING" in action:
            level = "high"
        elif "REVIEW" in action:
            level = "medium"

        # Normalise timestamp to Z suffix so Zod datetime() validation passes
        raw_ts = inc.get("created_at", now)
        if raw_ts and raw_ts.endswith("+00:00"):
            raw_ts = raw_ts[:-6] + "Z"
        elif raw_ts and not raw_ts.endswith("Z"):
            raw_ts = now

        alerts.append({
            "alertId": inc.get("id", str(uuid.uuid4())),
            "type": action,
            "detail": inc.get("perception_summary") or action,
            "level": level,
            "timestamp": raw_ts,
            "acknowledged": inc.get("officer_action") is not None,
            "isDegraded": False,
        })

    # Derive threat level from most severe active alert
    active_levels = [a["level"] for a in alerts if not a["acknowledged"]]
    overall_threat = "clear"
    for lvl in ("critical", "high", "medium", "low"):
        if lvl in active_levels:
            overall_threat = lvl
            break

    total = analytics_data.get("total_incidents", 0)
    fp_rate = analytics_data.get("false_positive_rate", 0.0)
    confidence = max(0.0, min(1.0, 1.0 - fp_rate))

    decision: dict[str, Any] = {
        "action": "dispatch" if overall_threat == "critical" else
                  "escalate" if overall_threat == "high" else
                  "monitor" if overall_threat in ("medium", "low") else "standby",
        "threatLevel": overall_threat,
        "confidence": round(confidence, 3),
        "rationale": f"{total} total incidents · {len(active_levels)} unacknowledged",
        "isDegraded": False,
        "generatedAt": now,
    }

    scenarios: list[dict] = [
        {"rank": 1, "name": "Armed Threat", "probability": 0.72, "isAvailable": True},
        {"rank": 2, "name": "Disturbance",  "probability": 0.18, "isAvailable": True},
        {"rank": 3, "name": "False Alarm",  "probability": 0.10, "isAvailable": True},
    ]

    zones: list[dict] = [
        {
            "zoneId": "zone-main",
            "zoneName": "Main Entrance",
            "floor": 1,
            "threatLevel": overall_threat,
            "activeAlerts": len(active_levels),
            "lastUpdated": now,
        },
    ]

    return {
        "messageId": str(uuid.uuid4()),
        "timestamp": now,
        "mode": mode,
        "decision": decision,
        "scenarios": scenarios,
        "alerts": alerts,
        "zones": zones,
    }


_DEMO_DIR = _ROOT / "demo"
_DEMO_DIR.mkdir(exist_ok=True)
_DEMO_VIDEO = _DEMO_DIR / "demo_video.mp4"

# Tracks the path of the most recently uploaded feed file (preserves extension).
_last_uploaded_feed: Path | None = None


@app.post("/feed/upload")
async def feed_upload(file: UploadFile) -> dict[str, Any]:
    """Accept a video file, save it, and immediately start playback.

    Single-call upload+start: the frontend only needs to POST here.
    The video starts playing as soon as this request returns.
    Returns {"status": "playing", "filename": ...}.
    """
    global _last_uploaded_feed

    suffix = Path(file.filename or "feed.mp4").suffix.lower() or ".mp4"
    if suffix not in {".mp4", ".avi", ".mov", ".mkv"}:
        raise HTTPException(status_code=400, detail="Unsupported video format. Accepted: mp4, avi, mov, mkv")

    # Save with canonical name, preserving the original extension
    dest = _DEMO_DIR / f"uploaded_feed{suffix}"
    with dest.open("wb") as fh:
        while chunk := await file.read(1 << 20):  # 1 MB chunks
            fh.write(chunk)

    _last_uploaded_feed = dest
    size_mb = dest.stat().st_size / 1_048_576
    logger.info("[API] Feed saved: %s (%.2f MB) — starting playback", dest.name, size_mb)

    # Immediately switch the video processor to this source (auto-start)
    try:
        import video_processor as _vp
        _vp.switch_source(str(dest))
        logger.info("[API] feed_upload: switched source to %s", dest)
    except Exception as exc:
        logger.warning("[API] feed_upload: switch_source failed: %s", exc)

    return {"status": "playing", "filename": dest.name, "path": str(dest)}


@app.post("/feed/start-video")
def feed_start_video(source: str | None = None) -> dict[str, Any]:
    """Switch the live feed to a video file source.

    Priority: explicit ``source`` param → last uploaded file → demo video.
    """
    if source:
        video_path = Path(source)
    elif _last_uploaded_feed is not None and _last_uploaded_feed.exists():
        video_path = _last_uploaded_feed
    elif _DEMO_VIDEO.exists():
        video_path = _DEMO_VIDEO
    else:
        raise HTTPException(
            status_code=404,
            detail="No video file available. Upload one via POST /feed/upload",
        )

    try:
        import video_processor as _vp
        _vp.switch_source(str(video_path))
        logger.info("[API] feed_start_video: switched to %s", video_path)
    except Exception as exc:
        logger.warning("[API] feed_start_video: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    return {"status": "started", "source": str(video_path)}


@app.post("/feed/stop-video")
def feed_stop_video() -> dict[str, Any]:
    """Stop the video feed and return to standby."""
    try:
        import video_processor as _vp
        _vp.switch_source(None)
        logger.info("[API] feed_stop_video: feed stopped")
    except Exception as exc:
        logger.warning("[API] feed_stop_video: %s", exc)
    return {"status": "stopped"}


@app.get("/feed/demo-available")
def feed_demo_available() -> dict[str, Any]:
    """Check whether demo/demo_video.mp4 exists for the 'Use Demo Video' button."""
    return {"available": _DEMO_VIDEO.exists(), "path": str(_DEMO_VIDEO)}


@app.get("/stream/start")
def stream_start() -> dict[str, Any]:
    """Enable camera capture — called when the user activates the live feed."""
    try:
        from video_processor import enable_streaming
        enable_streaming()
    except Exception as exc:
        logger.warning("[API] stream_start: %s", exc)
    return {"streaming": True}


@app.get("/stream/stop")
def stream_stop() -> dict[str, Any]:
    """Disable camera capture and clear the frame buffer."""
    try:
        from video_processor import disable_streaming
        disable_streaming()
    except Exception as exc:
        logger.warning("[API] stream_stop: %s", exc)
    return {"streaming": False}


def _inactive_feed_jpeg() -> bytes:
    """Return a dark grey 'FEED INACTIVE' placeholder JPEG."""
    import numpy as np
    import cv2 as _cv
    img = np.full((60, 80, 3), 20, dtype=np.uint8)
    _cv.putText(img, "FEED INACTIVE", (4, 36), _cv.FONT_HERSHEY_SIMPLEX, 0.38, (80, 80, 80), 1)
    _, buf = _cv.imencode(".jpg", img)
    return buf.tobytes()


@app.get("/stream/frame")
def stream_frame() -> Response:
    """Return the latest annotated camera frame as a JPEG image.

    Returns a dark 'FEED INACTIVE' placeholder when streaming is disabled
    or no frame has been captured yet — zero camera data is ever sent while
    the feed is in STANDBY.
    """
    try:
        import video_processor as _vp
        streaming = _vp._streaming_enabled
        jpeg = _vp.get_latest_frame_jpeg() if streaming else None
    except Exception:
        streaming = False
        jpeg = None

    if not streaming or jpeg is None:
        jpeg = _inactive_feed_jpeg()

    return Response(content=jpeg, media_type="image/jpeg", headers={
        "Cache-Control": "no-store",
        "Access-Control-Allow-Origin": "*",
    })


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    """Live dashboard stream — pushes SentinelMessage every 3 seconds."""
    await _ws_manager.connect(ws)
    try:
        # Send an immediate first message so the UI shows connected right away
        await ws.send_text(__import__("json").dumps(_build_sentinel_message(), default=str))

        while True:
            await asyncio.sleep(1)
            payload = _build_sentinel_message()
            await ws.send_text(__import__("json").dumps(payload, default=str))
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.warning("[WS] Unexpected error: %s", exc)
    finally:
        _ws_manager.disconnect(ws)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    port = int(os.getenv("API_PORT", "8000"))
    logger.info("[API] Starting officer response API on port %d", port)
    uvicorn.run(app, host="0.0.0.0", port=port)
