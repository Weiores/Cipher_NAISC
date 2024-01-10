"""
Cipher_NAISC – Main system orchestrator.

Ties together the video processor, perception layer, reasoning agent,
alert manager, incident database, and FastAPI server.

Flow:
  1. Start video processor (thread / generator loop)
  2. For each frame batch → run perception layer
  3. If danger detected:
       a. Send Telegram Alert #1 (initial detection)
       b. Query learning agent for similar past incidents
       c. Run reasoning agent (summarise + course of action via Groq)
       d. Save incident to database
       e. Send Telegram Alert #2 (AI summary + recommended action)
  4. If no danger → log "clear", continue loop
  5. Run FastAPI on port 8000 in a separate thread
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import threading
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path setup – allow imports from sibling packages
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "perception-layer"))
sys.path.insert(0, str(_ROOT / "reasoning-layer"))
sys.path.insert(0, str(_ROOT / "learning-layer"))

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

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("cipher_naisc.main")

# ---------------------------------------------------------------------------
# Component imports (all optional – system degrades gracefully)
# ---------------------------------------------------------------------------

from incident_database import IncidentDatabase
from alert_manager import AlertManager
from video_processor import VideoProcessor

try:
    from reasoning_agent import ReasoningAgent
    _reasoning_available = True
except ImportError:
    _reasoning_available = False
    logger.warning("[MAIN] ReasoningAgent unavailable")

try:
    from swarm_reasoning_agent import SwarmReasoningAgent
    _swarm_available = True
except ImportError:
    _swarm_available = False
    logger.warning("[MAIN] SwarmReasoningAgent unavailable")

try:
    from learning_agent import LearningAgent
    _learning_available = True
except ImportError:
    _learning_available = False
    logger.warning("[MAIN] LearningAgent unavailable")

# ---------------------------------------------------------------------------
# Global singletons
# ---------------------------------------------------------------------------

_db = IncidentDatabase()
_alert_mgr = AlertManager()
_reasoning: Any = ReasoningAgent() if _reasoning_available else None
_learning: Any = LearningAgent(db=_db) if _learning_available else None
_swarm: Any = SwarmReasoningAgent() if _swarm_available else None

_VIDEO_SOURCE = os.getenv("VIDEO_SOURCE", "0")
_API_PORT = int(os.getenv("API_PORT", "8000"))

# Detect demo/file mode — a non-digit, non-RTSP path means a pre-recorded video
_IS_FILE_SOURCE = (
    _VIDEO_SOURCE.strip() != ""
    and not _VIDEO_SOURCE.strip().isdigit()
    and not _VIDEO_SOURCE.strip().lower().startswith("rtsp")
)


# ---------------------------------------------------------------------------
# Danger handler
# ---------------------------------------------------------------------------

def _handle_danger(frame_result: dict[str, Any], frame_bgr: Any) -> None:
    """Process a danger event: alert → reason → store → alert again."""
    logger.warning(
        "[MAIN] *** DANGER DETECTED *** frame=%d reasons=%s",
        frame_result["frame_id"],
        frame_result.get("danger_reasons"),
    )

    perception_data = (frame_result.get("detections") or [{}])[0]

    # ── Alert #1: initial detection ──────────────────────────────────────
    logger.info("[MAIN] Step 1/4 — sending Alert 1 (Telegram initial notification)")
    asyncio.run(_alert_mgr.send_alert_1(perception_data, frame_bgr))

    # ── Reasoning ────────────────────────────────────────────────────────
    logger.info("[MAIN] Step 2/4 — running reasoning agent(s)")
    reasoning_result: dict[str, Any] = {}
    _reasoning_obj: Any = None
    if _reasoning or _swarm:
        incident_history: list[dict[str, Any]] = []
        learning_context: str = ""
        if _learning:
            try:
                incident_history = _learning.get_similar_incidents(
                    perception_data, top_k=5
                )
                learning_context = _learning.generate_learning_context(incident_history)
                logger.info("[MAIN] Learning context: %d similar incidents found", len(incident_history))
            except Exception as exc:
                logger.warning("[MAIN] Learning query failed: %s", exc)

        if _swarm and os.getenv("SWARM_ENABLED", "true").lower() == "true":
            logger.info("[MAIN] Running swarm reasoning (5 agents)…")
            try:
                swarm_result = _swarm.analyse(perception_data, learning_context)
                reasoning_result = swarm_result.to_dict()
                _reasoning_obj = swarm_result
                logger.info(
                    "[MAIN] Swarm complete: threat=%s action=%s conf=%.2f",
                    swarm_result.overall_threat_level,
                    swarm_result.final_action,
                    swarm_result.confidence,
                )
            except Exception as exc:
                logger.warning("[MAIN] Swarm failed, falling back to single agent: %s", exc)
                if _reasoning:
                    try:
                        result = _reasoning.process(perception_data, incident_history)
                        reasoning_result = result.to_dict()
                        _reasoning_obj = result
                        logger.info("[MAIN] Single-agent reasoning complete")
                    except Exception as exc2:
                        logger.error("[MAIN] Reasoning fallback also failed: %s", exc2)
        elif _reasoning:
            logger.info("[MAIN] Running single-agent reasoning…")
            try:
                result = _reasoning.process(perception_data, incident_history)
                reasoning_result = result.to_dict()
                _reasoning_obj = result
                logger.info("[MAIN] Single-agent reasoning complete")
            except Exception as exc:
                logger.error("[MAIN] Reasoning failed: %s", exc)
    else:
        logger.warning("[MAIN] No reasoning agent available — skipping AI analysis")

    # ── Store incident ────────────────────────────────────────────────────
    logger.info("[MAIN] Step 3/4 — saving incident to database")
    try:
        incident_id = _db.create_incident(
            perception_result={
                **frame_result,
                "summary": reasoning_result.get("summary", ""),
            },
            reasoning_result=reasoning_result,
            agent_reports=reasoning_result.get("agent_reports"),
        )
        logger.info("[MAIN] Incident saved: id=%s", incident_id)
    except Exception as exc:
        logger.error("[MAIN] DB write failed: %s", exc)
        incident_id = f"NODB-{frame_result['frame_id']}"

    # ── Alert #2: AI summary + recommended action ─────────────────────────
    logger.info("[MAIN] Step 4/4 — sending Alert 2 (AI summary)")
    _alert_mgr.send_alert_2(_reasoning_obj or incident_id)
    logger.info("[MAIN] Danger handling complete for frame %d", frame_result["frame_id"])


# ---------------------------------------------------------------------------
# Video processing thread
# ---------------------------------------------------------------------------

def _video_loop() -> None:
    logger.info("[MAIN] Video processor starting with source=%s", _VIDEO_SOURCE)

    frames_with_bgr: dict[int, Any] = {}

    def _on_frame(result: dict[str, Any]) -> None:
        fid = result["frame_id"]
        logger.info("[MAIN] Frame %d processed — danger=%s", fid, result["is_danger"])

    def _on_danger(result: dict[str, Any], frame_bgr: Any) -> None:
        fid = result["frame_id"]
        frames_with_bgr[fid] = frame_bgr

    processor = VideoProcessor(
        source=_VIDEO_SOURCE,
        fps_sample=float(os.getenv("SAMPLE_FPS", "3.0")),
        on_frame_result=_on_frame,
        on_danger=_on_danger,
    )

    for result in processor.stream():
        fid = result["frame_id"]
        if result["is_danger"]:
            frame = frames_with_bgr.pop(fid, None)
            threading.Thread(
                target=_handle_danger,
                args=(result, frame),
                daemon=True,
            ).start()
        else:
            logger.info("[MAIN] Frame %d: CLEAR", fid)
        frames_with_bgr.pop(fid, None)


# ---------------------------------------------------------------------------
# FastAPI thread
# ---------------------------------------------------------------------------

def _api_thread() -> None:
    import uvicorn
    from officer_response_api import app

    logger.info("[MAIN] FastAPI starting on port %d", _API_PORT)
    uvicorn.run(app, host="0.0.0.0", port=_API_PORT, log_level="warning")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    logger.info("=" * 60)
    logger.info("  Cipher_NAISC Security AI System – Starting")
    logger.info("=" * 60)
    logger.info("  Video source : %s", _VIDEO_SOURCE)
    logger.info("  Mode         : %s", "DEMO (file loop)" if _IS_FILE_SOURCE else "LIVE (webcam gate)")
    logger.info("  API port     : %d", _API_PORT)
    logger.info("  Groq model   : %s", os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"))
    logger.info("  Telegram     : %s", "enabled" if os.getenv("TELEGRAM_BOT_TOKEN") else "disabled")
    logger.info("=" * 60)

    # For file/demo sources, open the video immediately — no user consent gate needed.
    # For webcam sources, the gate stays closed until the user clicks "Activate" in the UI.
    if _IS_FILE_SOURCE:
        import video_processor as _vp
        logger.info("[MAIN] Demo mode — auto-enabling streaming for file source")
        _vp.enable_streaming(_VIDEO_SOURCE)

    # Start FastAPI in background thread
    api_t = threading.Thread(target=_api_thread, daemon=True)
    api_t.start()

    # Start video processor in background thread
    video_t = threading.Thread(target=_video_loop, daemon=True)
    video_t.start()

    logger.info("[MAIN] All components started. Press Ctrl+C to stop.")
    try:
        video_t.join()
    except KeyboardInterrupt:
        logger.info("[MAIN] Shutdown requested – stopping.")


if __name__ == "__main__":
    main()
