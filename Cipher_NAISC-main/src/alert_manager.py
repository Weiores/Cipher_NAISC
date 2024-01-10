"""
Telegram alert manager for Cipher_NAISC.

Sends two-stage alerts:
  Alert 1 – initial danger detection with annotated frame image.
  Alert 2 – AI reasoning summary + recommended course of action.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Allow importing swarm formatter when alert_manager is run standalone
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "reasoning-layer") not in sys.path:
    sys.path.insert(0, str(_ROOT / "reasoning-layer"))

import cv2
import numpy as np

logger = logging.getLogger(__name__)

try:
    from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup
    from telegram.error import TelegramError
    _TELEGRAM_AVAILABLE = True
except ImportError:
    _TELEGRAM_AVAILABLE = False
    logger.warning("[ALERT] python-telegram-bot not installed; alerts will be logged only")

_alert_ml_model: Any = None


def _get_alert_ml_model() -> Any:
    global _alert_ml_model
    if _alert_ml_model is None:
        try:
            _learning = _ROOT / "learning-layer"
            if str(_learning) not in sys.path:
                sys.path.insert(0, str(_learning))
            from ml_model import CipherMLModel  # type: ignore[import]
            _alert_ml_model = CipherMLModel()
        except Exception as exc:
            logger.debug("[ALERT] ML model unavailable: %s", exc)
    return _alert_ml_model


def _load_env() -> None:
    """Load .env from project root if present."""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_env()

_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
_DASHBOARD_URL = os.getenv("DASHBOARD_URL", "http://localhost:5173")


def _split_message(text: str, limit: int = 4096) -> list[str]:
    """Split a long string on newlines to stay under Telegram's char limit."""
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for line in text.splitlines(keepends=True):
        if current_len + len(line) > limit:
            chunks.append("".join(current))
            current, current_len = [], 0
        current.append(line)
        current_len += len(line)
    if current:
        chunks.append("".join(current))
    return chunks


def _annotate_frame(frame: np.ndarray, perception_result: dict[str, Any]) -> np.ndarray:
    """Draw detection overlays on a copy of the frame."""
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    weapon = perception_result.get("weapon", {})
    bbox = weapon.get("bbox", [])
    if bbox and len(bbox) >= 4:
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
        label = f"{weapon.get('label','?')} {weapon.get('confidence',0):.2f}"
        cv2.putText(annotated, label, (x1, max(y1 - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    emotion_label = perception_result.get("emotion", {}).get("label", "?")
    tone_label = perception_result.get("tone", {}).get("label", "?")
    cv2.putText(annotated, f"Emotion: {emotion_label}", (8, h - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(annotated, f"Tone: {tone_label}", (8, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    return annotated


class AlertManager:
    """Sends Telegram alerts when a security incident is detected."""

    def __init__(self) -> None:
        self._bot: Any = None
        self._enabled = bool(_BOT_TOKEN and _CHAT_ID and _TELEGRAM_AVAILABLE)
        if self._enabled:
            self._bot = Bot(token=_BOT_TOKEN)
            logger.info("[ALERT] Telegram bot initialised (chat_id=%s)", _CHAT_ID)
        else:
            logger.warning("[ALERT] Telegram not configured – alerts will be console-only")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def send_alert_1(
        self,
        perception_result: dict[str, Any],
        frame: np.ndarray | None = None,
    ) -> None:
        """Send initial danger alert with detection summary and annotated frame.

        Args:
            perception_result: Output from :class:`PerceptionLayer.process_frame`.
            frame:             The raw BGR video frame (will be annotated before sending).
        """
        ts = perception_result.get("timestamp", datetime.now(timezone.utc).isoformat())
        weapon = perception_result.get("weapon", {})
        emotion = perception_result.get("emotion", {})
        tone = perception_result.get("tone", {})
        uniform = perception_result.get("uniform", {})

        reasons = perception_result.get("danger_reasons", [])
        reasons_text = " | ".join(reasons) if reasons else "threshold exceeded"

        # ML threat probability (best-effort — no error if unavailable)
        ml_line = ""
        try:
            ml = _get_alert_ml_model()
            if ml is not None:
                pred = ml.predict(perception_result)
                n = pred.get("based_on_samples", 0)
                if n > 0:
                    prob_pct = int(pred["is_threat_probability"] * 100)
                    ml_line = f"\n🤖 <b>ML confidence:</b> {prob_pct}% threat ({n} training samples)"
        except Exception:
            pass

        message = (
            "🚨 <b>SECURITY ALERT – Danger Detected</b>\n\n"
            f"🕒 <b>Time:</b> {ts}\n"
            f"⚠️ <b>Trigger:</b> {reasons_text}\n\n"
            f"🔫 <b>Weapon:</b> {weapon.get('label','—')} "
            f"(conf: {weapon.get('confidence',0):.0%})\n"
            f"😠 <b>Emotion:</b> {emotion.get('label','—')} "
            f"(conf: {emotion.get('confidence',0):.0%})\n"
            f"🔊 <b>Tone:</b> {tone.get('label','—')} "
            f"(conf: {tone.get('confidence',0):.0%})\n"
            f"👮 <b>Uniform:</b> {'Yes' if uniform.get('present') else 'No'}"
            f"{ml_line}\n\n"
            "ℹ️ Reasoning agent is processing this incident…"
        )

        logger.info("[ALERT] Sending Telegram alert 1 (weapon=%s conf=%.2f)…",
                    perception_result.get("weapon", {}).get("label", "?"),
                    perception_result.get("weapon", {}).get("confidence", 0))
        logger.info("[ALERT1] %s", message.replace("\n", " | "))

        if not self._enabled:
            logger.warning("[ALERT] Telegram not configured — alert 1 logged only")
            return

        try:
            if frame is not None:
                annotated = _annotate_frame(frame, perception_result)
                tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
                tmp.close()
                cv2.imwrite(tmp.name, annotated)
                try:
                    with open(tmp.name, "rb") as photo:
                        await self._bot.send_photo(
                            chat_id=_CHAT_ID,
                            photo=photo,
                            caption=message,
                            parse_mode="HTML",
                        )
                finally:
                    Path(tmp.name).unlink(missing_ok=True)
            else:
                await self._bot.send_message(
                    chat_id=_CHAT_ID,
                    text=message,
                    parse_mode="HTML",
                )
            logger.info("[ALERT] Telegram alert 1 sent successfully")
        except TelegramError as exc:
            logger.error("[ALERT] ERROR sending alert 1: %s", exc)
        except Exception as exc:
            logger.error("[ALERT] ERROR sending alert 1 (unexpected): %s", exc)

    def send_alert_2(self, result: Any) -> None:
        """Sync wrapper — creates a fresh event loop so a previously closed loop
        (from send_alert_1's asyncio.run()) cannot cause RuntimeError."""
        try:
            asyncio.run(self._send_alert_2_async(result))
        except Exception as exc:
            logger.error("[ALERT] Alert 2 failed: %s", exc)

    async def _send_alert_2_async(self, result: Any) -> None:
        """Send the AI analysis for a detected incident.

        Accepts either a SwarmReasoningResult (swarm mode) or a
        ReasoningResult (single-agent fallback) or a plain incident_id
        string when no reasoning data is available.
        """
        if hasattr(result, "agent_reports"):
            # Swarm path — full war-room breakdown
            try:
                from swarm_reasoning_agent import format_swarm_output
                text = format_swarm_output(result)
            except Exception as exc:
                logger.warning("[ALERT2] format_swarm_output failed: %s", exc)
                text = (
                    f"📋 Incident {result.incident_id[:8]}\n\n"
                    f"Summary: {result.incident_summary}\n\n"
                    f"Action: {result.final_action}\n"
                    f"Confidence: {int(result.confidence * 100)}%"
                )
            incident_id = result.incident_id
            action = result.final_action
            confidence = result.confidence

        elif hasattr(result, "summary"):
            # Legacy single-agent ReasoningResult
            action_emoji = {
                "DISPATCH_OFFICERS": "🚔",
                "INCREASE_SURVEILLANCE": "📷",
                "ISSUE_VERBAL_WARNING": "📢",
                "REVIEW_FOOTAGE": "🎥",
                "FALSE_ALARM": "✅",
            }.get((result.course_of_action or "").upper(), "🔔")
            text = (
                f"{action_emoji} INCIDENT ANALYSIS COMPLETE\n\n"
                f"Incident: {result.incident_id[:8]}\n\n"
                f"Summary: {result.summary}\n\n"
                f"Action: {result.course_of_action}\n"
                f"Confidence: {int(result.confidence * 100)}%"
            )
            incident_id = result.incident_id
            action = result.course_of_action
            confidence = result.confidence

        else:
            # Fallback: result is an incident_id string or unknown
            incident_id = str(result)
            action = "REVIEW_FOOTAGE"
            confidence = 0.0
            text = (
                f"📋 Incident {incident_id[:8]}\n\n"
                "Security threat detected. Manual review required.\n\n"
                "Action: REVIEW_FOOTAGE"
            )

        logger.info("[ALERT] Sending Telegram alert 2 (incident=%s action=%s conf=%.2f)…",
                    incident_id, action, confidence)
        logger.info(
            "[ALERT2] incident=%s action=%s conf=%.2f",
            incident_id, action, confidence,
        )

        if not self._enabled:
            logger.warning("[ALERT] Telegram not configured — alert 2 logged only")
            return

        try:
            for chunk in _split_message(text, limit=4096):
                await self._bot.send_message(
                    chat_id=_CHAT_ID,
                    text=chunk,
                    disable_web_page_preview=True,
                )
            logger.info("[ALERT] Telegram alert 2 sent successfully")
        except TelegramError as exc:
            logger.error("[ALERT] ERROR sending alert 2: %s", exc)
        except Exception as exc:
            logger.error("[ALERT] ERROR sending alert 2 (unexpected): %s", exc)

        await self.send_feedback_prompt(incident_id)

    async def send_feedback_prompt(self, incident_id: str) -> None:
        """Send inline keyboard so officers can rate the alert directly in Telegram."""
        if not self._enabled or not _TELEGRAM_AVAILABLE:
            return

        def _cb(ftype: str) -> str:
            return f"feedback:{incident_id}:{ftype}"

        keyboard = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("✅ Confirmed",   callback_data=_cb("confirmed")),
                InlineKeyboardButton("❌ False Alarm", callback_data=_cb("false_alarm")),
                InlineKeyboardButton("⚠️ Partial",    callback_data=_cb("partial")),
            ],
            [
                InlineKeyboardButton("👍 Good Rec", callback_data=_cb("good_rec")),
                InlineKeyboardButton("👎 Bad Rec",  callback_data=_cb("bad_rec")),
            ],
        ])
        try:
            await self._bot.send_message(
                chat_id=_CHAT_ID,
                text=f"📝 <b>Rate this alert</b> (ID: <code>{incident_id[:8]}</code>):",
                reply_markup=keyboard,
                parse_mode="HTML",
            )
        except Exception as exc:
            logger.warning("[ALERT] Could not send feedback prompt: %s", exc)


# ---------------------------------------------------------------------------
# Telegram feedback listener — run as an asyncio background task
# ---------------------------------------------------------------------------

async def _handle_callback(bot: Any, cq: Any, db: Any, learning_agent: Any) -> None:
    """Process one inline-keyboard callback from an officer."""
    data = getattr(cq, "data", "") or ""
    if not data.startswith("feedback:"):
        return

    parts = data.split(":", 2)
    if len(parts) != 3:
        return
    _, incident_id, ftype = parts

    threat_feedback = ftype if ftype in ("confirmed", "false_alarm", "partial") else None
    rec_feedback    = {"good_rec": "good", "bad_rec": "bad"}.get(ftype)

    if db is not None:
        try:
            db.record_telegram_feedback(incident_id, threat_feedback, rec_feedback)
            logger.info("[FEEDBACK] Recorded: incident=%s type=%s", incident_id, ftype)
        except Exception as exc:
            logger.error("[FEEDBACK] DB error: %s", exc)

    try:
        await bot.answer_callback_query(
            callback_query_id=cq.id,
            text="✅ Feedback recorded. Thank you.",
        )
        if cq.message:
            await cq.message.edit_reply_markup(reply_markup=None)
    except Exception as exc:
        logger.warning("[FEEDBACK] Could not answer callback: %s", exc)

    if learning_agent is not None:
        try:
            learning_agent.update_from_feedback(incident_id)
        except Exception as exc:
            logger.error("[FEEDBACK] Learning update error: %s", exc)


async def run_feedback_listener(db: Any, learning_agent: Any) -> None:
    """Long-running coroutine: polls Telegram and handles officer feedback callbacks.

    Start this as an asyncio background task once at API startup.
    """
    if not (_BOT_TOKEN and _CHAT_ID and _TELEGRAM_AVAILABLE):
        logger.warning("[FEEDBACK] Telegram not configured — feedback listener inactive")
        return

    bot = Bot(token=_BOT_TOKEN)
    offset = 0
    logger.info("[FEEDBACK] Telegram feedback listener started")

    while True:
        try:
            updates = await bot.get_updates(
                offset=offset,
                timeout=30,
                allowed_updates=["callback_query"],
            )
            for update in updates:
                offset = update.update_id + 1
                if update.callback_query:
                    await _handle_callback(bot, update.callback_query, db, learning_agent)
        except asyncio.CancelledError:
            logger.info("[FEEDBACK] Feedback listener stopped")
            raise
        except Exception as exc:
            logger.error("[FEEDBACK] Polling error: %s", exc)
            await asyncio.sleep(5)


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.DEBUG)

    dummy_perception: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "danger_reasons": ["weapon:gun:0.85"],
        "weapon": {"label": "gun", "confidence": 0.85, "bbox": [100, 100, 300, 280]},
        "emotion": {"label": "angry", "confidence": 0.75},
        "tone": {"label": "threat", "confidence": 0.70},
        "uniform": {"present": False},
    }

    async def _test() -> None:
        mgr = AlertManager()
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        await mgr.send_alert_1(dummy_perception, dummy_frame)
        await mgr.send_alert_2("INC-TEST01")  # minimal fallback path
        print("Test complete")

    asyncio.run(_test())
