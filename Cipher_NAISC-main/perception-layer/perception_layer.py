"""
Perception Layer orchestrator for Cipher_NAISC.

Aggregates weapon, emotion, tone, and uniform detectors into a single
unified :class:`PerceptionResult`.  Also applies the danger threshold
logic that decides whether an alert should be triggered.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from weapon_detector import WeaponDetector, WeaponDetectionResult
from emotion_detector import EmotionDetector, EmotionDetectionResult
from tone_detector import ToneDetector, ToneDetectionResult
from uniform_detector import UniformDetector, UniformDetectionResult

logger = logging.getLogger(__name__)

_DANGER_WEAPON_THRESHOLD  = float(os.getenv("DANGER_WEAPON_THRESHOLD",  "0.50"))
_DANGER_EMOTION_THRESHOLD = float(os.getenv("DANGER_EMOTION_THRESHOLD", "0.65"))

_DANGER_EMOTIONS = {"angry", "fearful", "distressed"}

# ---------------------------------------------------------------------------
# Lazy ML model — loads CipherMLModel from learning-layer/ on first use
# ---------------------------------------------------------------------------

_perception_ml_model: Any = None
_LEARNING_LAYER = str(Path(__file__).resolve().parent.parent / "learning-layer")


def _get_perception_ml_model() -> Any:
    global _perception_ml_model
    if _perception_ml_model is None:
        try:
            if _LEARNING_LAYER not in sys.path:
                sys.path.insert(0, _LEARNING_LAYER)
            from ml_model import CipherMLModel  # type: ignore[import]
            _perception_ml_model = CipherMLModel()
            logger.info(
                f"[PERCEPTION] CipherMLModel loaded: fitted={_perception_ml_model._is_fitted} samples={_perception_ml_model._samples_seen}"
            )
        except Exception as exc:
            logger.debug(f"[PERCEPTION] ML model unavailable: {exc}")
            _perception_ml_model = False  # sentinel so we don't retry
    return _perception_ml_model if _perception_ml_model is not False else None


@dataclass
class PerceptionResult:
    """Unified result from the full perception pipeline for one frame."""

    frame_id: int
    timestamp: str
    weapon: WeaponDetectionResult
    emotion: EmotionDetectionResult
    tone: ToneDetectionResult
    uniform: UniformDetectionResult
    is_danger: bool = False
    danger_reasons: list[str] = field(default_factory=list)

    @property
    def danger(self) -> bool:
        return self.is_danger

    def to_dict(self) -> dict[str, Any]:
        """Serialisable dict for API / database storage."""
        return {
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "is_danger": self.is_danger,
            "danger_reasons": self.danger_reasons,
            "weapon": {
                "label": self.weapon.label,
                "confidence": self.weapon.confidence,
                "bbox": self.weapon.bbox,
            },
            "emotion": {
                "label": self.emotion.label,
                "confidence": self.emotion.confidence,
                "face_count": self.emotion.face_count,
            },
            "tone": {
                "label": self.tone.tone,
                "confidence": self.tone.confidence,
                "speech_present": self.tone.speech_present,
                "acoustic_events": self.tone.acoustic_events,
            },
            "uniform": {
                "present": self.uniform.uniform_present,
                "confidence": self.uniform.confidence,
                "label": self.uniform.label,
            },
        }


class PerceptionLayer:
    """Orchestrates all sub-detectors and applies danger threshold logic.

    Typical usage::

        layer = PerceptionLayer()
        result = layer.process_frame(frame_bgr, frame_id=42)
        if result.is_danger:
            # trigger alert
    """

    def __init__(self, video_path: str | Path | None = None) -> None:
        """Initialise all detectors.

        Args:
            video_path: If provided, audio tone analysis is run on the full
                        video once at construction time rather than per-frame.
        """
        logger.info("[PERCEPTION] Initialising detectors…")
        self.weapon = WeaponDetector()
        self.emotion = EmotionDetector()
        # Pass video_source so ToneDetector knows whether to load Whisper
        _vs = str(video_path) if video_path else os.getenv("VIDEO_SOURCE", "0")
        self.tone_detector = ToneDetector(video_source=_vs)
        self.uniform = UniformDetector()

        # Pre-compute audio analysis for the whole video if path provided
        self._precomputed_tone: ToneDetectionResult | None = None
        if video_path:
            self._precomputed_tone = self.tone_detector.detect_from_file(video_path)
            logger.info(
                f"[PERCEPTION] Pre-computed tone: {self._precomputed_tone.tone} ({self._precomputed_tone.confidence:.2f})"
            )

    # ------------------------------------------------------------------

    def process_frame(
        self,
        frame: np.ndarray,
        frame_id: int = 0,
        video_path: str | Path | None = None,
    ) -> PerceptionResult:
        """Run all detectors on a single BGR frame and return a unified result.

        Args:
            frame:      OpenCV BGR frame (numpy ndarray).
            frame_id:   Sequential frame index.
            video_path: If tone was not pre-computed, supply the video path
                        here to trigger on-demand audio analysis.
        """
        ts = datetime.now(timezone.utc).isoformat()

        weapon_result = self.weapon.detect(frame)
        emotion_result = self.emotion.detect(frame)

        # Use pre-computed tone or run on-demand; pass emotion as proxy hint
        if self._precomputed_tone is not None:
            tone_result = self._precomputed_tone
        elif video_path:
            tone_result = self.tone_detector.detect_from_file(video_path)
        else:
            tone_result = self.tone_detector.detect(
                frame=frame, emotion_hint=emotion_result.label
            )

        uniform_result = self.uniform.detect(frame)

        logger.info(
            f"[PERCEPTION] Weapon confidence: {weapon_result.confidence:.3f} (label={weapon_result.label}, threshold={_DANGER_WEAPON_THRESHOLD:.2f})"
        )

        # ML-adjusted threshold
        ml_threat_prob = self._get_ml_threat_prob(
            weapon_result, emotion_result, tone_result, uniform_result
        )

        is_danger, danger_reasons = self._apply_danger_logic(
            weapon_result, emotion_result, tone_result, uniform_result,
            ml_threat_prob=ml_threat_prob,
        )

        _eff_threshold = float(os.getenv("DANGER_WEAPON_THRESHOLD", "0.50")) * (1.5 - ml_threat_prob)
        _eff_threshold = max(0.15, min(0.85, _eff_threshold))
        logger.info(
            f"[PERCEPTION] weapon_conf={weapon_result.confidence:.4f} threshold={_eff_threshold:.2f} confirmed={weapon_result.confirmed} danger={is_danger}"
        )

        logger.info(
            f"[PERCEPTION] Danger decision: {is_danger}, reason: {danger_reasons if danger_reasons else 'none'}"
        )

        result = PerceptionResult(
            frame_id=frame_id,
            timestamp=ts,
            weapon=weapon_result,
            emotion=emotion_result,
            tone=tone_result,
            uniform=uniform_result,
            is_danger=is_danger,
            danger_reasons=danger_reasons,
        )

        logger.info(
            f"[PERCEPTION] Frame {frame_id}: weapon={weapon_result.label}({weapon_result.confidence:.2f}) "
            f"emotion={emotion_result.label}({emotion_result.confidence:.2f}) "
            f"tone={tone_result.tone} uniform={uniform_result.uniform_present} danger={is_danger}"
        )

        return result

    # ------------------------------------------------------------------

    def _get_ml_threat_prob(
        self,
        weapon: WeaponDetectionResult,
        emotion: EmotionDetectionResult,
        tone: ToneDetectionResult,
        uniform: UniformDetectionResult,
    ) -> float:
        """Return ML threat probability (0-1). Falls back to 0.5 on any failure."""
        try:
            ml = _get_perception_ml_model()
            if ml is None or not ml._is_fitted:
                return 0.5
            pred = ml.predict({
                "weapon":  {"confidence": weapon.confidence},
                "emotion": {"label": emotion.label},
                "tone":    {"label": tone.tone, "tone": tone.tone},
                "uniform": {"present": uniform.uniform_present},
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            return float(pred.get("is_threat_probability", 0.5))
        except Exception as exc:
            logger.debug(f"[PERCEPTION] ML predict error: {exc}")
            return 0.5

    def _apply_danger_logic(
        self,
        weapon: WeaponDetectionResult,
        emotion: EmotionDetectionResult,
        tone: ToneDetectionResult,  # noqa: ARG002 — kept for signature compat; not used as trigger
        uniform: UniformDetectionResult,
        ml_threat_prob: float = 0.5,
    ) -> tuple[bool, list[str]]:
        """Determine whether the frame represents a dangerous situation.

        Rules (tone is excluded — unreliable on webcam):
          Rule 1: weapon_confidence >= threshold AND confirmed in 3 consecutive frames
          Rule 2: weapon_confidence >= threshold AND subject is civilian (no uniform)
          Rule 3: dangerous emotion (angry/fearful) with confidence >= emotion_threshold
                  AND weapon_confidence > 0.30 (not emotion alone)

        The weapon threshold is dynamically adjusted by the ML model's threat
        probability: high ML confidence lowers the threshold (easier to trigger),
        low ML confidence raises it (harder to trigger).
        """
        reasons: list[str] = []

        # Labels that must never trigger a weapon alert regardless of confidence.
        # Common household objects frequently misidentified by COCO models.
        EXCLUDED_LABELS = {
            "bottle", "cup", "vase", "bowl",
            "wine glass", "fork", "spoon",
        }
        weapon_label = weapon.label
        if weapon_label in EXCLUDED_LABELS:
            logger.info(f"[PERCEPTION] Weapon label '{weapon_label}' is in EXCLUDED_LABELS — treating as unarmed")
            weapon_label = "unarmed"

        # Read base thresholds at call time so runtime .env overrides take effect
        base_threshold    = float(os.getenv("DANGER_WEAPON_THRESHOLD",  "0.50"))
        emotion_threshold = float(os.getenv("DANGER_EMOTION_THRESHOLD", "0.65"))

        # ML-adjusted weapon threshold
        # Only apply when ML has enough training data (prob >= 0.3).
        # At 0% probability the model is untrained and would raise the
        # threshold from 0.50 → 0.75, blocking real detections.
        if ml_threat_prob < 0.3:
            weapon_threshold = base_threshold  # ignore ML — not enough training data
        else:
            weapon_threshold = base_threshold * (1.5 - ml_threat_prob)
        weapon_threshold = max(0.15, min(0.85, weapon_threshold))
        logger.info(
            f"[PERCEPTION] ML adjusted threshold: {weapon_threshold:.2f} (base={base_threshold:.2f}, ml_prob={ml_threat_prob:.0%})"
        )

        weapon_above_threshold = (
            weapon_label not in {"unarmed", "unknown", "unknown_object"}
            and weapon.confidence >= weapon_threshold
        )

        # Rule 1 — confirmed weapon: must exceed threshold for N consecutive frames
        if weapon_above_threshold and weapon.confirmed:
            reason = f"weapon:{weapon_label}:{weapon.confidence:.2f}:confirmed"
            reasons.append(reason)
            logger.info(f"[PERCEPTION] Rule 1 triggered: {reason}")

        # Rule 2 — civilian with weapon (high risk even on single frame)
        if weapon_above_threshold and not uniform.uniform_present:
            reason = f"weapon_civilian_combo:{weapon_label}:{weapon.confidence:.2f}"
            if reason not in reasons:
                reasons.append(reason)
                logger.info(f"[PERCEPTION] Rule 2 triggered: {reason}")

        # Rule 3 — emotional escalation backed by some weapon presence
        emotion_dangerous = (
            emotion.label in _DANGER_EMOTIONS
            and emotion.confidence >= emotion_threshold
        )
        if emotion_dangerous and weapon.confidence > 0.30:
            reason = f"emotion:{emotion.label}:{emotion.confidence:.2f}+weapon_present"
            reasons.append(reason)
            logger.info(f"[PERCEPTION] Rule 3 triggered: {reason}")

        logger.info(
            f"[PERCEPTION] Danger check: weapon_conf={weapon.confidence:.3f} threshold={weapon_threshold:.2f} "
            f"confirmed={weapon.confirmed} civilian={not uniform.uniform_present} emotion={emotion.label}({emotion.confidence:.2f})"
        )

        return bool(reasons), reasons


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import cv2
    logging.basicConfig(level=logging.DEBUG)

    layer = PerceptionLayer()

    if len(sys.argv) > 1:
        source = sys.argv[1]
        cap = cv2.VideoCapture(source)
        ok, frame = cap.read()
        cap.release()
        if ok:
            result = layer.process_frame(frame, frame_id=0, video_path=source)
            import json
            print(json.dumps(result.to_dict(), indent=2))
        else:
            print("Could not read frame from source")
    else:
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        result = layer.process_frame(dummy, frame_id=0)
        import json
        print(json.dumps(result.to_dict(), indent=2))
