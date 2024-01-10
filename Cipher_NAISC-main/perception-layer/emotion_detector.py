"""
Emotion detector for Cipher_NAISC — EmotiEffLib ONNX backend.

Pipeline:
  1. OpenCV Haar cascade → face bounding boxes
  2. EmotiEffLibRecognizer (enet_b0_8_best_afew, ONNX) → per-face emotion
  3. Cascade brightness heuristic if EmotiEffLib unavailable
  4. Neutral placeholder as final fallback
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EmotionDetectionResult:
    """Single-frame emotion detection result."""

    label: str           # angry | fearful | distressed | neutral | unknown
    confidence: float
    face_count: int = 0
    bbox: list[float] = field(default_factory=list)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# EmotiEffLib class labels → system labels
_LABEL_MAP: dict[str, str] = {
    "Anger":    "angry",
    "Contempt": "angry",
    "Disgust":  "angry",
    "Fear":     "fearful",
    "Happiness":"neutral",
    "Neutral":  "neutral",
    "Sadness":  "neutral",
    "Surprise": "neutral",
}


class EmotionDetector:
    """Detect face emotions using EmotiEffLib (ONNX) with cascade fallback."""

    def __init__(self) -> None:
        self._recognizer = None
        self._img_size = 224

        # ── Primary: EmotiEffLib ONNX ─────────────────────────────────────
        try:
            from emotiefflib.facial_analysis import EmotiEffLibRecognizer
            self._recognizer = EmotiEffLibRecognizer(
                engine="onnx",
                model_name="enet_b0_8_best_afew",
            )
            self._img_size = int(self._recognizer.img_size)
            logger.info(
                "[EMOTION] EmotiEffLib ONNX ready (enet_b0_8_best_afew, img=%d)",
                self._img_size,
            )
        except Exception as exc:
            logger.warning("[EMOTION] EmotiEffLib unavailable: %s", exc)

        # ── Face detector (used by both backends) ─────────────────────────
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        try:
            self._cascade = cv2.CascadeClassifier(cascade_path)
            if self._cascade.empty():
                self._cascade = None
        except Exception:
            self._cascade = None

        if self._recognizer:
            logger.info("[EMOTION] Backend: EmotiEffLib ONNX")
        elif self._cascade:
            logger.info("[EMOTION] Backend: OpenCV cascade heuristic")
        else:
            logger.warning("[EMOTION] No backend — returning neutral placeholder")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> EmotionDetectionResult:
        """Run emotion detection on a single BGR frame."""
        faces = self._detect_faces(frame)

        if not faces:
            return EmotionDetectionResult(label="neutral", confidence=0.5, face_count=0)

        if self._recognizer is not None:
            return self._run_emotieff(frame, faces)

        if self._cascade is not None:
            return self._cascade_heuristic(frame, faces)

        return EmotionDetectionResult(label="neutral", confidence=0.5, face_count=0)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _detect_faces(
        self, frame: np.ndarray
    ) -> list[tuple[int, int, int, int]]:
        """Return list of (x, y, w, h) face boxes detected by the Haar cascade."""
        if self._cascade is None:
            return []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected = self._cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40)
        )
        if len(detected) == 0:
            return []
        return [(int(x), int(y), int(w), int(h)) for x, y, w, h in detected]

    def _run_emotieff(
        self,
        frame: np.ndarray,
        faces: list[tuple[int, int, int, int]],
    ) -> EmotionDetectionResult:
        """Pass face crops through EmotiEffLibRecognizer; return best result."""
        h_f, w_f = frame.shape[:2]
        best_label, best_conf, best_bbox = "neutral", 0.0, []

        for x, y, w, h in faces:
            # Expand crop slightly for better context
            pad = int(0.15 * max(w, h))
            x1, y1 = max(0, x - pad), max(0, y - pad)
            x2, y2 = min(w_f, x + w + pad), min(h_f, y + h + pad)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            crop_rs = cv2.resize(crop, (self._img_size, self._img_size))

            try:
                emotions, scores = self._recognizer.predict_emotions(
                    crop_rs, logits=False
                )
            except Exception as exc:
                logger.debug("[EMOTION] predict_emotions error: %s", exc)
                continue

            if not emotions:
                continue

            dominant = emotions[0]
            conf = float(scores[0].max())
            label = _LABEL_MAP.get(dominant, "neutral")

            logger.info(f"[EMOTION] EmotiEffLib: {dominant} → {label} ({conf:.0%})")

            if conf > best_conf:
                best_conf = conf
                best_label = label
                best_bbox = [float(x1), float(y1), float(x2), float(y2)]

        return EmotionDetectionResult(
            label=best_label,
            confidence=round(best_conf, 4),
            face_count=len(faces),
            bbox=best_bbox,
        )

    def _cascade_heuristic(
        self,
        frame: np.ndarray,
        faces: list[tuple[int, int, int, int]],
    ) -> EmotionDetectionResult:
        """Brightness-based heuristic when EmotiEffLib is not available."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightnesses = [
            float(np.mean(gray[y : y + h, x : x + w]))
            for x, y, w, h in faces
            if gray[y : y + h, x : x + w].size > 0
        ]
        avg = float(np.mean(brightnesses)) if brightnesses else 128.0

        if avg < 80:
            label, conf = "distressed", 0.58
        elif avg < 100:
            label, conf = "fearful", 0.55
        else:
            label, conf = "neutral", 0.62

        x, y, w, h = faces[0]
        return EmotionDetectionResult(
            label=label,
            confidence=conf,
            face_count=len(faces),
            bbox=[float(x), float(y), float(x + w), float(y + h)],
        )


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    detector = EmotionDetector()
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1])
        if img is not None:
            print(detector.detect(img))
        else:
            print("Could not load image")
    else:
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        print(detector.detect(dummy))
