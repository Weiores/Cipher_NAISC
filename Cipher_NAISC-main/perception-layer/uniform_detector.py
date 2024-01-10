"""
Uniform / civilian classifier for Cipher_NAISC.

Uses a lightweight YOLOv8 model to classify whether detected persons are
wearing a security/police uniform or civilian clothing.  Falls back to a
colour-heuristic approach if no model is available.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None  # type: ignore[assignment]

# Path for optional dedicated uniform model (env var override)
_ENV_MODEL = os.getenv("UNIFORM_MODEL_PATH")


@dataclass
class UniformDetectionResult:
    """Single-frame uniform/civilian detection result."""

    uniform_present: bool
    confidence: float
    bbox: list[float] = field(default_factory=list)
    label: str = "civilian"
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class UniformDetector:
    """Detect whether persons in frame are in uniform.

    Primary: YOLOv8 model (if UNIFORM_MODEL_PATH is set).
    Fallback: simple HSV colour heuristic (dark navy/khaki dominant = likely uniform).
    """

    # HSV ranges for common uniform colours (dark navy, khaki, black)
    _UNIFORM_COLOUR_RANGES = [
        ((100, 50, 20), (130, 255, 120)),   # dark navy blue
        ((20,  30, 80), (35,  120, 200)),   # khaki / tan
        ((0,   0,   0), (180,  50,  80)),   # black / very dark
    ]

    def __init__(self) -> None:
        self.model: Any = None
        if YOLO and _ENV_MODEL:
            try:
                self.model = YOLO(_ENV_MODEL)
                logger.info("[UNIFORM] Loaded model from %s", _ENV_MODEL)
            except Exception as exc:
                logger.warning("[UNIFORM] Could not load model: %s", exc)
        if self.model is None:
            logger.info("[UNIFORM] Using colour-heuristic fallback backend")

    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> UniformDetectionResult:
        """Classify uniform presence in a single BGR frame."""
        logger.info("[UNIFORM] Disabled - returning civilian")
        return UniformDetectionResult(uniform_present=False, confidence=0.0, label="civilian")

    # ------------------------------------------------------------------

    def _yolo_detect(self, frame: np.ndarray) -> UniformDetectionResult:
        results = self.model.predict(source=frame, verbose=False)
        if not results:
            return UniformDetectionResult(uniform_present=False, confidence=0.5)

        result = results[0]
        boxes = getattr(result, "boxes", None)
        names = getattr(result, "names", {})

        if boxes is None or len(boxes) == 0:
            return UniformDetectionResult(uniform_present=False, confidence=0.5)

        for box in boxes:
            label = str(names.get(int(box.cls.item()), "")).lower()
            conf = float(box.conf.item())
            if "uniform" in label or "officer" in label or "police" in label:
                bbox = [round(float(v), 2) for v in box.xyxy.tolist()[0]]
                return UniformDetectionResult(
                    uniform_present=True,
                    confidence=round(conf, 4),
                    bbox=bbox,
                    label="uniform",
                )

        return UniformDetectionResult(uniform_present=False, confidence=0.6)

    def _colour_heuristic(self, frame: np.ndarray) -> UniformDetectionResult:
        """Estimate uniform presence based on dominant HSV colour."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, w = hsv.shape[:2]
        total_pixels = h * w
        uniform_pixels = 0

        for lower, upper in self._UNIFORM_COLOUR_RANGES:
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            uniform_pixels += int(np.count_nonzero(mask))

        ratio = uniform_pixels / total_pixels if total_pixels else 0.0
        uniform_present = ratio > 0.15
        confidence = min(0.85, 0.40 + ratio * 2.0) if uniform_present else 0.55

        logger.debug("[UNIFORM] Colour ratio=%.3f → uniform=%s", ratio, uniform_present)
        return UniformDetectionResult(
            uniform_present=uniform_present,
            confidence=round(confidence, 4),
            label="uniform" if uniform_present else "civilian",
        )


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.DEBUG)
    detector = UniformDetector()
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1])
        if img is not None:
            print(detector.detect(img))
        else:
            print("Could not load image")
    else:
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        print(detector.detect(dummy))
