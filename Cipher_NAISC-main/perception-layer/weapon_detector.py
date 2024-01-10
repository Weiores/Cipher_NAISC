"""
YOLOv8-based weapon detector for Cipher_NAISC.

Primary model: custom Weapons-and-Knives-Detector (classes: Gun, Knife).
COCO fallback: yolov8n.pt — catches scissors (76), knife (43), bat (38).
               Bottle (39) is excluded to prevent false positives.

Bounding boxes are drawn directly onto the frame for every detection >= 0.20.

Dual-mode detection:
  File source  — pure YOLOv8, no Groq Vision calls (full speed).
  Webcam (0)   — Groq Vision called only when YOLOv8 confidence is in the
                 uncertain zone [0.15, 0.50], rate-limited to 1 call per
                 GROQ_VISION_MIN_INTERVAL seconds.
"""

from __future__ import annotations

import base64
import logging
import os
import time as _time_module
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Confirmation buffer — module-level so it persists across calls
_CONFIRMATION_FRAMES: int = int(os.getenv("WEAPON_CONFIRMATION_FRAMES", "3"))
_confirmation_buffer: deque[float] = deque(maxlen=_CONFIRMATION_FRAMES)

# Groq Vision rate-limiting state
_last_groq_vision_call: float = 0.0
_GROQ_UNCERTAIN_LOW  = 0.15   # below this → nothing detected, skip Groq
_GROQ_UNCERTAIN_HIGH = 0.50   # above this → YOLOv8 confident, skip Groq

# Whether the current VIDEO_SOURCE is a webcam (digit) or file
_VIDEO_SOURCE = os.getenv("VIDEO_SOURCE", "0")
_IS_WEBCAM = str(_VIDEO_SOURCE).strip().isdigit()

# Path to the trained YOLOv8 weights bundled with the repo
_BUNDLED_MODEL = (
    Path(__file__).resolve().parent
    / "Weapons-and-Knives-Detector-with-YOLOv8"
    / "runs"
    / "detect"
    / "Normal"
    / "weights"
    / "best.pt"
)

# COCO class IDs that indicate a potential weapon/dangerous object.
# Bottle (39) is intentionally excluded — too many false positives.
_COCO_WEAPON_CLASSES: dict[int, str] = {
    38: "bat",       # baseball bat
    43: "knife",
    76: "scissors",
}
_COCO_CONF_THRESHOLD = 0.10   # lowered: scissors score 0.06–0.10 in live webcam
_ANNOTATE_MIN_CONF   = 0.20   # draw box for any detection at or above this

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None  # type: ignore[assignment]


@dataclass
class WeaponDetectionResult:
    """Single-frame weapon detection result."""

    label: str
    confidence: float
    bbox: list[float] = field(default_factory=list)
    confirmed: bool = False  # True when confidence >= threshold for N consecutive frames
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class WeaponDetector:
    """YOLOv8-based weapon detector with COCO fallback.

    Primary: custom Gun/Knife model.
    Fallback: yolov8n.pt (COCO) to catch scissors (76), knife (43), bat (38).
    Bounding boxes are drawn on every frame for debugging.
    """

    _LABEL_MAP: dict[str, str] = {
        "gun": "gun", "pistol": "gun", "revolver": "gun",
        "rifle": "gun", "firearm": "gun", "shotgun": "gun",
        "knife": "knife", "blade": "knife", "dagger": "knife",
        "scissors": "scissors",
        "bat": "bat", "baseball bat": "bat", "stick": "stick",
        "unarmed": "unarmed",
    }

    # Colours per label (BGR)
    _BOX_COLORS: dict[str, tuple[int, int, int]] = {
        "gun":      (0, 0, 255),    # red
        "knife":    (0, 0, 220),    # red-ish
        "scissors": (0, 80, 255),   # orange-red
        "bat":      (0, 120, 255),  # orange
        "unknown_object": (0, 0, 200),
    }
    _DEFAULT_COLOR = (0, 0, 255)

    def __init__(self) -> None:
        model_path = os.getenv("WEAPON_MODEL_PATH") or (
            str(_BUNDLED_MODEL) if _BUNDLED_MODEL.exists() else None
        )
        self.model: Any = None
        self.coco_model: Any = None

        if YOLO is None:
            logger.warning("[WEAPON] ultralytics not installed; using placeholder backend")
        else:
            # Primary model
            if model_path:
                try:
                    self.model = YOLO(model_path)
                    logger.info(f"[WEAPON] Loaded primary model from {model_path}")
                    names = getattr(self.model, "names", {})
                    logger.info(f"[WEAPON] Primary model classes: {list(names.values())}")
                except Exception as exc:
                    logger.error(f"[WEAPON] Failed to load primary model: {exc}")
            else:
                logger.warning("[WEAPON] No primary model weights found")

            # COCO fallback (yolov8n.pt — auto-downloads on first use)
            try:
                self.coco_model = YOLO("yolov8n.pt")
                logger.info(
                    f"[WEAPON] COCO fallback model ready (watching classes: { {v: k for k, v in _COCO_WEAPON_CLASSES.items()} })"
                )
            except Exception as exc:
                logger.warning(f"[WEAPON] COCO fallback model unavailable: {exc}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> WeaponDetectionResult:
        """Run weapon detection on a single BGR frame.

        Runs both the primary model and the COCO fallback, annotates the
        frame with bounding boxes, and returns the best detection.
        Updates the module-level confirmation buffer.
        """
        logger.info(f"[DEBUG] Weapon detector received frame shape={frame.shape}")
        threshold = float(os.getenv("DANGER_WEAPON_THRESHOLD", "0.50"))

        all_detections: list[dict] = []  # {label, raw, conf, bbox, source}

        # ── Primary model ─────────────────────────────────────────────────
        if self.model is not None:
            primary_dets = self._run_primary(frame)
            all_detections.extend(primary_dets)

        # ── COCO fallback ─────────────────────────────────────────────────
        if self.coco_model is not None:
            coco_dets = self._run_coco(frame)
            logger.info(f"[DEBUG] COCO results: {len(coco_dets)} boxes")
            all_detections.extend(coco_dets)

        # ── Frame scan log (one line per frame for easy grepping) ────────
        custom_dets = [d for d in all_detections if d["source"] == "primary"]
        coco_dets   = [d for d in all_detections if d["source"] == "coco"]
        custom_conf = max((d["conf"] for d in custom_dets), default=0.0)
        if coco_dets:
            best_coco = max(coco_dets, key=lambda d: d["conf"])
            coco_str  = f"{best_coco['label']}:{best_coco['conf']:.2f}"
        else:
            coco_str = "none:0.00"
        logger.info(
            "[WEAPON] Frame scan: custom=%.2f coco=%s",
            custom_conf, coco_str,
        )

        # ── Log every frame ───────────────────────────────────────────────
        if all_detections:
            summary = ", ".join(
                f"{d['label']}({d['source']}):{d['conf']:.2f}"
                for d in sorted(all_detections, key=lambda x: -x["conf"])
            )
        else:
            summary = "none"
        logger.info(f"[WEAPON] Frame detections: {summary}")

        # ── Annotate frame with boxes (all detections >= 0.20) ────────────
        self._annotate_frame(frame, all_detections)

        # ── Pick best result ──────────────────────────────────────────────
        # Exclude "unarmed" from selection — primary model may label a person
        # as "unarmed" at high confidence, which would mask a real weapon
        # detection from COCO at lower confidence.
        weapon_dets = [
            d for d in all_detections
            if d["label"] not in {"unarmed", "unknown", "unknown_object"}
        ]
        best = max(weapon_dets, key=lambda d: d["conf"], default=None)

        if best is None or best["conf"] < 0.35:
            result = WeaponDetectionResult(label="unarmed", confidence=0.0)
        else:
            result = WeaponDetectionResult(
                label=best["label"],
                confidence=round(best["conf"], 4),
                bbox=best["bbox"],
            )

        # ── Groq Vision (webcam only, uncertain zone) ─────────────────────
        # Only triggered when YOLOv8 is uncertain: 0.15 ≤ conf ≤ 0.50.
        # File-source demos skip this entirely to run at full speed.
        if (
            _IS_WEBCAM
            and best is not None
            and _GROQ_UNCERTAIN_LOW <= best["conf"] <= _GROQ_UNCERTAIN_HIGH
        ):
            groq_result = self._groq_vision_check(frame, best["label"], best["conf"])
            if groq_result is not None:
                result = groq_result

        # ── Confirmation buffer ───────────────────────────────────────────
        is_weapon = result.label not in {"unarmed", "unknown", "unknown_object"}
        _confirmation_buffer.append(result.confidence if is_weapon else 0.0)

        confirmed = (
            len(_confirmation_buffer) == _confirmation_buffer.maxlen
            and all(c >= threshold for c in _confirmation_buffer)
        )
        result.confirmed = confirmed

        logger.info(
            "[WEAPON] Final result: %s %.4f | buffer=%s confirmed=%s",
            result.label, result.confidence,
            [round(c, 3) for c in _confirmation_buffer],
            confirmed,
        )

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_primary(self, frame: np.ndarray) -> list[dict]:
        """Run the custom weapon model; return raw detections above MIN_CONF."""
        MIN_CONF = 0.10
        LOG_CONF  = 0.05  # lower threshold just for verbose logging
        try:
            results = self.model.predict(source=frame, verbose=False, conf=LOG_CONF)
        except Exception as exc:
            logger.warning(f"[WEAPON] Primary model predict failed: {exc}")
            return []

        if not results:
            logger.info("[WEAPON] CUSTOM model raw: []")
            return []
        result = results[0]
        boxes = getattr(result, "boxes", None)
        names = getattr(result, "names", {})

        if boxes is None or len(boxes) == 0:
            logger.info("[WEAPON] CUSTOM model raw: []")
            return []

        all_raw = []
        dets = []
        for box in boxes:
            conf = float(box.conf.item())
            raw = str(names.get(int(box.cls.item()), "unknown")).lower()
            all_raw.append((raw, round(conf, 3)))
            if conf < MIN_CONF:
                continue
            label = self._LABEL_MAP.get(raw, "unknown_object")
            bbox = [round(float(v), 2) for v in box.xyxy.tolist()[0]]
            dets.append({"label": label, "raw": raw, "conf": conf,
                         "bbox": bbox, "source": "primary"})

        logger.info(f"[WEAPON] CUSTOM model raw: {all_raw}")
        return dets

    def _run_coco(self, frame: np.ndarray) -> list[dict]:
        """Run yolov8n COCO model; return only weapon-class detections."""
        LOG_CONF = 0.05  # very low — log everything, filter on return
        try:
            results = self.coco_model.predict(
                source=frame, verbose=False, conf=LOG_CONF
            )
        except Exception as exc:
            logger.warning(f"[WEAPON] COCO model predict failed: {exc}")
            return []

        if not results:
            logger.info("[WEAPON] COCO model raw: []")
            return []
        result = results[0]
        boxes = getattr(result, "boxes", None)
        coco_names = getattr(result, "names", {})

        if boxes is None or len(boxes) == 0:
            logger.info("[WEAPON] COCO model raw: []")
            return []

        all_raw = []
        dets = []
        for box in boxes:
            cls_id = int(box.cls.item())
            conf   = float(box.conf.item())
            cname  = str(coco_names.get(cls_id, cls_id)).lower()
            all_raw.append((cname, cls_id, round(conf, 3)))

            if cls_id not in _COCO_WEAPON_CLASSES:
                continue
            if conf < _COCO_CONF_THRESHOLD:
                continue
            raw = _COCO_WEAPON_CLASSES[cls_id]
            label = self._LABEL_MAP.get(raw, raw)
            bbox = [round(float(v), 2) for v in box.xyxy.tolist()[0]]
            dets.append({"label": label, "raw": raw, "conf": conf,
                         "bbox": bbox, "source": "coco"})

        logger.info(f"[WEAPON] COCO model raw: {all_raw}")
        return dets

    def _annotate_frame(self, frame: np.ndarray, detections: list[dict]) -> None:
        """Draw bounding boxes and labels on the frame in-place."""
        for det in detections:
            if det["conf"] < _ANNOTATE_MIN_CONF:
                continue
            bbox = det["bbox"]
            if len(bbox) != 4:
                continue

            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            color = self._BOX_COLORS.get(det["label"], self._DEFAULT_COLOR)
            label_text = f"{det['label'].upper()} {det['conf']*100:.0f}%"
            if det["source"] == "coco":
                label_text += " [COCO]"

            # Rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=3)

            # Label background
            font      = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.65
            thickness  = 2
            (tw, th), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
            ty = max(y1 - 6, th + 4)
            cv2.rectangle(frame, (x1, ty - th - baseline - 2),
                          (x1 + tw + 4, ty + 2), color, cv2.FILLED)

            # Label text (black on coloured background)
            cv2.putText(frame, label_text, (x1 + 2, ty - baseline),
                        font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # Groq Vision fallback (webcam mode, uncertain zone only)
    # ------------------------------------------------------------------

    def _groq_vision_check(
        self,
        frame: np.ndarray,
        yolo_label: str,
        yolo_conf: float,
    ) -> "WeaponDetectionResult | None":
        """Call Groq Vision to resolve an uncertain YOLOv8 detection.

        Only executed when:
          • GROQ_VISION_ENABLED=true
          • At least GROQ_VISION_MIN_INTERVAL seconds since the last call
          • YOLOv8 confidence is in [GROQ_UNCERTAIN_LOW, GROQ_UNCERTAIN_HIGH]
        Returns a WeaponDetectionResult on success, None to keep YOLOv8 result.
        """
        global _last_groq_vision_call

        if os.getenv("GROQ_VISION_ENABLED", "true").lower() != "true":
            return None

        min_interval = float(os.getenv("GROQ_VISION_MIN_INTERVAL", "10"))
        now = _time_module.time()
        if now - _last_groq_vision_call < min_interval:
            return None

        api_key = os.getenv("GROQ_API_KEY", "")
        if not api_key:
            return None

        logger.info(
            "[WEAPON] Groq Vision triggered (YOLOv8 uncertain: %.2f, label=%s)",
            yolo_conf, yolo_label,
        )

        try:
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                return None
            b64_frame = base64.b64encode(buf.tobytes()).decode("utf-8")

            from groq import Groq  # type: ignore[import]
            client = Groq(api_key=api_key)

            response = client.chat.completions.create(
                model=os.getenv("GROQ_VISION_MODEL", "llama-4-scout-17b-16e-instruct"),
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64_frame}"},
                        },
                        {
                            "type": "text",
                            "text": (
                                f"Security camera analysis. YOLOv8 detected a possible "
                                f"'{yolo_label}' at {yolo_conf:.0%} confidence. "
                                "Is there a weapon (gun, knife, scissors, bat) clearly "
                                "visible? Reply with ONLY: weapon_type|confidence_0_to_1 "
                                "Examples: 'gun|0.92'  'knife|0.78'  'unarmed|0.95'"
                            ),
                        },
                    ],
                }],
                max_tokens=20,
                temperature=0.1,
            )

            _last_groq_vision_call = _time_module.time()

            raw = response.choices[0].message.content.strip().lower()
            logger.info(f"[WEAPON] Groq Vision response: {raw}")

            if "|" in raw:
                label_raw, conf_raw = raw.split("|", 1)
                label_raw = label_raw.strip()
                try:
                    conf = max(0.0, min(1.0, float(conf_raw.strip())))
                except ValueError:
                    conf = 0.5
                label = self._LABEL_MAP.get(label_raw, label_raw)
                return WeaponDetectionResult(label=label, confidence=round(conf, 4))

        except Exception as exc:
            logger.warning(f"[WEAPON] Groq Vision call failed: {exc}")

        return None

    def _normalize_label(self, raw: str) -> str:
        return self._LABEL_MAP.get(raw.lower(), "unknown_object")


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    detector = WeaponDetector()
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1])
        if img is not None:
            result = detector.detect(img)
            print(result)
            cv2.imwrite("annotated_output.jpg", img)
            print("Annotated frame saved to annotated_output.jpg")
        else:
            print("Could not load image")
    else:
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        print(detector.detect(dummy))
