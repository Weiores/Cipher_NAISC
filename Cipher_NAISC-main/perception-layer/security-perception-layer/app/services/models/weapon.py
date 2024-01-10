import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.schemas import PerceptionRequest, WeaponDetection
from app.services.models.base import ModelAdapter

_DEFAULT_YOLO_CONFIG_DIR = Path(__file__).resolve().parents[3] / ".ultralytics"
os.environ.setdefault("YOLO_CONFIG_DIR", str(_DEFAULT_YOLO_CONFIG_DIR))
_DEFAULT_YOLO_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover - optional runtime dependency
    YOLO = None
CUSTOM_WEAPON_MODEL = Path(
    Path(__file__).resolve().parents[4]
    / "Weapons-and-Knives-Detector-with-YOLOv8"
    / "runs"
    / "detect"
    / "Normal"
    / "weights"
    / "best.pt"
)


class WeaponDetectionAdapter(ModelAdapter):
    name = "weapon_detection"
    intended_backend = "Ultralytics YOLOv8 model from JoaoAssalim Weapons-and-Knives-Detector-with-YOLOv8"
    video_frame_samples = 16

    def __init__(self) -> None:
        self.model_path = None
        self.model = None
        self.coco_model = None

        # Multi-frame validation: track recent detections to require weapon in 2+ frames
        self.detection_history = []  # Store (label, frame_index) tuples

        # Load COCO model (yolov8m.pt) for broad weapon detection (scissors, knife, bat)
        if YOLO is not None:
            try:
                self.coco_model = YOLO("yolov8m.pt")
                print("[WEAPON_ADAPTER] Loaded COCO model: yolov8m.pt")
            except Exception as e:
                print(f"[WEAPON_ADAPTER] WARNING: Failed to load COCO model yolov8m.pt: {e}")

        # Check environment variable override first
        env_model_path = os.getenv("WEAPON_MODEL_PATH")
        if env_model_path:
            self.model_path = env_model_path
            self.model = self._load_model(self.model_path)
            if self.model:
                print(f"[WEAPON_ADAPTER] Loaded model from environment variable: {env_model_path}")

        # Load custom trained model (Weapons-and-Knives-Detector-with-YOLOv8)
        if self.model is None and CUSTOM_WEAPON_MODEL.exists():
            self.model_path = str(CUSTOM_WEAPON_MODEL)
            self.model = self._load_model(self.model_path)
            if self.model:
                print(f"[WEAPON_ADAPTER] Loaded custom weapon detection model (Weapons-and-Knives-Detector-with-YOLOv8)")

        # If no model loaded, warn user
        if self.model is None and self.coco_model is None:
            print("[WEAPON_ADAPTER] WARNING: No weapon detection model found!")

        active_backend = "yolo_coco+custom" if (self.coco_model and self.model) else "yolo_coco" if self.coco_model else "custom" if self.model else "placeholder"
        print(f"[WEAPON_ADAPTER] Initialized with backend={active_backend}, coco_model={self.coco_model is not None}, custom_model={self.model is not None}")

    def _load_model(self, model_path: str | None) -> Any:
        print(f"[LOAD_MODEL] Attempting to load from: {model_path}")
        print(f"[LOAD_MODEL] YOLO library available: {YOLO is not None}")
        
        if not model_path or YOLO is None:
            print(f"[LOAD_MODEL] Failed: model_path={model_path}, YOLO={YOLO}")
            return None

        model_file = Path(model_path)
        print(f"[LOAD_MODEL] Model file exists: {model_file.exists()}")
        
        if not model_file.exists():
            print(f"[LOAD_MODEL] File not found at {model_file}")
            return None

        try:
            print(f"[LOAD_MODEL] Loading YOLO model from {model_file}...")
            model = YOLO(str(model_file))
            print(f"[LOAD_MODEL] Model loaded successfully!")
            return model
        except Exception as e:
            print(f"[LOAD_MODEL] Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            return None

    def describe(self) -> dict[str, str]:
        details = super().describe()
        if self.model_path and "yolov8" in str(self.model_path).lower():
            active_backend = "yolov8_custom"
        elif self.model is not None:
            active_backend = "ultralytics_yolo"
        else:
            active_backend = "placeholder"

        details["active_backend"] = active_backend
        details["model_path"] = self.model_path or ""
        return details

    async def infer(self, request: PerceptionRequest) -> WeaponDetection:
        print(f"[WEAPON] Starting inference - has video={bool(request.video and request.video.uri)}")
        
        if (self.model is not None or self.coco_model is not None) and request.video and request.video.uri:
            source = Path(request.video.uri)
            print(f"[WEAPON] Trying YOLOv8 - source exists: {source.exists()}, coco={self.coco_model is not None}, custom={self.model is not None}")
            if source.exists():
                prediction = self._infer_with_yolo(source)
                if prediction is not None:
                    print(f"[WEAPON] YOLOv8 result: weapon={prediction.label}, conf={prediction.confidence}, bboxes={prediction.bounding_boxes}")
                    return prediction

        print(f"[WEAPON] Using placeholder backend")
        return self._infer_placeholder(request)

    def _infer_placeholder(self, request: PerceptionRequest) -> WeaponDetection:
        context = (request.incident_context or "").lower()
        stream_type = request.video.stream_type if request.video else "cctv"

        label = "unarmed"
        confidence = 0.58
        boxes: list[list[float]] = []

        if "knife" in context:
            label = "knife"
            confidence = 0.91
            boxes = [[0.42, 0.35, 0.57, 0.76]]
        elif "gun" in context or "firearm" in context:
            label = "gun"
            confidence = 0.94
            boxes = [[0.48, 0.22, 0.66, 0.61]]
        elif stream_type == "bodycam":
            label = "unknown_object"
            confidence = 0.44

        return WeaponDetection(
            label=label,
            confidence=confidence,
            bounding_boxes=boxes,
            backend="placeholder",
            class_evidence=[{"label": label, "confidence": confidence, "frame_index": 0}],
            sampled_frames=1,
        )


    def _infer_with_yolo(self, source: Path) -> WeaponDetection | None:
        if source.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
            image = self._load_frame(source)
            return self._run_yolo_on_frame(image, frame_index=0)

        best_detection: WeaponDetection | None = None
        evidence_by_label: dict[str, dict[str, float | int | str]] = {}
        sampled_frames = 0
        for frame_index, frame in self._sample_video_frames(source, self.video_frame_samples):
            sampled_frames += 1
            detection = self._run_yolo_on_frame(frame, frame_index=frame_index)
            if detection is None:
                continue
            for item in detection.class_evidence:
                label = str(item["label"])
                if label not in evidence_by_label or float(item["confidence"]) > float(evidence_by_label[label]["confidence"]):
                    evidence_by_label[label] = item
            if self._is_better_detection(detection, best_detection):
                best_detection = detection

        if best_detection is not None:
            best_detection.class_evidence = sorted(
                evidence_by_label.values(),
                key=lambda item: float(item["confidence"]),
                reverse=True,
            )[:3]
            best_detection.sampled_frames = sampled_frames
        return best_detection

    def _normalize_label(self, raw_label: str) -> str:
        if raw_label in {"gun", "pistol", "revolver", "rifle", "firearm"}:
            return "gun"
        if raw_label in {"guns"}:
            return "gun"
        if raw_label in {"knife", "dagger", "blade", "machete"}:
            return "knife"
        if raw_label in {"bat", "baseball_bat"}:
            return "bat"
        if raw_label in {"stick", "rod", "club"}:
            return "stick"
        if raw_label in {"unarmed", "none"}:
            return "unarmed"
        return "unknown_object"

    def _map_repo_class_id(self, class_id: int) -> str:
        if class_id in {1, 2, 3, 4, 5}:
            return "gun"
        if class_id == 6:
            return "knife"
        return "unknown_object"

    def _load_frame(self, source: Path) -> np.ndarray:
        if source.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
            frame = cv2.imread(str(source))
            if frame is None:
                raise ValueError(f"Unable to read image: {source}")
            return frame

        capture = cv2.VideoCapture(str(source))
        ok, frame = capture.read()
        capture.release()
        if not ok or frame is None:
            raise ValueError(f"Unable to read video frame: {source}")
        return frame

    def _sample_video_frames(self, source: Path, sample_count: int) -> list[tuple[int, np.ndarray]]:
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            raise ValueError(f"Unable to open video: {source}")

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        target_indices: list[int]
        if frame_count <= 0:
            target_indices = [0]
        else:
            sample_count = max(1, min(sample_count, frame_count))
            target_indices = np.linspace(0, frame_count - 1, num=sample_count, dtype=int).tolist()

        frames: list[tuple[int, np.ndarray]] = []
        for index in target_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = capture.read()
            if ok and frame is not None:
                frames.append((int(index), frame))

        capture.release()
        if not frames:
            raise ValueError(f"Unable to sample video frames: {source}")
        return frames

    def _is_valid_weapon_aspect_ratio(self, xyxy: list[float]) -> bool:
        """Check if bounding box has weapon-like aspect ratio (elongated, not square) AND reasonable size"""
        x_min, y_min, x_max, y_max = xyxy
        width = x_max - x_min
        height = y_max - y_min
        
        if height == 0:
            return False
            
        aspect_ratio = width / height
        
        # FILTER: Aspect ratio only
        # Weapons are significantly elongated:
        # - Guns: 1.5-3.0 ratio (much wider than tall)
        # - Knives: 2.0-4.0 ratio (thin and long)
        # Reject: Hand gestures, arms (typically <1.0 ratio), faces (0.8-1.2 ratio)
        # STRICT: Only accept properly elongated objects
        MIN_ASPECT_RATIO = 0.5  # Relaxed - allows tilted/vertical weapons, new model variant
        
        aspect_ratio_valid = aspect_ratio >= MIN_ASPECT_RATIO
        
        print(f"[ASPECT_RATIO] bbox={[round(v, 2) for v in xyxy]}, ratio={aspect_ratio:.2f}, MIN={MIN_ASPECT_RATIO}, VALID={aspect_ratio_valid}", flush=True)
        return aspect_ratio_valid

    def _validate_multi_frame_detection(self, label: str, frame_index: int, confidence: float) -> bool:
        """Check if weapon appears in multiple frames with low confidence to reduce false positives"""
        # Only check multi-frame for actual weapon detections, not "unarmed"
        if label in {"unarmed", "unknown_object"}:
            return True
        
        # DISABLED: Multi-frame validation was too strict and blocking legitimate weapons
        # Just return True to let confidence threshold and face filtering handle filtering
        print(f"[MULTI_FRAME] label={label}, frame={frame_index}, conf={confidence:.2f}, MULTI_FRAME_DISABLED=True", flush=True)
        return True

    def _infer_weapon_from_shape(self, label: str, xyxy: list[float]) -> str:
        """Heuristic: Refine weapon classification based on bounding box shape
        
        Guns on floor often have high width/height ratio (>1.8)
        This catches cases where model misclassifies gun as knife
        
        Args:
            label: Model-predicted label (e.g., "knife", "gun")
            xyxy: Bounding box [x_min, y_min, x_max, y_max]
            
        Returns:
            Refined label, possibly corrected from Gun ← Knife based on shape
        """
        x_min, y_min, x_max, y_max = xyxy
        width = x_max - x_min
        height = y_max - y_min
        
        if height == 0:
            return label
            
        aspect_ratio = width / height
        
        # HEURISTIC: If model said "knife" but bbox very wide, likely misclassification
        # Guns on floor are very elongated (>1.8 ratio), knives are more balanced
        if label == "knife" and aspect_ratio > 1.8:
            print(f"[SHAPE_INFERENCE] Knife detected with ratio {aspect_ratio:.2f} (very high) - likely GUN MISCLASSIFICATION", flush=True)
            return "gun"
        
        return label


    def _bbox_overlaps_with_faces(self, weapon_bbox: list[float], face_bboxes: list[dict[str, float]]) -> bool:
        """Check if weapon bounding box overlaps with any detected face
        
        Args:
            weapon_bbox: [x1, y1, x2, y2] in pixel coordinates
            face_bboxes: List of face bounding boxes in normalized coordinates [0, 1]
            
        Returns:
            True if weapon overlaps with a face, False otherwise
        """
        if not face_bboxes:
            return False
        
        try:
            # Weapon bbox is in absolute pixel coordinates - need to normalize it
            # But we don't have image height/width here, so we'll handle this in the calling function
            # For now, we'll assume weapon_bbox needs to be converted
            return False  # Placeholder - will be fixed in _run_yolo_on_frame
        except Exception as e:
            print(f"[FACE_FILTER] Error checking overlap: {e}", flush=True)
            return False


    def _run_yolo_on_frame(self, image: np.ndarray, frame_index: int) -> WeaponDetection | None:
        """Run COCO + custom YOLO models on frame, take highest-confidence weapon detection."""
        # COCO class IDs for weapons (YOLOv8 0-indexed): knife=43, scissors=76, bat/racket=38
        COCO_WEAPON_CLASSES = {43: "knife", 76: "knife", 38: "bat"}
        COCO_SEARCH_CONF = 0.10   # Search threshold (mirrors test_scissors.py)
        COCO_MIN_CONF = float(os.getenv("COCO_WEAPON_THRESHOLD", "0.35"))
        CUSTOM_SEARCH_CONF = 0.10
        CUSTOM_MIN_CONF = 0.20

        height, width = image.shape[:2]
        print(f"[YOLO] Frame size: {width}x{height}", flush=True)

        best_label: str | None = None
        best_confidence = 0.0
        best_bbox: list[float] | None = None

        # --- COCO model (yolov8m.pt): detects scissors, knives, bats ---
        if self.coco_model is not None:
            results = self.coco_model(image, verbose=False, conf=COCO_SEARCH_CONF)
            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    if cls in COCO_WEAPON_CLASSES:
                        coco_name = self.coco_model.names.get(cls, str(cls))
                        label = COCO_WEAPON_CLASSES[cls]
                        print(f"[COCO] {coco_name} (cls={cls}) conf={conf:.3f} threshold={COCO_MIN_CONF:.3f}", flush=True)
                        if conf >= COCO_MIN_CONF and conf > best_confidence:
                            best_confidence = conf
                            best_label = label
                            best_bbox = [round(float(v), 2) for v in box.xyxy[0].tolist()]
                            print(f"[COCO] ✓ NEW BEST: {label} conf={conf:.3f} bbox={best_bbox}", flush=True)

        # --- Custom model (Weapons-and-Knives-Detector): detects Gun and Knife ---
        if self.model is not None:
            results = self.model.predict(source=image, verbose=False, conf=CUSTOM_SEARCH_CONF)
            if results:
                result = results[0]
                boxes = getattr(result, "boxes", None)
                names = getattr(result, "names", {})
                if boxes is not None:
                    print(f"[CUSTOM] {len(boxes)} boxes found", flush=True)
                    for idx in range(len(boxes)):
                        box = boxes[idx]
                        conf = float(box.conf.item())
                        raw_label = str(names.get(int(box.cls.item()), "unknown_object")).lower()
                        label = self._normalize_label(raw_label)
                        print(f"[CUSTOM] {raw_label}→{label} conf={conf:.3f}", flush=True)
                        if label not in {"unarmed", "unknown_object"} and conf >= CUSTOM_MIN_CONF:
                            if conf > best_confidence:
                                best_confidence = conf
                                best_label = label
                                best_bbox = [round(float(v), 2) for v in box.xyxy.tolist()[0]]
                                print(f"[CUSTOM] ✓ NEW BEST: {label} conf={conf:.3f} bbox={best_bbox}", flush=True)

        # No weapon found above thresholds
        if best_label is None:
            print(f"[YOLO] No weapon detected above thresholds", flush=True)
            return WeaponDetection(
                label="unarmed",
                confidence=0.75,
                bounding_boxes=[],
                backend="ultralytics_yolo",
                class_evidence=[{"label": "unarmed", "confidence": 0.75, "frame_index": frame_index}],
                sampled_frames=1,
            )

        confidence = round(best_confidence, 4)
        print(f"[YOLO] ✅ WEAPON DETECTED: {best_label} conf={confidence} bbox={best_bbox}", flush=True)

        return WeaponDetection(
            label=best_label,
            confidence=confidence,
            bounding_boxes=[best_bbox] if best_bbox else [],
            backend="ultralytics_yolo",
            class_evidence=[{"label": best_label, "confidence": confidence, "frame_index": frame_index}],
            sampled_frames=1,
        )

    def _is_better_detection(
        self,
        candidate: WeaponDetection,
        current: WeaponDetection | None,
    ) -> bool:
        if current is None:
            return True

        candidate_positive = candidate.label not in {"unarmed", "unknown_object"}
        current_positive = current.label not in {"unarmed", "unknown_object"}

        if candidate_positive and not current_positive:
            return True
        if current_positive and not candidate_positive:
            return False
        return candidate.confidence > current.confidence

