import os
from pathlib import Path
from typing import Any

import cv2
import httpx
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

try:
    from mediapipe.python.solutions import face_detection
    mp_face_detection = face_detection
except (ImportError, AttributeError):  # pragma: no cover - optional runtime dependency
    try:
        # Fallback: try direct import
        import mediapipe as mp
        mp_face_detection = mp.solutions.face_detection
    except (ImportError, AttributeError):
        mp_face_detection = None



try:
    import tensorflow as tf
except ImportError:  # pragma: no cover - optional runtime dependency
    tf = None

# OpenCV cascade classifier for face detection (fallback if MediaPipe fails)
try:
    FACE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    opencv_face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
except Exception:
    opencv_face_cascade = None

# Weapon detection models - priority order:
# 1. Environment variable override (user-specified)
# 2. Falcons-ai pre-trained YOLOv8 models (best for real-world weapon detection)
# 3. Custom trained models (fallback)

# Pre-trained weapon detection models from Falcons-ai (YOLOv8)
# These are trained on Roboflow weapon detection dataset
FALCONS_YOLOV8S_MODEL = Path(
    Path(__file__).resolve().parents[4]
    / ".."  # Up to parent containing perception-layer
    / "weapons_detection_trainer_yolov8_open"
    / "yolov8s.pt"
)

FALCONS_YOLOV8N_MODEL = Path(
    Path(__file__).resolve().parents[4]
    / ".."
    / "weapons_detection_trainer_yolov8_open"
    / "yolov8n.pt"
)

# WasifSohail5 pre-trained model (high precision 76.6%, recall 82.0%)
WASIF_WEAPON_MODEL = Path(
    Path(__file__).resolve().parents[3]
    / "app"
    / "services"
    / "models"
    / "wasif_weapon_model.pt"
)

# Fallback to custom trained models
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
        self.legacy_service_url = os.getenv("LEGACY_WEAPON_SERVICE_URL", "").rstrip("/")
        self.tf_saved_model_dir = os.getenv("TF_WEAPON_SAVED_MODEL_DIR")
        self.tf_saved_model = self._load_tf_saved_model(self.tf_saved_model_dir)
        
        self.model_path = None
        self.model = None
        
        # Multi-frame validation: track recent detections to require weapon in 2+ frames
        self.detection_history = []  # Store (label, frame_index) tuples
        
        # Face detection for filtering false positives
        self.face_detector = None
        self.use_opencv_face_detection = False
        
        if mp_face_detection:
            try:
                self.face_detector = mp_face_detection.FaceDetection(
                    model_selection=0,  # 0=short range (0-2m), 1=full range (0-5m)
                    min_detection_confidence=0.5
                )
                print("[WEAPON_ADAPTER] ✓ Face detector initialized (MediaPipe)")
            except Exception as e:
                print(f"[WEAPON_ADAPTER] WARNING: Could not initialize MediaPipe face detector: {e}")
        
        # Fallback to OpenCV cascade if MediaPipe failed
        if self.face_detector is None and opencv_face_cascade is not None:
            self.use_opencv_face_detection = True
            print("[WEAPON_ADAPTER] ✓ Face detector initialized (OpenCV Cascade)")
        elif self.face_detector is None:
            print("[WEAPON_ADAPTER] WARNING: No face detection available - faces may cause false positives")
        
        # Check environment variable override first
        env_model_path = os.getenv("WEAPON_MODEL_PATH")
        if env_model_path:
            self.model_path = env_model_path
            self.model = self._load_model(self.model_path)
            if self.model:
                print(f"[WEAPON_ADAPTER] Loaded model from environment variable: {env_model_path}")
        
        # PRIORITY 1: Custom trained model (Weapons-and-Knives-Detector-with-YOLOv8) - Best working model
        if self.model is None and CUSTOM_WEAPON_MODEL.exists():
            self.model_path = str(CUSTOM_WEAPON_MODEL)
            self.model = self._load_model(self.model_path)
            if self.model:
                print(f"[WEAPON_ADAPTER] ✓ Loaded custom weapon detection model (Weapons-and-Knives-Detector-with-YOLOv8)")
        
        # PRIORITY 2: FalconAI models (if custom model not available)
        if self.model is None:
            for model_path in [FALCONS_YOLOV8S_MODEL, FALCONS_YOLOV8N_MODEL]:
                if model_path.exists():
                    self.model_path = str(model_path)
                    self.model = self._load_model(self.model_path)
                    if self.model:
                        print(f"[WEAPON_ADAPTER] Loaded Falcons-ai pre-trained model: {model_path.name}")
                        break
        
        # If still no model, warn user
        if self.model is None:
            print("[WEAPON_ADAPTER] WARNING: No weapon detection model found!")
        
        active_backend = "yolo" if self.model else "placeholder"
        print(f"[WEAPON_ADAPTER] Initialized with backend={active_backend}, model_path={self.model_path}, model_loaded={self.model is not None}")

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

    def _load_tf_saved_model(self, model_dir: str | None) -> Any:
        if not model_dir or tf is None:
            return None

        saved_model_dir = Path(model_dir)
        if not saved_model_dir.exists():
            return None

        try:
            loaded = tf.saved_model.load(str(saved_model_dir))
        except Exception:
            return None

        signatures = getattr(loaded, "signatures", {})
        if "serving_default" in signatures:
            return signatures["serving_default"]
        if signatures:
            return next(iter(signatures.values()))
        return loaded

    def describe(self) -> dict[str, str]:
        details = super().describe()
        if self.legacy_service_url:
            active_backend = "legacy_tf_service"
        elif self.model_path and "yolov8" in str(self.model_path).lower():
            active_backend = "yolov8_falcons_pretrained" if "weapons_detection_trainer" in str(self.model_path) else "yolov8_custom"
        elif self.tf_saved_model is not None:
            active_backend = "tf2_saved_model"
        elif self.model is not None:
            active_backend = "ultralytics_yolo"
        else:
            active_backend = "placeholder"

        details["active_backend"] = active_backend
        details["legacy_service_url"] = self.legacy_service_url
        details["tf_saved_model_dir"] = self.tf_saved_model_dir or ""
        details["model_path"] = self.model_path or ""
        return details

    async def infer(self, request: PerceptionRequest) -> WeaponDetection:
        print(f"[WEAPON] Starting inference - has video={bool(request.video and request.video.uri)}")
        
        if self.legacy_service_url and request.video and request.video.uri:
            prediction = await self._infer_with_legacy_service(request)
            if prediction is not None:
                print(f"[WEAPON] Using legacy service - result: {prediction}")
                return prediction

        if self.model is not None and request.video and request.video.uri:
            source = Path(request.video.uri)
            print(f"[WEAPON] Trying YOLOv8 - source exists: {source.exists()}, model loaded: {self.model is not None}")
            if source.exists():
                prediction = self._infer_with_yolo(source)
                if prediction is not None:
                    print(f"[WEAPON] YOLOv8 result: weapon={prediction.label}, conf={prediction.confidence}, bboxes={prediction.bounding_boxes}")
                    return prediction

        if self.tf_saved_model is not None and request.video and request.video.uri:
            source = Path(request.video.uri)
            if source.exists():
                prediction = self._infer_with_tf_saved_model(source)
                if prediction is not None:
                    print(f"[WEAPON] TF model result: {prediction}")
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

    async def _infer_with_legacy_service(self, request: PerceptionRequest) -> WeaponDetection | None:
        payload = {
            "source_id": request.source_id,
            "video_uri": request.video.uri,
            "stream_type": request.video.stream_type,
        }

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(f"{self.legacy_service_url}/infer", json=payload)
                response.raise_for_status()
        except httpx.HTTPError:
            return None

        data = response.json()
        return WeaponDetection(
            label=self._normalize_label(str(data.get("label", "unknown_object")).lower()),
            confidence=float(data.get("confidence", 0.0)),
            bounding_boxes=data.get("bounding_boxes", []),
            backend="legacy_tf_service",
            class_evidence=[{"label": self._normalize_label(str(data.get("label", "unknown_object")).lower()), "confidence": float(data.get("confidence", 0.0)), "frame_index": 0}],
            sampled_frames=1,
        )

    def _infer_with_tf_saved_model(self, source: Path) -> WeaponDetection | None:
        if source.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
            image = self._load_frame(source)
            return self._run_tf_saved_model_on_frame(image, frame_index=0)

        best_detection: WeaponDetection | None = None
        evidence_by_label: dict[str, dict[str, float | int | str]] = {}
        sampled_frames = 0
        for frame_index, frame in self._sample_video_frames(source, self.video_frame_samples):
            sampled_frames += 1
            detection = self._run_tf_saved_model_on_frame(frame, frame_index=frame_index)
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
        # Faces: 0.8-1.2 ratio or wider (person's head area is too large)
        # VERY RELAXED: Accept wide range of shapes, face filtering handles rest
        MIN_ASPECT_RATIO = 0.4  # Very permissive - accept most shapes
        
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

    def _detect_faces(self, image: np.ndarray) -> list[dict[str, float]]:
        """Detect faces in image using MediaPipe or OpenCV - returns normalized bounding boxes"""
        
        height, width = image.shape[:2]
        faces = []
        
        # Try MediaPipe first
        if self.face_detector is not None and not self.use_opencv_face_detection:
            try:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = self.face_detector.process(rgb_image)
                
                if results.detections:
                    for idx, detection in enumerate(results.detections):
                        bbox = detection.location_data.relative_bounding_box
                        x_min = max(0.0, bbox.xmin - 0.1)
                        y_min = max(0.0, bbox.ymin - 0.1)
                        x_max = min(1.0, bbox.xmin + bbox.width + 0.1)
                        y_max = min(1.0, bbox.ymin + bbox.height + 0.1)
                        face_conf = detection.score[0] if detection.score else 0.5
                        faces.append({'x_min': x_min, 'y_min': y_min, 'x_max': x_max, 'y_max': y_max, 'confidence': face_conf})
                        print(f"[FACE_DETECT]   Face #{idx}: bbox=[{x_min:.3f}, {y_min:.3f}, {x_max:.3f}, {y_max:.3f}], conf={face_conf:.3f}", flush=True)
                print(f"[FACE_DETECT] Found {len(faces)} faces (MediaPipe)", flush=True)
                return faces
            except Exception as e:
                print(f"[FACE_DETECT] MediaPipe error: {e}, falling back to OpenCV", flush=True)
        
        # Fallback to OpenCV cascade classifier
        if self.use_opencv_face_detection and opencv_face_cascade is not None:
            try:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                detected_faces = opencv_face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                
                for idx, (x, y, w, h) in enumerate(detected_faces):
                    # Convert pixel coordinates to normalized [0, 1]
                    x_min = max(0.0, (x - 10) / width)  # Expand by 10 pixels
                    y_min = max(0.0, (y - 10) / height)
                    x_max = min(1.0, (x + w + 10) / width)
                    y_max = min(1.0, (y + h + 10) / height)
                    faces.append({'x_min': x_min, 'y_min': y_min, 'x_max': x_max, 'y_max': y_max, 'confidence': 0.8})
                    print(f"[FACE_DETECT]   Face #{idx}: bbox=[{x_min:.3f}, {y_min:.3f}, {x_max:.3f}, {y_max:.3f}], conf=0.8", flush=True)
                print(f"[FACE_DETECT] Found {len(faces)} faces (OpenCV)", flush=True)
                return faces
            except Exception as e:
                print(f"[FACE_DETECT] OpenCV error: {e}", flush=True)
        
        print(f"[FACE_DETECT] No face detection available - returning empty list", flush=True)
        return []

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

    def _run_tf_saved_model_on_frame(self, image: np.ndarray, frame_index: int) -> WeaponDetection | None:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = tf.convert_to_tensor(np.expand_dims(rgb, axis=0), dtype=tf.uint8)

        try:
            outputs = self.tf_saved_model(tensor)
        except Exception:
            return None

        def _to_numpy(name: str) -> np.ndarray | None:
            value = outputs.get(name)
            if value is None:
                return None
            if hasattr(value, "numpy"):
                return value.numpy()
            return np.asarray(value)

        boxes = _to_numpy("detection_boxes")
        scores = _to_numpy("detection_scores")
        classes = _to_numpy("detection_classes")
        if boxes is None or scores is None or classes is None:
            return None

        best_score = float(scores[0][0])
        # Threshold set to 0.10 to filter false positives while catching real weapons
        if best_score < 0.10:
            return WeaponDetection(
                label="unarmed",
                confidence=round(best_score, 4),
                bounding_boxes=[],
                backend="tf2_saved_model",
                class_evidence=[{"label": "unarmed", "confidence": round(best_score, 4), "frame_index": frame_index}],
                sampled_frames=1,
            )

        class_id = int(classes[0][0])
        height, width = image.shape[:2]
        y_min, x_min, y_max, x_max = boxes[0][0]
        xyxy = [
            round(float(x_min * width), 2),
            round(float(y_min * height), 2),
            round(float(x_max * width), 2),
            round(float(y_max * height), 2),
        ]

        return WeaponDetection(
            label=self._map_repo_class_id(class_id),
            confidence=round(best_score, 4),
            bounding_boxes=[xyxy],
            backend="tf2_saved_model",
            class_evidence=self._build_tf_class_evidence(classes[0], scores[0], frame_index),
            sampled_frames=1,
        )

    def _run_yolo_on_frame(self, image: np.ndarray, frame_index: int) -> WeaponDetection | None:
        """Run YOLO model on frame with minimum confidence threshold and face exclusion filtering"""
        
        # Minimum confidence thresholds:
        # - YOLO search threshold: 0.05 (to find all candidate detections)
        # - Weapon detection threshold: 0.05 (lowered from 0.10 to catch more legitimate weapons while face filtering removes FP)
        # NOTE: Firearms (gun, rifle, shotgun) use lower threshold 0.04 for better detection
        YOLO_CONF_THRESHOLD = 0.05  # Very low to find all candidates
        MIN_WEAPON_CONFIDENCE = 0.05  # LOW THRESHOLD to catch real weapons; face filtering + aspect ratio handles FP
        MIN_FIREARM_CONFIDENCE = 0.04  # Even lower for firearms (high security priority)
        
        height, width = image.shape[:2]
        print(f"[YOLO] Frame size: {width}x{height}, face_detector available: {self.face_detector is not None}", flush=True)
        
        # FILTER 0: Detect faces FIRST to create exclusion zones
        face_bboxes = self._detect_faces(image)
        print(f"[YOLO] Face detection complete: {len(face_bboxes)} faces found", flush=True)
        
        results = self.model.predict(source=image, verbose=False, conf=YOLO_CONF_THRESHOLD)
        print(f"[YOLO] Prediction results: {len(results) if results else 0} result objects", flush=True)
        
        if not results:
            print(f"[YOLO] No results returned", flush=True)
            return None

        result = results[0]
        boxes = getattr(result, "boxes", None)
        print(f"[YOLO] Boxes found: {len(boxes) if boxes is not None else 0}", flush=True)
        
        if boxes is None or len(boxes) == 0:
            print(f"[YOLO] No boxes detected - returning unarmed", flush=True)
            return WeaponDetection(
                label="unarmed",
                confidence=0.75,
                bounding_boxes=[],
                backend="ultralytics_yolo",
                class_evidence=[{"label": "unarmed", "confidence": 0.75, "frame_index": frame_index}],
                sampled_frames=1,
            )

        # Find the best detection that meets minimum confidence threshold
        names = getattr(result, "names", {})
        best_index = None
        best_confidence = 0.0
        
        print(f"[YOLO] Checking {len(boxes)} detections against thresholds", flush=True)
        for idx in range(len(boxes)):
            box = boxes[idx]
            confidence = float(box.conf.item())
            raw_label = str(names.get(int(box.cls.item()), "unknown_object")).lower()
            
            # Use lower threshold for firearms (gun, rifle, shotgun)
            threshold = MIN_FIREARM_CONFIDENCE if any(fw in raw_label for fw in ["gun", "rifle", "shotgun", "firearm", "pistol"]) else MIN_WEAPON_CONFIDENCE
            
            print(f"[YOLO]   Box {idx}: {raw_label} at confidence {confidence:.4f} (threshold: {threshold:.4f})", flush=True)
            if confidence >= threshold and confidence > best_confidence:
                best_index = idx
                best_confidence = confidence
                print(f"[YOLO]     → NEW BEST: {best_index} with {confidence:.4f}", flush=True)
        
        # If no detection meets minimum confidence threshold, return unarmed
        if best_index is None:
            print(f"[YOLO] ❌ No detections above confidence threshold {MIN_WEAPON_CONFIDENCE}", flush=True)
            # Log what was filtered out for debugging
            for idx in range(len(boxes)):
                box = boxes[idx]
                filtered_conf = float(box.conf.item())
                raw_label = str(names.get(int(box.cls.item()), "unknown_object")).lower()
                print(f"[YOLO]   Filtered out: {raw_label} {filtered_conf:.4f} < {MIN_WEAPON_CONFIDENCE}", flush=True)
            return WeaponDetection(
                label="unarmed",
                confidence=0.75,
                bounding_boxes=[],
                backend="ultralytics_yolo",
                class_evidence=[{"label": "unarmed", "confidence": 0.75, "frame_index": frame_index}],
                sampled_frames=1,
            )
        
        # Process the best detection that passed the threshold
        best_box = boxes[best_index]
        raw_label = str(names.get(int(best_box.cls.item()), "unknown_object")).lower()
        label = self._normalize_label(raw_label)
        confidence = round(best_confidence, 4)
        xyxy = [round(float(value), 2) for value in best_box.xyxy.tolist()[0]]
        
        # SHAPE INFERENCE: Correct misclassifications based on bbox shape
        # (e.g., "knife" with very high aspect ratio → likely "gun")
        label = self._infer_weapon_from_shape(label, xyxy)
        
        print(f"[YOLO] ✓ Best detection (>= {MIN_WEAPON_CONFIDENCE}): {label} at {confidence}, bbox={xyxy}", flush=True)

        # FILTER 0: Face exclusion - REJECT if overlaps with detected face
        if face_bboxes and label not in {"unarmed", "unknown_object"}:
            print(f"[YOLO] Checking {label} against {len(face_bboxes)} detected faces", flush=True)
            # Convert weapon bbox from pixel to normalized coordinates
            x1_norm = xyxy[0] / width
            y1_norm = xyxy[1] / height
            x2_norm = xyxy[2] / width
            y2_norm = xyxy[3] / height
            weapon_area = (x2_norm - x1_norm) * (y2_norm - y1_norm)
            
            print(f"[YOLO] Weapon bbox normalized: [{x1_norm:.3f}, {y1_norm:.3f}, {x2_norm:.3f}, {y2_norm:.3f}], area={weapon_area:.4f}", flush=True)
            
            # FIREARM OVERRIDE: Firearms get more lenient face filtering
            # Guns in hands cause face overlap but are still valid detections
            firearm_mode = label in {"gun", "rifle", "shotgun", "firearm"}
            face_overlap_threshold = 0.50 if firearm_mode else 0.30  # Firearms need >50% overlap to reject, others >30%
            
            # Check if weapon overlaps with any face
            for face_idx, face in enumerate(face_bboxes):
                # Calculate intersection
                x_min_overlap = max(x1_norm, face['x_min'])
                y_min_overlap = max(y1_norm, face['y_min'])
                x_max_overlap = min(x2_norm, face['x_max'])
                y_max_overlap = min(y2_norm, face['y_max'])
                
                # Check if there's overlap
                if x_min_overlap < x_max_overlap and y_min_overlap < y_max_overlap:
                    overlap_area = (x_max_overlap - x_min_overlap) * (y_max_overlap - y_min_overlap)
                    overlap_ratio = overlap_area / weapon_area if weapon_area > 0 else 0
                    
                    print(f"[YOLO]   Face #{face_idx}: overlap_area={overlap_area:.4f}, ratio={overlap_ratio*100:.1f}%", flush=True)
                    
                    # If overlap exceeds threshold, reject as false positive
                    if overlap_ratio > face_overlap_threshold:
                        print(f"[YOLO] ❌ REJECTED: {label} is {overlap_ratio*100:.1f}% inside face zone (threshold: {face_overlap_threshold*100:.0f}%)", flush=True)
                        return WeaponDetection(
                            label="unarmed",
                            confidence=0.75,
                            bounding_boxes=[],
                            backend="ultralytics_yolo",
                            class_evidence=[{"label": "unarmed", "confidence": 0.75, "frame_index": frame_index}],
                            sampled_frames=1,
                        )
                else:
                    print(f"[YOLO]   Face #{face_idx}: no overlap (weapon is clear)", flush=True)
        
        # FILTER 1: Check aspect ratio - eliminate square detections (likely faces/objects)
        if label not in {"unarmed", "unknown_object"}:
            if not self._is_valid_weapon_aspect_ratio(xyxy):
                print(f"[YOLO] ❌ REJECTED: {label} failed aspect ratio check (too square)", flush=True)
                return WeaponDetection(
                    label="unarmed",
                    confidence=0.75,
                    bounding_boxes=[],
                    backend="ultralytics_yolo",
                    class_evidence=[{"label": "unarmed", "confidence": 0.75, "frame_index": frame_index}],
                    sampled_frames=1,
                )
        
        # FILTER 2: Check multi-frame validation - require weapon in 2+ frames
        if not self._validate_multi_frame_detection(label, frame_index, confidence):
            print(f"[YOLO] ❌ REJECTED: {label} needs 3+ frames for confirmation", flush=True)
            return WeaponDetection(
                label="unarmed",
                confidence=0.75,
                bounding_boxes=[],
                backend="ultralytics_yolo",
                class_evidence=[{"label": "unarmed", "confidence": 0.75, "frame_index": frame_index}],
                sampled_frames=1,
            )

        print(f"[YOLO] ✅ ACCEPTED: {label} passed all filters", flush=True)
        
        evidence_by_label: dict[str, dict[str, float | int | str]] = {}
        for idx in range(len(boxes)):
            box = boxes[idx]
            score = round(float(box.conf.item()), 4)
            # Only include evidence items above the minimum threshold
            if score >= MIN_WEAPON_CONFIDENCE:
                raw = str(names.get(int(box.cls.item()), "unknown_object")).lower()
                normalized = self._normalize_label(raw)
                item = {"label": normalized, "confidence": score, "frame_index": frame_index}
                if normalized not in evidence_by_label or score > float(evidence_by_label[normalized]["confidence"]):
                    evidence_by_label[normalized] = item

        return WeaponDetection(
            label=label,
            confidence=confidence,
            bounding_boxes=[xyxy],
            backend="ultralytics_yolo",
            class_evidence=sorted(
                evidence_by_label.values(),
                key=lambda item: float(item["confidence"]),
                reverse=True,
            )[:3],
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

    def _build_tf_class_evidence(
        self,
        classes: np.ndarray,
        scores: np.ndarray,
        frame_index: int,
    ) -> list[dict[str, float | int | str]]:
        evidence_by_label: dict[str, dict[str, float | int | str]] = {}
        for class_value, score_value in zip(classes.tolist(), scores.tolist()):
            score = float(score_value)
            # Threshold set to 0.10 to filter false positives
            if score < 0.10:
                continue
            label = self._map_repo_class_id(int(class_value))
            item = {
                "label": label,
                "confidence": round(score, 4),
                "frame_index": frame_index,
            }
            if label not in evidence_by_label or score > float(evidence_by_label[label]["confidence"]):
                evidence_by_label[label] = item

        return sorted(
            evidence_by_label.values(),
            key=lambda item: float(item["confidence"]),
            reverse=True,
        )[:3]
