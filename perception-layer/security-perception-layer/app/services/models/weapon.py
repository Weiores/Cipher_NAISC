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
    import tensorflow as tf
except ImportError:  # pragma: no cover - optional runtime dependency
    tf = None

DEFAULT_YOLO_WEAPON_MODEL = Path(
    Path(__file__).resolve().parents[4]
    / "Weapons-and-Knives-Detector-with-YOLOv8"
    / "runs"
    / "detect"
    / "Normal_Compressed"
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
        self.model_path = os.getenv("WEAPON_MODEL_PATH") or str(DEFAULT_YOLO_WEAPON_MODEL)
        self.model = self._load_model(self.model_path)

    def _load_model(self, model_path: str | None) -> Any:
        if not model_path or YOLO is None:
            return None

        model_file = Path(model_path)
        if not model_file.exists():
            return None

        return YOLO(str(model_file))

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
        if self.legacy_service_url and request.video and request.video.uri:
            prediction = await self._infer_with_legacy_service(request)
            if prediction is not None:
                return prediction

        if self.model is not None and request.video and request.video.uri:
            source = Path(request.video.uri)
            if source.exists():
                prediction = self._infer_with_yolo(source)
                if prediction is not None:
                    return prediction

        if self.tf_saved_model is not None and request.video and request.video.uri:
            source = Path(request.video.uri)
            if source.exists():
                prediction = self._infer_with_tf_saved_model(source)
                if prediction is not None:
                    return prediction

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
        if best_score < 0.25:
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
        results = self.model.predict(source=image, verbose=False, conf=0.25)
        if not results:
            return None

        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return WeaponDetection(
                label="unarmed",
                confidence=0.75,
                bounding_boxes=[],
                backend="ultralytics_yolo",
                class_evidence=[{"label": "unarmed", "confidence": 0.75, "frame_index": frame_index}],
                sampled_frames=1,
            )

        names = getattr(result, "names", {})
        best_index = int(boxes.conf.argmax().item())
        best_box = boxes[best_index]
        raw_label = str(names.get(int(best_box.cls.item()), "unknown_object")).lower()
        label = self._normalize_label(raw_label)
        confidence = round(float(best_box.conf.item()), 4)
        xyxy = [round(float(value), 2) for value in best_box.xyxy.tolist()[0]]

        evidence_by_label: dict[str, dict[str, float | int | str]] = {}
        for idx in range(len(boxes)):
            box = boxes[idx]
            raw = str(names.get(int(box.cls.item()), "unknown_object")).lower()
            normalized = self._normalize_label(raw)
            score = round(float(box.conf.item()), 4)
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
            if score < 0.25:
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
