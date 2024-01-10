from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.schemas import PerceptionRequest, UniformDetection
from app.services.models.base import ModelAdapter

try:
    import tensorflow as tf
except ImportError:  # pragma: no cover - optional runtime dependency
    tf = None

DEFAULT_UNIFORM_GRAPH_PATH = Path(
    Path(__file__).resolve().parents[4]
    / "Gun-Uniform-Detection"
    / "frozen_inference_graphFinal.pb"
)


class UniformDetectionAdapter(ModelAdapter):
    name = "uniform_detection"
    intended_backend = "TensorFlow frozen graph from TalhaKhalil Gun-Uniform-Detection"
    video_frame_samples = 16

    def __init__(self) -> None:
        self.graph_path = Path(DEFAULT_UNIFORM_GRAPH_PATH)
        self.model = self._load_frozen_graph(self.graph_path)

    def describe(self) -> dict[str, str]:
        details = super().describe()
        details["active_backend"] = "tf_frozen_graph" if self.model is not None else "placeholder"
        details["graph_path"] = str(self.graph_path)
        return details

    async def infer(self, request: PerceptionRequest) -> UniformDetection:
        if self.model is not None and request.video and request.video.uri:
            source = Path(request.video.uri)
            if source.exists():
                prediction = self._infer_with_frozen_graph(source)
                if prediction is not None:
                    return prediction
        return self._infer_placeholder(request)

    def _load_frozen_graph(self, graph_path: Path) -> Any:
        if tf is None or not graph_path.exists():
            return None

        try:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(graph_path.read_bytes())
            graph = tf.Graph()
            with graph.as_default():
                tf.import_graph_def(graph_def, name="")
            session = tf.compat.v1.Session(graph=graph)
        except Exception:
            return None

        try:
            return {
                "session": session,
                "image_tensor": graph.get_tensor_by_name("image_tensor:0"),
                "boxes": graph.get_tensor_by_name("detection_boxes:0"),
                "scores": graph.get_tensor_by_name("detection_scores:0"),
                "classes": graph.get_tensor_by_name("detection_classes:0"),
                "num": graph.get_tensor_by_name("num_detections:0"),
            }
        except KeyError:
            session.close()
            return None

    def _infer_placeholder(self, request: PerceptionRequest) -> UniformDetection:
        context = (request.incident_context or "").lower()
        uniform_present = any(token in context for token in {"uniform", "officer", "police", "security"})
        confidence = 0.72 if uniform_present else 0.18
        evidence: list[dict[str, float | int | str]] = []
        if uniform_present:
            evidence.append({"label": "uniform", "confidence": confidence, "frame_index": 0})

        return UniformDetection(
            uniform_present=uniform_present,
            confidence=confidence,
            bounding_boxes=[],
            backend="placeholder",
            class_evidence=evidence,
            sampled_frames=1,
        )

    def _infer_with_frozen_graph(self, source: Path) -> UniformDetection | None:
        if source.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
            image = self._load_frame(source)
            return self._run_graph_on_frame(image, frame_index=0)

        best_detection: UniformDetection | None = None
        evidence_by_label: dict[str, dict[str, float | int | str]] = {}
        sampled_frames = 0

        for frame_index, frame in self._sample_video_frames(source, self.video_frame_samples):
            sampled_frames += 1
            detection = self._run_graph_on_frame(frame, frame_index=frame_index)
            if detection is None:
                continue
            for item in detection.class_evidence:
                label = str(item["label"])
                if label not in evidence_by_label or float(item["confidence"]) > float(evidence_by_label[label]["confidence"]):
                    evidence_by_label[label] = item
            if best_detection is None or detection.confidence > best_detection.confidence:
                best_detection = detection

        if best_detection is not None:
            best_detection.class_evidence = sorted(
                evidence_by_label.values(),
                key=lambda item: float(item["confidence"]),
                reverse=True,
            )[:3]
            best_detection.sampled_frames = sampled_frames
        return best_detection

    def _run_graph_on_frame(self, image: np.ndarray, frame_index: int) -> UniformDetection | None:
        session = self.model["session"]
        image_tensor = self.model["image_tensor"]
        boxes_tensor = self.model["boxes"]
        scores_tensor = self.model["scores"]
        classes_tensor = self.model["classes"]
        num_tensor = self.model["num"]

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        expanded = np.expand_dims(rgb, axis=0)

        try:
            boxes, scores, classes, _ = session.run(
                [boxes_tensor, scores_tensor, classes_tensor, num_tensor],
                feed_dict={image_tensor: expanded},
            )
        except Exception:
            return None

        class_evidence = self._build_class_evidence(image, classes[0], scores[0], boxes[0], frame_index)
        uniform_evidence = [item for item in class_evidence if str(item["label"]) == "uniform"]
        cleaned_evidence: list[dict[str, float | int | str]] = []
        for item in class_evidence:
            cleaned = dict(item)
            cleaned.pop("bounding_box", None)
            cleaned_evidence.append(cleaned)

        if not uniform_evidence:
            best_score = max((float(item["confidence"]) for item in class_evidence), default=0.0)
            return UniformDetection(
                uniform_present=False,
                confidence=round(best_score, 4),
                bounding_boxes=[],
                backend="tf_frozen_graph",
                class_evidence=cleaned_evidence,
                sampled_frames=1,
            )

        best_uniform = max(uniform_evidence, key=lambda item: float(item["confidence"]))
        matching_boxes = [
            item["bounding_box"]
            for item in class_evidence
            if str(item["label"]) == "uniform" and "bounding_box" in item
        ]

        return UniformDetection(
            uniform_present=True,
            confidence=round(float(best_uniform["confidence"]), 4),
            bounding_boxes=matching_boxes,
            backend="tf_frozen_graph",
            class_evidence=cleaned_evidence,
            sampled_frames=1,
        )

    def _build_class_evidence(
        self,
        image: np.ndarray,
        classes: np.ndarray,
        scores: np.ndarray,
        boxes: np.ndarray,
        frame_index: int,
    ) -> list[dict[str, float | int | str | list[float]]]:
        height, width = image.shape[:2]
        evidence: list[dict[str, float | int | str | list[float]]] = []

        for class_value, score_value, box_value in zip(classes.tolist(), scores.tolist(), boxes.tolist()):
            score = float(score_value)
            if score < 0.25:
                continue

            label = self._map_class_id(int(class_value))
            y_min, x_min, y_max, x_max = box_value
            xyxy = [
                round(float(x_min * width), 2),
                round(float(y_min * height), 2),
                round(float(x_max * width), 2),
                round(float(y_max * height), 2),
            ]
            evidence.append(
                {
                    "label": label,
                    "confidence": round(score, 4),
                    "frame_index": frame_index,
                    "bounding_box": xyxy,
                }
            )

        evidence_by_label: dict[str, dict[str, float | int | str | list[float]]] = {}
        for item in evidence:
            label = str(item["label"])
            if label not in evidence_by_label or float(item["confidence"]) > float(evidence_by_label[label]["confidence"]):
                evidence_by_label[label] = item

        return sorted(
            evidence_by_label.values(),
            key=lambda item: float(item["confidence"]),
            reverse=True,
        )[:3]

    def _map_class_id(self, class_id: int) -> str:
        if class_id == 1:
            return "gun"
        if class_id == 2:
            return "uniform"
        return "unknown"

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
