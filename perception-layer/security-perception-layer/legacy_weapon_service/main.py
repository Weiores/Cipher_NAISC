import os
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

try:
    import tensorflow as tf
except ImportError:  # pragma: no cover - legacy runtime dependency
    tf = None


app = FastAPI(title="Legacy Weapon Service", version="0.1.0")

CLASS_MAP = {
    1: "gun",
    2: "gun",
    3: "gun",
    4: "gun",
    5: "gun",
    6: "knife",
}


class InferRequest(BaseModel):
    source_id: str
    video_uri: str
    stream_type: str = "cctv"


class TensorFlowWeaponModel:
    def __init__(self) -> None:
        self.model_dir = Path(os.getenv("LEGACY_WEAPON_MODEL_DIR", "")).expanduser()
        self.graph_path = self.model_dir / "frozen_inference_graph.pb"
        self.graph = None
        self.session = None
        self.image_tensor = None
        self.boxes_tensor = None
        self.scores_tensor = None
        self.classes_tensor = None
        self.num_detections_tensor = None

        if tf is not None and self.graph_path.exists():
            self._load_graph()

    def _load_graph(self) -> None:
        self.graph = tf.Graph()
        with self.graph.as_default():
            graph_def = tf.GraphDef()
            with tf.gfile.GFile(str(self.graph_path), "rb") as handle:
                graph_def.ParseFromString(handle.read())
                tf.import_graph_def(graph_def, name="")

            self.session = tf.Session(graph=self.graph)
            self.image_tensor = self.graph.get_tensor_by_name("image_tensor:0")
            self.boxes_tensor = self.graph.get_tensor_by_name("detection_boxes:0")
            self.scores_tensor = self.graph.get_tensor_by_name("detection_scores:0")
            self.classes_tensor = self.graph.get_tensor_by_name("detection_classes:0")
            self.num_detections_tensor = self.graph.get_tensor_by_name("num_detections:0")

    def describe(self) -> dict:
        return {
            "model_dir": str(self.model_dir),
            "graph_exists": self.graph_path.exists(),
            "tensorflow_available": tf is not None,
            "loaded": self.session is not None,
        }

    def infer(self, video_uri: str) -> dict:
        if self.session is None:
            raise RuntimeError("Legacy TensorFlow graph is not loaded")

        image = self._load_frame(Path(video_uri))
        height, width = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        batch = np.expand_dims(rgb, axis=0)

        boxes, scores, classes, _ = self.session.run(
            [
                self.boxes_tensor,
                self.scores_tensor,
                self.classes_tensor,
                self.num_detections_tensor,
            ],
            feed_dict={self.image_tensor: batch},
        )

        score = float(scores[0][0])
        class_id = int(classes[0][0])

        if score < 0.25:
            return {
                "label": "unarmed",
                "confidence": round(score, 4),
                "bounding_boxes": [],
            }

        y_min, x_min, y_max, x_max = boxes[0][0]
        xyxy = [
            round(float(x_min * width), 2),
            round(float(y_min * height), 2),
            round(float(x_max * width), 2),
            round(float(y_max * height), 2),
        ]

        return {
            "label": CLASS_MAP.get(class_id, "unknown_object"),
            "confidence": round(score, 4),
            "bounding_boxes": [xyxy],
        }

    def _load_frame(self, source: Path) -> np.ndarray:
        if not source.exists():
            raise FileNotFoundError(str(source))

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


weapon_model = TensorFlowWeaponModel()


@app.get("/health")
def health() -> dict:
    details = weapon_model.describe()
    return {"status": "ok" if details["loaded"] else "degraded", "details": details}


@app.post("/infer")
def infer(request: InferRequest) -> dict:
    try:
        result = weapon_model.infer(request.video_uri)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    result["source_id"] = request.source_id
    return result
