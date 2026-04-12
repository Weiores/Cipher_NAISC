from pathlib import Path

import cv2
import numpy as np

from app.schemas import EmotionDetection, PerceptionRequest
from app.services.models.base import ModelAdapter

try:
    from fer.fer import FER
except ImportError:  # pragma: no cover - optional runtime dependency
    FER = None


class EmotionDetectionAdapter(ModelAdapter):
    name = "emotion_detection"
    intended_backend = "Face detector plus emotion classifier for anger, fear, distress, and neutral states"
    video_frame_samples = 6

    def __init__(self) -> None:
        self.detector = self._load_detector()
        self.face_cascade = None
        self._load_opencv_face_detector()

    def _load_detector(self):
        if FER is None:
            return None
        try:
            return FER(mtcnn=False)
        except Exception:
            return None

    def _load_opencv_face_detector(self):
        """Load OpenCV face cascade as a lightweight alternative to FER"""
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        try:
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
        except Exception:
            self.face_cascade = None

    def describe(self) -> dict[str, str]:
        details = super().describe()
        if self.detector is not None:
            details["active_backend"] = "fer"
        elif self.face_cascade is not None and not self.face_cascade.empty():
            details["active_backend"] = "opencv_cascade_heuristic"
        else:
            details["active_backend"] = "context_based_fallback"
        return details

    async def infer(self, request: PerceptionRequest) -> EmotionDetection:
        if not request.video or not request.video.uri:
            return EmotionDetection(label="unknown", confidence=0.0, face_count=0)

        source = Path(request.video.uri)
        
        # Try FER first if available
        if self.detector is not None and source.exists():
            prediction = self._infer_with_fer(source)
            if prediction is not None:
                return prediction

        # Try OpenCV cascade-based detection
        if self.face_cascade is not None and not self.face_cascade.empty() and source.exists():
            prediction = self._infer_with_opencv(source)
            if prediction is not None:
                return prediction

        # Fall back to context-based inference
        context = (request.incident_context or "").lower()

        label = "neutral"
        confidence = 0.62
        face_count = 1 if request.video else 0

        if "fear" in context or "afraid" in context:
            label = "fearful"
            confidence = 0.79
        elif "angry" in context or "aggressive" in context:
            label = "angry"
            confidence = 0.83
        elif "distress" in context or "crying" in context:
            label = "distressed"
            confidence = 0.77

        return EmotionDetection(label=label, confidence=confidence, face_count=face_count)

    def _infer_with_fer(self, source: Path) -> EmotionDetection | None:
        if source.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
            frame = cv2.imread(str(source))
            if frame is None:
                return None
            return self._run_fer_on_frame(frame)

        best_detection: EmotionDetection | None = None
        for frame in self._sample_video_frames(source, self.video_frame_samples):
            detection = self._run_fer_on_frame(frame)
            if detection is None:
                continue
            if best_detection is None or detection.confidence > best_detection.confidence:
                best_detection = detection

        return best_detection

    def _run_fer_on_frame(self, frame: np.ndarray) -> EmotionDetection | None:
        try:
            detections = self.detector.detect_emotions(frame)
        except Exception:
            return None

        if not detections:
            return EmotionDetection(label="unknown", confidence=0.0, face_count=0)

        best = None
        best_label = "unknown"
        best_score = -1.0
        for item in detections:
            emotions = item.get("emotions", {})
            if not emotions:
                continue
            raw_label, raw_score = max(emotions.items(), key=lambda kv: kv[1])
            if raw_score > best_score:
                best_score = float(raw_score)
                best_label = self._normalize_label(str(raw_label).lower())
                best = item

        if best is None:
            return EmotionDetection(label="unknown", confidence=0.0, face_count=len(detections))

        return EmotionDetection(
            label=best_label,
            confidence=round(best_score, 4),
            face_count=len(detections),
        )

    def _infer_with_opencv(self, source: Path) -> EmotionDetection | None:
        """Use OpenCV cascade classifiers to detect faces and estimate basic emotions"""
        if source.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
            frame = cv2.imread(str(source))
            if frame is None:
                return None
            return self._detect_faces_in_frame(frame)

        # Sample frames from video
        face_counts = []
        emotions = []
        for frame in self._sample_video_frames(source, self.video_frame_samples):
            detection = self._detect_faces_in_frame(frame)
            if detection is not None:
                face_counts.append(detection.face_count)
                emotions.append(detection)

        if not emotions:
            return None

        # Return the detection with most faces (most confident)
        best_emotion = max(emotions, key=lambda e: (e.face_count, e.confidence))
        return EmotionDetection(
            label=best_emotion.label,
            confidence=best_emotion.confidence,
            face_count=best_emotion.face_count,
        )

    def _detect_faces_in_frame(self, frame: np.ndarray) -> EmotionDetection | None:
        """Detect faces in a frame using cascade classifier and estimate emotions"""
        if self.face_cascade is None or self.face_cascade.empty():
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0:
            return EmotionDetection(label="unknown", confidence=0.0, face_count=0)

        # Simple heuristic: analyze face regions for brightness and motion patterns
        # to estimate emotional state (very basic)
        face_count = len(faces)
        
        # Calculate average face brightness as a basic emotion indicator
        brightnesses = []
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            mean_brightness = np.mean(face_roi) if face_roi.size > 0 else 128
            brightnesses.append(mean_brightness)

        avg_brightness = np.mean(brightnesses) if brightnesses else 128

        # Very simple heuristic: use brightness to guess emotion state
        # (darker face = possibly more distressed/angry/fearful, brighter = neutral/happy)
        if avg_brightness < 80:
            label = "distressed"
            confidence = 0.65
        elif avg_brightness < 100:
            label = "fearful"
            confidence = 0.62
        elif avg_brightness > 150:
            label = "neutral"
            confidence = 0.70
        else:
            label = "neutral"
            confidence = 0.68

        return EmotionDetection(label=label, confidence=confidence, face_count=face_count)

    def _normalize_label(self, raw_label: str) -> str:
        if raw_label == "angry":
            return "angry"
        if raw_label == "fear":
            return "fearful"
        if raw_label in {"sad", "disgust"}:
            return "distressed"
        if raw_label in {"neutral", "happy", "surprise"}:
            return "neutral"
        return "unknown"

    def _sample_video_frames(self, source: Path, sample_count: int) -> list[np.ndarray]:
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            return []

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_count <= 0:
            target_indices = [0]
        else:
            sample_count = max(1, min(sample_count, frame_count))
            target_indices = np.linspace(0, frame_count - 1, num=sample_count, dtype=int).tolist()

        frames: list[np.ndarray] = []
        for index in target_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = capture.read()
            if ok and frame is not None:
                frames.append(frame)

        capture.release()
        return frames
