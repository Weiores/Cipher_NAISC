"""
CipherMLModel — incremental online learning model for Cipher_NAISC.

Uses SGDClassifier(loss='log_loss') for binary threat classification.
Supports partial_fit() so each officer feedback immediately updates
the model without full retraining.

Saved to:  data/cipher_ml_model.joblib
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_MODEL_PATH = _ROOT / "data" / "cipher_ml_model.joblib"


def _load_env() -> None:
    env_path = _ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_env()

try:
    import numpy as np
    from sklearn.linear_model import SGDClassifier
    from sklearn.preprocessing import StandardScaler
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False
    logger.warning("[ML] scikit-learn not installed; CipherMLModel will be a no-op")

try:
    import joblib
    _JOBLIB_AVAILABLE = True
except ImportError:
    _JOBLIB_AVAILABLE = False
    logger.warning("[ML] joblib not installed; model persistence disabled")


class CipherMLModel:
    """Lightweight online learning model that improves from officer feedback.

    Features used (6-dimensional vector):
      0  weapon_confidence   float  0-1
      1  emotion_encoded     int    angry=2, fearful/sad/disgust=1, other=0
      2  tone_encoded        int    aggressive/threat=2, tense/suspicious=1, calm=0
      3  has_uniform         int    1=officer present, 0=civilian
      4  time_of_day_encoded int    night(0-5,22-23)=2, evening(18-21)=1, day=0
      5  weapon_detected     int    1 if label is a known weapon or conf>0.15, else 0

    Predicts:
      is_real_threat (0=false alarm, 1=confirmed)
    """

    _EMOTION_MAP: dict[str, int] = {
        "angry": 2, "anger": 2,
        "fearful": 1, "fear": 1, "disgust": 1, "sad": 1, "sadness": 1,
    }
    _TONE_MAP: dict[str, int] = {
        "aggressive": 2, "threat": 2, "threatening": 2,
        "tense": 1, "suspicious": 1, "nervous": 1,
    }
    _CLASSES = [0, 1]
    _ACTION_MAP = [
        "DISPATCH_OFFICERS",
        "INCREASE_SURVEILLANCE",
        "ISSUE_VERBAL_WARNING",
        "REVIEW_FOOTAGE",
        "MONITOR_ONLY",
    ]

    def __init__(self, model_path: str | Path | None = None) -> None:
        self._model_path = Path(model_path) if model_path else _DEFAULT_MODEL_PATH
        self._classifier: Any = None
        self._scaler: Any = None
        self._samples_seen: int = 0
        self._accuracy: float = 0.0
        self._last_updated: str | None = None
        self._is_fitted: bool = False

        if _SKLEARN_AVAILABLE:
            self._classifier = SGDClassifier(
                loss="log_loss",
                random_state=42,
                class_weight="balanced",
                max_iter=1000,
                tol=1e-3,
            )
            self._scaler = StandardScaler()
            self._load_model()
        else:
            logger.warning("[ML] sklearn unavailable — predictions will be defaults")

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    # Labels that count as "weapon detected" for the weapon_detected feature
    _WEAPON_LABELS = {"gun", "knife", "blade", "scissors", "bat", "stick", "pistol",
                      "rifle", "firearm", "shotgun", "dagger", "machete"}

    def _extract_features(self, perception: dict) -> "np.ndarray":
        # ── Flat dict format ─────────────────────────────────────────────
        # e.g. {'weapon_confidence': 0.81, 'emotion_label': 'neutral', ...}
        # Used by the verification command and perception_layer.py directly.
        if "weapon_confidence" in perception:
            weapon_conf  = float(perception.get("weapon_confidence", 0))
            em_label     = str(perception.get("emotion_label", "")).lower()
            tone_label   = str(perception.get("tone_label", "")).lower()
            has_uniform  = int(bool(perception.get("has_uniform", False)))
            tod          = self._time_of_day(perception.get("timestamp", ""))
            emotion_enc  = self._EMOTION_MAP.get(em_label, 0)
            tone_enc     = self._TONE_MAP.get(tone_label, 0)
            # weapon_detected: confidence > 0.15 treated as a real weapon present
            weapon_det   = int(weapon_conf > 0.15)
            return np.array(
                [[weapon_conf, emotion_enc, tone_enc, has_uniform, tod, weapon_det]],
                dtype=float,
            )

        # ── Nested dict format ───────────────────────────────────────────
        # e.g. {'weapon': {'confidence': ..., 'label': ...}, 'emotion': {...}, ...}
        weapon  = perception.get("weapon", {}) or {}
        emotion = perception.get("emotion", {}) or {}
        tone_d  = perception.get("tone", {}) or {}
        uniform = perception.get("uniform", {}) or {}

        weapon_conf  = float(weapon.get("confidence", 0))
        weapon_label = str(weapon.get("label", "")).lower()
        em_label     = str(emotion.get("label", "")).lower()
        emotion_enc  = self._EMOTION_MAP.get(em_label, 0)
        tone_label   = str(tone_d.get("label", tone_d.get("tone", ""))).lower()
        tone_enc     = self._TONE_MAP.get(tone_label, 0)
        has_uniform  = int(bool(uniform.get("present", False)))
        tod          = self._time_of_day(perception.get("timestamp", ""))
        weapon_det   = int(weapon_label in self._WEAPON_LABELS or weapon_conf > 0.15)

        return np.array(
            [[weapon_conf, emotion_enc, tone_enc, has_uniform, tod, weapon_det]],
            dtype=float,
        )

    def _extract_features_from_sample(self, sample: dict) -> "np.ndarray":
        feats        = sample.get("features", {})
        weapon_conf  = float(feats.get("weapon_conf", 0))
        weapon_label = str(feats.get("weapon_label", "")).lower()
        em_label     = str(feats.get("emotion_label", "")).lower()
        emotion_enc  = self._EMOTION_MAP.get(em_label, 0)
        tone_label   = str(feats.get("tone_label", "")).lower()
        tone_enc     = self._TONE_MAP.get(tone_label, 0)
        has_uniform  = int(feats.get("has_uniform", 0))
        tod          = self._time_of_day(sample.get("timestamp", ""))
        # weapon_detected: explicit label wins; fall back to confidence heuristic
        weapon_det   = int(weapon_label in self._WEAPON_LABELS or weapon_conf > 0.15)
        return np.array(
            [[weapon_conf, emotion_enc, tone_enc, has_uniform, tod, weapon_det]],
            dtype=float,
        )

    @staticmethod
    def _time_of_day(ts: str) -> int:
        try:
            hour = int(str(ts)[11:13])
            if hour < 6 or hour >= 22:
                return 2
            if hour >= 18:
                return 1
        except (ValueError, IndexError):
            pass
        return 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train_initial(self, training_data: list[dict]) -> dict[str, Any]:
        """Batch-train (or retrain) from all labelled feedback samples."""
        if not _SKLEARN_AVAILABLE or not training_data:
            return {"samples": 0, "accuracy": 0.0}

        X_list, y_list = [], []
        for sample in training_data:
            try:
                X_list.append(self._extract_features_from_sample(sample)[0])
                y_list.append(int(sample["label"]))
            except Exception:
                continue

        if not X_list:
            return {"samples": 0, "accuracy": 0.0}

        X = np.array(X_list)
        y = np.array(y_list)

        X_scaled = self._scaler.fit_transform(X)
        self._classifier.fit(X_scaled, y)

        preds = self._classifier.predict(X_scaled)
        self._samples_seen = len(X_list)
        self._accuracy = float(np.mean(preds == y))
        self._is_fitted = True
        self._last_updated = datetime.now(timezone.utc).isoformat()

        self._save_model()
        logger.info("[ML] Batch train complete: samples=%d accuracy=%.3f",
                    self._samples_seen, self._accuracy)
        return {"samples": self._samples_seen, "accuracy": self._accuracy}

    def update(self, incident: dict, feedback: dict) -> dict[str, Any]:
        """Incremental update (partial_fit) with a single new labelled sample."""
        if not _SKLEARN_AVAILABLE:
            return {"samples": self._samples_seen, "accuracy": self._accuracy}

        label = feedback.get("label", -1)
        if label not in (0, 1):
            return {"samples": self._samples_seen, "accuracy": self._accuracy}

        # Extract perception from nested detections if present
        detections = incident.get("detections") or {}
        if isinstance(detections, list) and detections:
            det = detections[0]
        elif isinstance(detections, dict):
            det = detections
        else:
            det = {}

        perception = {
            "weapon":    (det.get("weapon", {}) if isinstance(det, dict) else {}),
            "emotion":   (det.get("emotion", {}) if isinstance(det, dict) else {}),
            "tone":      (det.get("tone", {}) if isinstance(det, dict) else {}),
            "uniform":   (det.get("uniform", {}) if isinstance(det, dict) else {}),
            "timestamp": incident.get("timestamp", incident.get("created_at", "")),
        }

        X = self._extract_features(perception)

        if not self._is_fitted:
            # Bootstrap scaler on first sample
            self._scaler.fit(X)

        X_scaled = self._scaler.transform(X)
        self._classifier.partial_fit(X_scaled, [label], classes=self._CLASSES)
        self._samples_seen += 1
        self._is_fitted = True
        self._last_updated = datetime.now(timezone.utc).isoformat()
        self._save_model()

        logger.info("[ML] Incremental update: sample=%d label=%d", self._samples_seen, label)
        return {"samples": self._samples_seen, "accuracy": self._accuracy}

    def predict(self, perception_result: dict) -> dict[str, Any]:
        """Return threat probability and suggested action for a perception result."""
        default = {
            "is_threat_probability": 0.5,
            "suggested_action":      "REVIEW_FOOTAGE",
            "confidence":            0.0,
            "based_on_samples":      self._samples_seen,
        }

        if not _SKLEARN_AVAILABLE or not self._is_fitted:
            return default

        try:
            # Normalise: handle nested detections list
            perception = perception_result
            dets = perception_result.get("detections")
            if isinstance(dets, list) and dets and isinstance(dets[0], dict):
                det = dets[0]
                perception = {
                    "weapon":    det.get("weapon", {}),
                    "emotion":   det.get("emotion", {}),
                    "tone":      det.get("tone", {}),
                    "uniform":   det.get("uniform", {}),
                    "timestamp": perception_result.get("timestamp", ""),
                }

            X = self._extract_features(perception)
            X_scaled = self._scaler.transform(X)
            proba = self._classifier.predict_proba(X_scaled)[0]
            threat_prob = float(proba[1]) if len(proba) > 1 else 0.5

            if threat_prob > 0.85:
                action = "DISPATCH_OFFICERS"
            elif threat_prob > 0.65:
                action = "INCREASE_SURVEILLANCE"
            elif threat_prob > 0.45:
                action = "ISSUE_VERBAL_WARNING"
            elif threat_prob > 0.25:
                action = "REVIEW_FOOTAGE"
            else:
                action = "MONITOR_ONLY"

            confidence = min(1.0, abs(threat_prob - 0.5) * 2)

            return {
                "is_threat_probability": round(threat_prob, 4),
                "suggested_action":      action,
                "confidence":            round(confidence, 4),
                "based_on_samples":      self._samples_seen,
            }
        except Exception as exc:
            logger.error("[ML] Prediction error: %s", exc)
            return default

    def save_model(self, path: str | Path | None = None) -> None:
        self._save_model(path)

    def load_model(self, path: str | Path | None = None) -> bool:
        return self._load_model(path)

    def get_stats(self) -> dict[str, Any]:
        return {
            "is_fitted":         self._is_fitted,
            "samples_seen":      self._samples_seen,
            "accuracy":          round(self._accuracy, 4),
            "last_updated":      self._last_updated,
            "sklearn_available": _SKLEARN_AVAILABLE,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_model(self, path: str | Path | None = None) -> None:
        if not (_SKLEARN_AVAILABLE and _JOBLIB_AVAILABLE and self._is_fitted):
            return
        p = Path(path) if path else self._model_path
        p.parent.mkdir(parents=True, exist_ok=True)
        try:
            joblib.dump({
                "classifier":   self._classifier,
                "scaler":       self._scaler,
                "samples_seen": self._samples_seen,
                "accuracy":     self._accuracy,
                "last_updated": self._last_updated,
                "is_fitted":    self._is_fitted,
            }, str(p))
            logger.debug("[ML] Model saved → %s", p)
        except Exception as exc:
            logger.error("[ML] Save failed: %s", exc)

    def _load_model(self, path: str | Path | None = None) -> bool:
        if not (_SKLEARN_AVAILABLE and _JOBLIB_AVAILABLE):
            return False
        p = Path(path) if path else self._model_path
        if not p.exists():
            return False
        try:
            state = joblib.load(str(p))
            self._classifier   = state["classifier"]
            self._scaler       = state["scaler"]
            self._samples_seen = state.get("samples_seen", 0)
            self._accuracy     = state.get("accuracy", 0.0)
            self._last_updated = state.get("last_updated")
            self._is_fitted    = state.get("is_fitted", False)
            logger.info("[ML] Model loaded: samples=%d accuracy=%.3f",
                        self._samples_seen, self._accuracy)
            return True
        except Exception as exc:
            logger.error("[ML] Load failed: %s", exc)
            return False


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.DEBUG)

    m = CipherMLModel()
    print("Stats (unfitted):", m.get_stats())

    # Fake training data
    samples = [
        {"features": {"weapon_conf": 0.9, "emotion_label": "angry", "tone_label": "aggressive",
                       "has_uniform": 0, "threat_reasons": "gun spotted"}, "label": 1, "timestamp": "2024-01-01T14:00:00"},
        {"features": {"weapon_conf": 0.1, "emotion_label": "neutral", "tone_label": "calm",
                       "has_uniform": 1, "threat_reasons": "officer patrol"}, "label": 0, "timestamp": "2024-01-01T10:00:00"},
        {"features": {"weapon_conf": 0.8, "emotion_label": "fearful", "tone_label": "tense",
                       "has_uniform": 0, "threat_reasons": "knife threat"}, "label": 1, "timestamp": "2024-01-01T22:00:00"},
        {"features": {"weapon_conf": 0.05, "emotion_label": "neutral", "tone_label": "calm",
                       "has_uniform": 0, "threat_reasons": "normal activity"}, "label": 0, "timestamp": "2024-01-01T12:00:00"},
        {"features": {"weapon_conf": 0.85, "emotion_label": "angry", "tone_label": "threat",
                       "has_uniform": 0, "threat_reasons": "weapon in crowd"}, "label": 1, "timestamp": "2024-01-01T20:00:00"},
        {"features": {"weapon_conf": 0.02, "emotion_label": "neutral", "tone_label": "calm",
                       "has_uniform": 1, "threat_reasons": "routine check"}, "label": 0, "timestamp": "2024-01-01T09:00:00"},
        {"features": {"weapon_conf": 0.7, "emotion_label": "angry", "tone_label": "aggressive",
                       "has_uniform": 0, "threat_reasons": "confrontation"}, "label": 1, "timestamp": "2024-01-01T23:00:00"},
        {"features": {"weapon_conf": 0.15, "emotion_label": "neutral", "tone_label": "calm",
                       "has_uniform": 0, "threat_reasons": "false detection"}, "label": 0, "timestamp": "2024-01-01T11:00:00"},
        {"features": {"weapon_conf": 0.95, "emotion_label": "angry", "tone_label": "threat",
                       "has_uniform": 0, "threat_reasons": "armed robbery"}, "label": 1, "timestamp": "2024-01-01T02:00:00"},
        {"features": {"weapon_conf": 0.03, "emotion_label": "neutral", "tone_label": "calm",
                       "has_uniform": 1, "threat_reasons": "patrol normal"}, "label": 0, "timestamp": "2024-01-01T08:00:00"},
    ]

    result = m.train_initial(samples)
    print("After training:", result)
    print("Stats:", m.get_stats())

    pred = m.predict({"weapon": {"confidence": 0.9}, "emotion": {"label": "angry"},
                       "tone": {"label": "aggressive"}, "uniform": {"present": False}})
    print("Prediction (high threat):", pred)

    pred2 = m.predict({"weapon": {"confidence": 0.05}, "emotion": {"label": "neutral"},
                        "tone": {"label": "calm"}, "uniform": {"present": True}})
    print("Prediction (low threat):", pred2)
