from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


class VideoInput(BaseModel):
    stream_type: Literal["cctv", "bodycam", "uploaded_video"] = "cctv"
    uri: str | None = None
    frame_sample_fps: float = Field(default=2.0, ge=0.1, le=30.0)
    camera_id: str | None = None


class AudioInput(BaseModel):
    source_type: Literal["microphone", "bodycam_mic", "stream_audio"] = "microphone"
    uri: str | None = None
    sample_rate_hz: int = Field(default=16000, ge=8000, le=96000)
    channel_count: int = Field(default=1, ge=1, le=8)


class SensorInput(BaseModel):
    sensor_type: str
    value: str | float | int | bool


class PerceptionRequest(BaseModel):
    source_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    video: VideoInput | None = None
    audio: AudioInput | None = None
    sensors: list[SensorInput] = Field(default_factory=list)
    incident_context: str | None = None


class WeaponDetection(BaseModel):
    label: Literal["gun", "knife", "bat", "stick", "unarmed", "unknown_object"]
    confidence: float = Field(ge=0.0, le=1.0)
    bounding_boxes: list[list[float]] = Field(default_factory=list)
    backend: str = "placeholder"
    class_evidence: list[dict[str, float | int | str]] = Field(default_factory=list)
    sampled_frames: int = 0


class EmotionDetection(BaseModel):
    label: Literal["angry", "fearful", "distressed", "neutral", "unknown"]
    confidence: float = Field(ge=0.0, le=1.0)
    face_count: int = Field(ge=0)


class AudioDetection(BaseModel):
    tone: Literal["calm", "panic", "threat", "abnormal", "unknown"]
    confidence: float = Field(ge=0.0, le=1.0)
    speech_present: bool
    acoustic_events: list[str] = Field(default_factory=list)
    transcript: str | None = None
    keyword_flags: list[str] = Field(default_factory=list)


class UniformDetection(BaseModel):
    uniform_present: bool
    confidence: float = Field(ge=0.0, le=1.0)
    bounding_boxes: list[list[float]] = Field(default_factory=list)
    backend: str = "placeholder"
    class_evidence: list[dict[str, float | int | str]] = Field(default_factory=list)
    sampled_frames: int = 0


class FusedEvent(BaseModel):
    source_id: str
    timestamp: datetime
    weapon_detected: str
    emotion: str
    tone: str
    confidence_scores: dict[str, float]
    risk_hints: list[str]
    rationale_summary: str
    suppression_reason: str | None = None


class UnifiedTraits(BaseModel):
    weapon_detected: str
    raw_weapon_detected: str
    weapon_class_evidence: list[dict[str, float | int | str]] = Field(default_factory=list)
    visual_secondary_evidence: list[dict[str, float | int | str]] = Field(default_factory=list)
    uniform_present: bool
    uniform_confidence: float = Field(ge=0.0, le=1.0)
    uniform_evidence: list[dict[str, float | int | str]] = Field(default_factory=list)
    weapon_suppressed_due_to_uniform: bool = False
    suppression_reason: str | None = None
    emotion: str
    tone: str
    speech_present: bool
    acoustic_events: list[str] = Field(default_factory=list)
    transcript: str | None = None
    keyword_flags: list[str] = Field(default_factory=list)


class UnifiedDecision(BaseModel):
    threat_level: Literal["low", "medium", "high", "critical"]
    anomaly_type: list[str] = Field(default_factory=list)
    recommended_response: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    rationale_summary: str


class UnifiedPerceptionOutput(BaseModel):
    source_id: str
    timestamp: datetime
    model_backends: dict[str, str]
    confidence_scores: dict[str, float]
    traits: UnifiedTraits
    risk_hints: list[str]
    decision: UnifiedDecision


class PerceptionResponse(BaseModel):
    weapon_model_output: WeaponDetection
    uniform_model_output: UniformDetection
    emotion_model_output: EmotionDetection
    audio_model_output: AudioDetection
    fused_event: FusedEvent
    unified_output: UnifiedPerceptionOutput


class HealthResponse(BaseModel):
    status: str
