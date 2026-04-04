from datetime import datetime, timezone
from typing import Literal
from pydantic import BaseModel, Field


class RecommendedAction(BaseModel):
    """Recommended action based on threat assessment"""
    action: Literal["immediate_alert", "escalate", "elevated_monitoring", "monitor", "de_escalate", "all_clear"]
    priority: Literal["critical", "high", "medium", "low"]
    reason: str
    confidence: float = Field(ge=0.0, le=1.0)


class ThreatMetrics(BaseModel):
    """Detailed metrics for dashboard visualization"""
    weapon_threat_score: float = Field(ge=0.0, le=1.0)
    emotion_threat_score: float = Field(ge=0.0, le=1.0)
    audio_threat_score: float = Field(ge=0.0, le=1.0)
    behavioral_anomaly_score: float = Field(ge=0.0, le=1.0)
    combined_threat_score: float = Field(ge=0.0, le=1.0)
    trend: Literal["escalating", "stable", "de_escalating"]
    frames_in_history: int
    context_anomaly_flag: bool = False


class ReasoningExplanation(BaseModel):
    """Human-readable reasoning and justification"""
    summary: str
    key_factors: list[str] = Field(default_factory=list)
    evidence: dict[str, list[str]] = Field(default_factory=dict)  # e.g., {"weapon": [...], "audio": [...]}
    anomalies_detected: list[str] = Field(default_factory=list)
    temporal_analysis: str | None = None  # e.g., "Threat escalating over last 10 frames"
    confidence_reasoning: str  # Why we have this confidence level


class ReasoningOutput(BaseModel):
    """Complete reasoning layer output"""
    source_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    threat_level: Literal["low", "medium", "high", "critical"]
    confidence: float = Field(ge=0.0, le=1.0)
    recommended_action: RecommendedAction
    explanation: ReasoningExplanation
    anomaly_types: list[str] = Field(default_factory=list)
    metrics: ThreatMetrics
    reasoning_version: str = "1.0"


class ContextSnapshot(BaseModel):
    """Context information for anomaly detection"""
    baseline_threat_level: Literal["low", "medium", "high", "critical"]
    typical_weapon_detections: int
    typical_emotion_distribution: dict[str, float]  # e.g., {"neutral": 0.8, "calm": 0.2}
    location_type: str | None = "unknown"
    time_of_day_category: str = "unknown"
