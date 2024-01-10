from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.reasoning_import import ensure_reasoning_path

ensure_reasoning_path()

from schemas import ReasoningOutput  # type: ignore  # noqa: E402


class TelegramAlertRequest(BaseModel):
    reasoning: ReasoningOutput
    location: str = "Unknown"
    anomaly_type: str | None = None
    top_scenario: str | None = None
    source: str | None = None
    external_event_id: str | None = None
    video_path: str | None = None
    video_caption: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "TelegramAlertRequest":
        if "reasoning" in payload:
            return cls.model_validate(payload)

        reasoning = ReasoningOutput.model_validate(payload)
        metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}

        return cls(
            reasoning=reasoning,
            location=str(payload.get("location") or payload.get("source_id") or "Unknown"),
            anomaly_type=payload.get("anomaly_type"),
            top_scenario=payload.get("top_scenario"),
            source=payload.get("source"),
            external_event_id=payload.get("external_event_id"),
            video_path=payload.get("video_path"),
            video_caption=payload.get("video_caption"),
            metadata=metadata,
        )


class AlertResponse(BaseModel):
    status: Literal["sent", "suppressed", "demo_logged"]
    alert_id: str
    dedupe_key: str
    detail: str
    telegram_enabled: bool


class AlertFeedbackRecord(BaseModel):
    alert_id: str
    action: Literal["acknowledge", "dispatch", "escalate", "false_alarm", "why"]
    callback_id: str | None = None
    actor_id: int | None = None
    actor_name: str | None = None
    timestamp: str


class AlertRecord(BaseModel):
    alert_id: str
    incident_id: str
    incident_key: str
    dedupe_key: str
    location: str
    anomaly_type: str
    threat_level: str
    message_html: str
    why_summary: str
    top_scenario: str
    source_id: str
    created_at: str
    external_event_id: str | None = None
    telegram_message_id: int | None = None
    delivery_chat_id: str | None = None
    status: str = "pending"


class TelegramHealth(BaseModel):
    status: str
    demo_mode: bool
    telegram_enabled: bool
    active_alerts: int
    feedback_events: int
