from __future__ import annotations

from html import escape
from datetime import datetime
from urllib.parse import urlencode

from app.policy import normalize_anomaly_type
from app.schemas import TelegramAlertRequest


def format_confidence(score: float) -> str:
    return f"{score * 100:.1f}%"


def build_why_summary(alert: TelegramAlertRequest) -> str:
    explanation = alert.reasoning.explanation
    factors = ", ".join(explanation.key_factors[:3]) if explanation.key_factors else "No key factors provided"
    anomalies = ", ".join(alert.reasoning.anomaly_types[:3]) if alert.reasoning.anomaly_types else "No anomaly labels"
    return (
        f"{explanation.summary} Key factors: {factors}. "
        f"Detected anomalies: {anomalies}. "
        f"Confidence rationale: {explanation.confidence_reasoning}"
    )


def format_why_followup(alert_id: str, summary: str) -> str:
    return f"<b>Why Alert {escape(alert_id)}?</b>\n{escape(summary)}"


def format_escalation_message(
    incident_id: str,
    alert_id: str,
    location: str,
    anomaly_type: str,
    threat_level: str,
    top_scenario: str,
) -> str:
    return (
        f"<b>Escalation:</b> Incident {escape(incident_id)} remains unresolved.\n"
        f"<b>Alert ID:</b> {escape(alert_id)}\n"
        f"<b>Location:</b> {escape(location)}\n"
        f"<b>Anomaly Type:</b> {escape(anomaly_type)}\n"
        f"<b>Severity:</b> {escape(threat_level.upper())}\n"
        f"<b>Scenario:</b> {escape(top_scenario)}"
    )


def format_alert_message(alert: TelegramAlertRequest, incident_id: str) -> str:
    reasoning = alert.reasoning
    anomaly_type = normalize_anomaly_type(alert)
    recommendation = reasoning.recommended_action
    top_scenario = alert.top_scenario or "Not provided"
    current_timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")

    return "\n\n".join(
        [
            f"<b>Incident ID:</b> {escape(incident_id)}",
            f"<b>Severity:</b> {escape(reasoning.threat_level.upper())}",
            f"📍 <b>Location:</b> {escape(alert.location)}",
            f"<b>Anomaly Type:</b> {escape(anomaly_type)}",
            f"🛡️ <b>Recommended Response:</b> {escape(recommendation.action)}",
            f"📊 <b>Confidence:</b> {escape(format_confidence(reasoning.confidence))}",
            f"<b>Rationale:</b> {escape(reasoning.explanation.summary)}",
            f"<b>Top Scenario:</b> {escape(top_scenario)}",
            f"⏰ <b>Timestamp:</b> {escape(current_timestamp)}",
        ]
    )


def format_private_action_message(message_html: str) -> str:
    return (
        "<b>Operator Actions</b>\n"
        "This alert came from the channel. Use the buttons below to update the incident.\n\n"
        f"{message_html}"
    )


def format_incident_update_message(
    incident_id: str,
    previous_threat_level: str,
    new_threat_level: str,
    recommendation: str,
    summary: str,
) -> str:
    return "\n".join(
        [
            f"<b>Incident Update:</b> {escape(incident_id)}",
            f"<b>Threat Change:</b> {escape(previous_threat_level.upper())} -> {escape(new_threat_level.upper())}",
            f"<b>Recommended Response:</b> {escape(recommendation)}",
            f"<b>Summary:</b> {escape(summary)}",
        ]
    )


def build_console_url(
    base_url: str | None,
    incident_id: str,
    alert_id: str,
    source_id: str,
    location: str,
    anomaly_type: str,
) -> str | None:
    if not base_url:
        return None

    query = urlencode(
        {
            "incident_id": incident_id,
            "alert_id": alert_id,
            "source_id": source_id,
            "location": location,
            "anomaly_type": anomaly_type,
        }
    )
    separator = "&" if "?" in base_url else "?"
    return f"{base_url}{separator}{query}"


def format_video_caption(incident_id: str, location: str, anomaly_type: str) -> str:
    return "\n".join(
        [
            f"🎥 <b>Incident Clip:</b> {escape(incident_id)}",
            f"📍 <b>Location:</b> {escape(location)}",
            f"🧩 <b>Anomaly Type:</b> {escape(anomaly_type)}",
        ]
    )


def format_clip_omitted_message(incident_id: str, reason: str) -> str:
    return "\n".join(
        [
            f"📎 <b>Clip Omitted:</b> {escape(incident_id)}",
            f"ℹ️ <b>Reason:</b> {escape(reason)}",
        ]
    )