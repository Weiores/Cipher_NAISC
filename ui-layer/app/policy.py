from __future__ import annotations

from dataclasses import dataclass

from app.config import Settings
from app.schemas import TelegramAlertRequest


THREAT_LEVEL_ORDER = {
    "low": 0,
    "medium": 1,
    "high": 2,
    "critical": 3,
}


@dataclass(slots=True)
class PolicyDecision:
    should_send: bool
    reason: str
    dedupe_key: str


def normalize_anomaly_type(alert: TelegramAlertRequest) -> str:
    if alert.anomaly_type:
        return alert.anomaly_type.lower()
    if alert.reasoning.anomaly_types:
        return alert.reasoning.anomaly_types[0].lower()
    return "unspecified_anomaly"


def build_dedupe_key(alert: TelegramAlertRequest) -> str:
    anomaly_type = normalize_anomaly_type(alert)
    return "|".join(
        [
            alert.location.strip().lower() or "unknown",
            anomaly_type,
            alert.reasoning.threat_level.lower(),
        ]
    )


def should_route_alert(alert: TelegramAlertRequest, settings: Settings) -> PolicyDecision:
    anomaly_type = normalize_anomaly_type(alert)
    dedupe_key = build_dedupe_key(alert)
    threat_level = alert.reasoning.threat_level.lower()

    if anomaly_type in settings.always_send_anomalies:
        return PolicyDecision(True, f"always_send anomaly={anomaly_type}", dedupe_key)

    if threat_level in settings.always_send_threat_levels:
        return PolicyDecision(True, f"always_send threat_level={threat_level}", dedupe_key)

    minimum_level = THREAT_LEVEL_ORDER.get(settings.alert_min_threat_level, THREAT_LEVEL_ORDER["high"])
    current_level = THREAT_LEVEL_ORDER.get(threat_level, THREAT_LEVEL_ORDER["low"])
    if current_level >= minimum_level:
        return PolicyDecision(True, f"meets minimum threshold={settings.alert_min_threat_level}", dedupe_key)

    return PolicyDecision(False, "below routing threshold", dedupe_key)
