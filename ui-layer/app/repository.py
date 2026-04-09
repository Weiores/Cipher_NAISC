from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone

from app.schemas import AlertFeedbackRecord, AlertRecord


class InMemoryAlertRepository:
    """Hackathon-friendly state store that can later be swapped for a database."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._alerts: dict[str, AlertRecord] = {}
        self._latest_by_dedupe_key: dict[str, str] = {}
        self._latest_by_incident_key: dict[str, str] = {}
        self._feedback: list[AlertFeedbackRecord] = []

    def save_alert(self, alert: AlertRecord) -> None:
        with self._lock:
            self._alerts[alert.alert_id] = alert
            self._latest_by_dedupe_key[alert.dedupe_key] = alert.alert_id
            self._latest_by_incident_key[alert.incident_key] = alert.alert_id

    def get_alert(self, alert_id: str) -> AlertRecord | None:
        with self._lock:
            return self._alerts.get(alert_id)

    def get_recent_incident(self, incident_key: str, window_seconds: int) -> AlertRecord | None:
        with self._lock:
            alert_id = self._latest_by_incident_key.get(incident_key)
            if not alert_id:
                return None

            record = self._alerts.get(alert_id)
            if not record:
                return None

            created_at = datetime.fromisoformat(record.created_at)
            if datetime.now(timezone.utc) - created_at <= timedelta(seconds=window_seconds):
                return record
            return None

    def get_recent_duplicate(self, dedupe_key: str, window_seconds: int) -> AlertRecord | None:
        with self._lock:
            alert_id = self._latest_by_dedupe_key.get(dedupe_key)
            if not alert_id:
                return None

            record = self._alerts.get(alert_id)
            if not record:
                return None

            created_at = datetime.fromisoformat(record.created_at)
            if datetime.now(timezone.utc) - created_at <= timedelta(seconds=window_seconds):
                return record
            return None

    def set_message_id(self, alert_id: str, message_id: int | None) -> None:
        with self._lock:
            record = self._alerts.get(alert_id)
            if record:
                record.telegram_message_id = message_id

    def update_status(self, alert_id: str, status: str) -> AlertRecord | None:
        with self._lock:
            record = self._alerts.get(alert_id)
            if record:
                record.status = status
            return record

    def add_feedback(self, feedback: AlertFeedbackRecord) -> None:
        with self._lock:
            self._feedback.append(feedback)

    def feedback_count(self) -> int:
        with self._lock:
            return len(self._feedback)

    def active_alert_count(self) -> int:
        with self._lock:
            return sum(1 for record in self._alerts.values() if record.status in {"pending", "dispatched", "escalated"})
