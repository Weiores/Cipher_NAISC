from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from gtts import gTTS

from app.config import Settings
from app.formatter import (
    build_console_url,
    build_why_summary,
    format_clip_omitted_message,
    format_alert_message,
    format_escalation_message,
    format_incident_update_message,
    format_video_caption,
)
from app.policy import build_dedupe_key, normalize_anomaly_type, should_route_alert
from app.repository import InMemoryAlertRepository
from app.schemas import AlertRecord, AlertResponse, TelegramAlertRequest
from app.telegram_client import TelegramClient, TelegramDeliveryError


logger = logging.getLogger(__name__)


class TelegramAlertService:
    def __init__(
        self,
        settings: Settings,
        repository: InMemoryAlertRepository,
        telegram_client: TelegramClient,
    ) -> None:
        self.settings = settings
        self.repository = repository
        self.telegram_client = telegram_client

    async def send_alert(self, alert: TelegramAlertRequest) -> AlertResponse:
        policy = should_route_alert(alert, self.settings)
        incident_key = self._build_incident_key(alert)
        related_incident = self.repository.get_recent_incident(incident_key, self.settings.dedupe_window_seconds)
        if not policy.should_send:
            return AlertResponse(
                status="suppressed",
                alert_id=f"suppressed-{uuid4().hex[:8]}",
                dedupe_key=policy.dedupe_key,
                detail=policy.reason,
                telegram_enabled=self.settings.telegram_enabled,
            )

        duplicate = self.repository.get_recent_duplicate(policy.dedupe_key, self.settings.dedupe_window_seconds)
        if duplicate:
            return AlertResponse(
                status="suppressed",
                alert_id=duplicate.alert_id,
                dedupe_key=policy.dedupe_key,
                detail=f"duplicate within {self.settings.dedupe_window_seconds}s window",
                telegram_enabled=self.settings.telegram_enabled,
            )

        alert_id = uuid4().hex
        incident_id = related_incident.incident_id if related_incident else self._build_incident_id(alert_id)
        message_html = format_alert_message(alert, incident_id)
        why_summary = build_why_summary(alert)
        top_scenario = alert.top_scenario or "Not provided"
        anomaly_type = normalize_anomaly_type(alert)
        now = datetime.now(timezone.utc).isoformat()
        is_update = self._should_send_update(alert, related_incident)

        record = AlertRecord(
            alert_id=alert_id,
            incident_id=incident_id,
            incident_key=incident_key,
            dedupe_key=build_dedupe_key(alert),
            location=alert.location,
            anomaly_type=anomaly_type,
            threat_level=alert.reasoning.threat_level,
            message_html=message_html,
            why_summary=why_summary,
            top_scenario=top_scenario,
            source_id=alert.reasoning.source_id,
            created_at=now,
            external_event_id=alert.external_event_id,
            delivery_chat_id=self.settings.telegram_chat_id,
        )
        self.repository.save_alert(record)

        try:
            if is_update and related_incident and related_incident.telegram_message_id:
                update_html = format_incident_update_message(
                    incident_id=incident_id,
                    previous_threat_level=related_incident.threat_level,
                    new_threat_level=alert.reasoning.threat_level,
                    recommendation=alert.reasoning.recommended_action.action,
                    summary=alert.reasoning.explanation.summary,
                )
                message_id = await asyncio.to_thread(
                    self.telegram_client.send_alert_message,
                    update_html,
                    {"inline_keyboard": []},
                    related_incident.telegram_message_id,
                )
                delivery_detail = f"incident update posted for {incident_id}"
            else:
                message_id = await asyncio.to_thread(
                    self.telegram_client.send_alert_message,
                    message_html,
                    self._build_broadcast_keyboard(record),
                )
                delivery_detail = policy.reason
            self.repository.set_message_id(alert_id, message_id)
            status = "sent" if self.settings.telegram_enabled else "demo_logged"
        except TelegramDeliveryError as exc:
            logger.warning("[TELEGRAM][FALLBACK] Live delivery failed for alert %s: %s", alert_id, exc)
            self.repository.update_status(alert_id, "delivery_failed")
            logger.warning("[TELEGRAM][DEMO] %s", message_html.replace("\n", " | "))
            delivery_detail = f"telegram delivery failed; logged locally instead: {exc}"
            status = "demo_logged"

        if self.settings.escalation_enabled:
            asyncio.create_task(self._schedule_escalation(alert_id))

        if record.telegram_message_id and alert.video_path:
            await self._send_incident_video(record, alert.video_path, alert.video_caption)

        if self.settings.tts_enabled and record.telegram_message_id:
            await self._send_alert_tts(record, alert.reasoning.explanation.summary)

        return AlertResponse(
            status=status,
            alert_id=alert_id,
            dedupe_key=policy.dedupe_key,
            detail=delivery_detail if self.settings.telegram_enabled else "telegram credentials missing; logged locally",
            telegram_enabled=self.settings.telegram_enabled,
        )

    async def _schedule_escalation(self, alert_id: str) -> None:
        await asyncio.sleep(self.settings.escalation_timeout_seconds)
        record = self.repository.get_alert(alert_id)
        if not record or record.status != "pending":
            return

        self.repository.update_status(alert_id, "escalated")
        logger.warning("[TELEGRAM] Escalating unacknowledged alert %s", alert_id)
        try:
            await asyncio.to_thread(
                self.telegram_client.send_alert_message,
                self._build_escalation_message(record),
                {"inline_keyboard": []},
                record.telegram_message_id,
            )
        except TelegramDeliveryError as exc:
            logger.warning("[TELEGRAM][FALLBACK] Timed escalation follow-up failed: %s", exc)

    def _build_broadcast_keyboard(self, record: AlertRecord) -> dict:
        rows: list[list[dict[str, str]]] = []
        console_url = build_console_url(
            base_url=self.settings.operator_console_url,
            incident_id=record.incident_id,
            alert_id=record.alert_id,
            source_id=record.source_id,
            location=record.location,
            anomaly_type=record.anomaly_type,
        )
        if console_url:
            rows.append([{"text": "Open Console", "url": console_url}])
        if not rows:
            return {"inline_keyboard": []}
        return {"inline_keyboard": rows}

    def _build_escalation_message(self, record: AlertRecord) -> str:
        return format_escalation_message(
            incident_id=record.incident_id,
            alert_id=record.alert_id,
            location=record.location,
            anomaly_type=record.anomaly_type,
            threat_level=record.threat_level,
            top_scenario=record.top_scenario,
        )

    def _build_incident_key(self, alert: TelegramAlertRequest) -> str:
        anomaly_type = normalize_anomaly_type(alert)
        return "|".join([alert.location.strip().lower() or "unknown", anomaly_type])

    def _build_incident_id(self, alert_id: str) -> str:
        return f"INC-{alert_id[:6].upper()}"

    def _should_send_update(self, alert: TelegramAlertRequest, related_incident: AlertRecord | None) -> bool:
        if not related_incident:
            return False

        action = alert.reasoning.recommended_action.action
        if action in {"de_escalate", "all_clear"}:
            return True

        return alert.reasoning.threat_level != related_incident.threat_level

    async def _send_incident_video(self, record: AlertRecord, video_path: str, video_caption: str | None) -> None:
        resolved_path = self._resolve_video_path(video_path)
        if not resolved_path:
            logger.warning("[TELEGRAM][VIDEO] Skipping missing or unsafe clip path: %s", video_path)
            await self._send_clip_omitted_note(record, "Clip file missing or outside allowed repo paths.")
            return
        if resolved_path.stat().st_size > self.settings.telegram_max_video_bytes:
            logger.warning(
                "[TELEGRAM][VIDEO] Skipping oversized clip for alert %s: %s bytes exceeds %s",
                record.alert_id,
                resolved_path.stat().st_size,
                self.settings.telegram_max_video_bytes,
            )
            await self._send_clip_omitted_note(
                record,
                f"Clip too large for Telegram upload ({resolved_path.stat().st_size} bytes).",
            )
            return

        caption_html = video_caption or format_video_caption(
            incident_id=record.incident_id,
            location=record.location,
            anomaly_type=record.anomaly_type,
        )

        try:
            await asyncio.to_thread(
                self.telegram_client.send_video,
                record.delivery_chat_id,
                str(resolved_path),
                caption_html,
                record.telegram_message_id,
            )
        except TelegramDeliveryError as exc:
            logger.warning("[TELEGRAM][VIDEO] Video delivery failed for alert %s: %s", record.alert_id, exc)
            await self._send_clip_omitted_note(record, "Clip upload failed; alert sent without video.")

    def _resolve_video_path(self, video_path: str) -> Path | None:
        candidate = Path(video_path)
        if not candidate.is_absolute():
            candidate = (Path(__file__).resolve().parents[2] / video_path).resolve()
        else:
            candidate = candidate.resolve()

        repo_root = Path(__file__).resolve().parents[2]
        try:
            candidate.relative_to(repo_root)
        except ValueError:
            return None

        if not candidate.exists() or not candidate.is_file():
            return None
        return candidate

    async def _send_clip_omitted_note(self, record: AlertRecord, reason: str) -> None:
        if not record.telegram_message_id:
            return
        try:
            await asyncio.to_thread(
                self.telegram_client.send_alert_message,
                format_clip_omitted_message(record.incident_id, reason),
                {"inline_keyboard": []},
                record.telegram_message_id,
            )
        except TelegramDeliveryError as exc:
            logger.warning("[TELEGRAM][VIDEO] Clip omission note failed for alert %s: %s", record.alert_id, exc)

    async def _send_alert_tts(self, record: AlertRecord, summary: str) -> None:
        tts_text = f"Alert at {record.location}: {summary}"
        voice_file = f"temp_alert_{record.alert_id}.ogg"

        try:
            tts = gTTS(text=tts_text, lang="en")
            tts.save(voice_file)

            await asyncio.to_thread(
                self.telegram_client.send_voice,
                record.delivery_chat_id,
                voice_file,
                record.telegram_message_id,
            )
        except Exception as exc:
            logger.warning("[TELEGRAM][TTS] Voice delivery failed for alert %s: %s", record.alert_id, exc)
        finally:
            if os.path.exists(voice_file):
                try:
                    os.remove(voice_file)
                except OSError:
                    pass
