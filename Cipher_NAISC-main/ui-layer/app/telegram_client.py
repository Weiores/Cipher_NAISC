from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import requests

from app.config import Settings


logger = logging.getLogger(__name__)


class TelegramDeliveryError(Exception):
    """Raised when a live Telegram API call cannot be completed."""


class TelegramClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @property
    def is_enabled(self) -> bool:
        return self.settings.telegram_enabled

    def _build_url(self, method: str) -> str:
        token = self.settings.telegram_bot_token
        if not token:
            raise RuntimeError("TELEGRAM_BOT_TOKEN is missing")
        return f"https://api.telegram.org/bot{token}/{method}"

    def send_alert_message(
        self,
        message_html: str,
        reply_markup: dict[str, Any],
        reply_to_message_id: int | None = None,
    ) -> int | None:
        return self.send_message(self.settings.telegram_chat_id, message_html, reply_markup, reply_to_message_id)

    def send_message(
        self,
        chat_id: str | int | None,
        message_html: str,
        reply_markup: dict[str, Any],
        reply_to_message_id: int | None = None,
    ) -> int | None:
        if not self.is_enabled:
            logger.warning("[TELEGRAM][DEMO] %s", message_html.replace("\n", " | "))
            return None
        if chat_id in (None, ""):
            raise TelegramDeliveryError("chat_id is missing")

        try:
            payload = {
                "chat_id": chat_id,
                "text": message_html,
                "parse_mode": "HTML",
                "reply_markup": reply_markup,
                "disable_web_page_preview": True,
            }
            if reply_to_message_id is not None:
                payload["reply_parameters"] = {"message_id": reply_to_message_id}

            response = requests.post(
                self._build_url("sendMessage"),
                json=payload,
                timeout=15,
            )
            response.raise_for_status()
            payload = response.json()
            return payload.get("result", {}).get("message_id")
        except requests.RequestException as exc:
            raise TelegramDeliveryError(str(exc)) from exc

    def answer_callback_query(self, callback_query_id: str, text: str) -> None:
        if not self.is_enabled:
            logger.info("[TELEGRAM][DEMO] callback answered: %s", text)
            return

        try:
            response = requests.post(
                self._build_url("answerCallbackQuery"),
                json={"callback_query_id": callback_query_id, "text": text, "show_alert": False},
                timeout=15,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise TelegramDeliveryError(str(exc)) from exc

    def send_voice(
        self,
        chat_id: str | int | None,
        voice_path: str,
        reply_to_message_id: int | None = None,
    ) -> int | None:
        if not self.is_enabled:
            logger.warning("[TELEGRAM][DEMO] voice=%s", voice_path)
            return None
        if chat_id in (None, ""):
            raise TelegramDeliveryError("chat_id is missing")

        file_path = Path(voice_path)
        if not file_path.exists():
            raise TelegramDeliveryError(f"voice file not found: {voice_path}")

        try:
            data: dict[str, Any] = {"chat_id": chat_id}
            if reply_to_message_id is not None:
                data["reply_parameters"] = {"message_id": reply_to_message_id}

            with file_path.open("rb") as voice_file:
                response = requests.post(
                    self._build_url("sendVoice"),
                    data=data,
                    files={"voice": (file_path.name, voice_file, "audio/ogg")},
                    timeout=30,
                )
            response.raise_for_status()
            payload = response.json()
            return payload.get("result", {}).get("message_id")
        except requests.RequestException as exc:
            raise TelegramDeliveryError(str(exc)) from exc

    def send_followup_message(self, text: str) -> None:
        self.send_alert_message(text, reply_markup={"inline_keyboard": []})

    def send_video(
        self,
        chat_id: str | int | None,
        video_path: str,
        caption_html: str | None = None,
        reply_to_message_id: int | None = None,
    ) -> int | None:
        if not self.is_enabled:
            logger.warning("[TELEGRAM][DEMO] video=%s caption=%s", video_path, (caption_html or "").replace("\n", " | "))
            return None
        if chat_id in (None, ""):
            raise TelegramDeliveryError("chat_id is missing")

        file_path = Path(video_path)
        if not file_path.exists():
            raise TelegramDeliveryError(f"video file not found: {video_path}")

        try:
            data: dict[str, Any] = {"chat_id": chat_id}
            if caption_html:
                data["caption"] = caption_html
                data["parse_mode"] = "HTML"
            if reply_to_message_id is not None:
                data["reply_parameters"] = {"message_id": reply_to_message_id}

            with file_path.open("rb") as video_file:
                response = requests.post(
                    self._build_url("sendVideo"),
                    data=data,
                    files={"video": (file_path.name, video_file, "video/mp4")},
                    timeout=60,
                )
            response.raise_for_status()
            payload = response.json()
            return payload.get("result", {}).get("message_id")
        except requests.RequestException as exc:
            raise TelegramDeliveryError(str(exc)) from exc
