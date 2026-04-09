from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def load_env_file(env_path: Path) -> None:
    """Load simple KEY=VALUE pairs from a local .env file if present."""
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _csv_set(value: str, default: set[str]) -> set[str]:
    if not value:
        return default
    items = {item.strip().lower() for item in value.split(",") if item.strip()}
    return items or default


@dataclass(slots=True)
class Settings:
    app_name: str
    app_version: str
    telegram_bot_token: str | None
    telegram_chat_id: str | None
    operator_console_url: str | None
    telegram_max_video_bytes: int
    dedupe_window_seconds: int
    escalation_timeout_seconds: int
    alert_min_threat_level: str
    always_send_threat_levels: set[str] = field(default_factory=set)
    always_send_anomalies: set[str] = field(default_factory=set)
    escalation_enabled: bool = True
    tts_enabled: bool = False

    @property
    def telegram_enabled(self) -> bool:
        return bool(self.telegram_bot_token and self.telegram_chat_id)

    @property
    def demo_mode(self) -> bool:
        return not self.telegram_enabled


def get_settings() -> Settings:
    base_dir = Path(__file__).resolve().parents[1]
    load_env_file(base_dir / ".env")

    return Settings(
        app_name="NAISC Telegram Alert Service",
        app_version="0.1.0",
        telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN"),
        telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID"),
        operator_console_url=os.getenv("OPERATOR_CONSOLE_URL"),
        telegram_max_video_bytes=int(os.getenv("TELEGRAM_MAX_VIDEO_BYTES", "45000000")),
        dedupe_window_seconds=int(os.getenv("ALERT_DEDUPE_WINDOW_SECONDS", "300")),
        escalation_timeout_seconds=int(os.getenv("ALERT_ESCALATION_TIMEOUT_SECONDS", "180")),
        alert_min_threat_level=os.getenv("ALERT_MIN_THREAT_LEVEL", "high").lower(),
        always_send_threat_levels=_csv_set(
            os.getenv("ALWAYS_SEND_THREAT_LEVELS", "critical"),
            {"critical"},
        ),
        always_send_anomalies=_csv_set(
            os.getenv("ALWAYS_SEND_ANOMALIES", "weapon_detected"),
            {"weapon_detected"},
        ),
        escalation_enabled=os.getenv("ALERT_ESCALATION_ENABLED", "true").lower() != "false",
        tts_enabled=os.getenv("ALERT_TTS_ENABLED", "false").lower() == "true",
    )
