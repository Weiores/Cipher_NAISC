from __future__ import annotations

import logging
from typing import Any

from fastapi import Depends, FastAPI, HTTPException

from app.dependencies import repository_dependency, settings_dependency, telegram_service_dependency
from app.schemas import TelegramAlertRequest, TelegramHealth
from app.service import TelegramAlertService


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

app = FastAPI(
    title="NAISC UI Layer Telegram Alerts",
    version="0.1.0",
    description="Demo-friendly Telegram alerting service for reasoning outputs.",
)


@app.get("/telegram/health", response_model=TelegramHealth)
async def telegram_health() -> TelegramHealth:
    settings = settings_dependency()
    repository = repository_dependency()
    return TelegramHealth(
        status="ok",
        demo_mode=settings.demo_mode,
        telegram_enabled=settings.telegram_enabled,
        active_alerts=repository.active_alert_count(),
        feedback_events=repository.feedback_count(),
    )


@app.post("/telegram/alert")
async def send_telegram_alert(
    payload: dict[str, Any],
    service: TelegramAlertService = Depends(telegram_service_dependency),
):
    try:
        alert_request = TelegramAlertRequest.from_payload(payload)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid reasoning payload: {exc}") from exc

    return await service.send_alert(alert_request)


@app.get("/")
async def root() -> dict[str, str]:
    return {
        "service": "telegram-alerting",
        "health": "/telegram/health",
        "send_alert": "/telegram/alert",
    }
