from __future__ import annotations

from functools import lru_cache

from app.config import Settings, get_settings
from app.repository import InMemoryAlertRepository
from app.service import TelegramAlertService
from app.telegram_client import TelegramClient


@lru_cache(maxsize=1)
def settings_dependency() -> Settings:
    return get_settings()


@lru_cache(maxsize=1)
def repository_dependency() -> InMemoryAlertRepository:
    return InMemoryAlertRepository()


@lru_cache(maxsize=1)
def telegram_service_dependency() -> TelegramAlertService:
    settings = settings_dependency()
    repository = repository_dependency()
    client = TelegramClient(settings)
    return TelegramAlertService(settings=settings, repository=repository, telegram_client=client)
