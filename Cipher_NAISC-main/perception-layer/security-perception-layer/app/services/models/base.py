from abc import ABC, abstractmethod
from typing import Any

from app.schemas import PerceptionRequest


class ModelAdapter(ABC):
    name: str
    intended_backend: str

    @abstractmethod
    async def infer(self, request: PerceptionRequest) -> Any:
        raise NotImplementedError

    def describe(self) -> dict[str, str]:
        return {
            "name": self.name,
            "intended_backend": self.intended_backend,
        }
