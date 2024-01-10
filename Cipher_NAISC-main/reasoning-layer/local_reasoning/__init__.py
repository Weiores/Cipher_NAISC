"""
Local Reasoning - Lightweight reasoning for edge devices.

This package contains the LocalReasoningAgent which provides:
- Fast threat assessment (<100ms)
- Weapon-based scoring (primary factor)
- Offline operation (no cloud needed)
- Minimal resources (mobile-friendly)
- Good accuracy with simplicity

Deployment: Officer's phone, body cam, edge device
Latency: <100ms
Offline: Yes (fully offline)
"""

from .local_reasoning_agent import (
    LocalReasoningAgent,
    RiskLevel,
)

__all__ = [
    "LocalReasoningAgent",
    "RiskLevel",
]

__version__ = "1.0.0"
