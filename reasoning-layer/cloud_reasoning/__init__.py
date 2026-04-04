"""
Cloud Reasoning - Full-featured reasoning for cloud servers.

This package contains the CloudReasoningAgent which provides:
- 4-factor threat analysis (weapon, emotion, audio, behavioral)
- SOP context integration
- Learning agent predictions
- Detailed explanation generation
- High accuracy reasoning

Deployment: Cloud server (AWS Lambda, Docker, etc)
Latency: 100-500ms
Offline: No (requires server)
"""

from .cloud_reasoning_agent import (
    CloudReasoningAgent,
    ScenarioPrediction,
    SOPContext,
    RiskLevel,
)

__all__ = [
    "CloudReasoningAgent",
    "ScenarioPrediction",
    "SOPContext",
    "RiskLevel",
]

__version__ = "1.0.0"
