"""
Standalone smoke-test for SwarmReasoningAgent.

Usage:
    python reasoning-layer/test_swarm.py

Requires GROQ_API_KEY in .env or environment.
Does NOT need camera, Telegram, or database.
"""

import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

# Allow imports from reasoning-layer directory
sys.path.insert(0, str(Path(__file__).resolve().parent))

from swarm_reasoning_agent import SwarmReasoningAgent


# ---------------------------------------------------------------------------
# Minimal mock dataclasses that mimic the perception layer output
# ---------------------------------------------------------------------------

@dataclass
class MockWeapon:
    label: str = "gun"
    confidence: float = 0.82
    bbox: list = field(default_factory=list)


@dataclass
class MockEmotion:
    label: str = "angry"
    confidence: float = 0.75


@dataclass
class MockTone:
    label: str = "aggressive"
    confidence: float = 0.68


@dataclass
class MockUniform:
    present: bool = False
    confidence: float = 0.0


@dataclass
class MockPerceptionResult:
    frame_id: int = 1
    timestamp: str = "2026-04-19T12:00:00Z"
    weapon: MockWeapon = field(default_factory=MockWeapon)
    emotion: MockEmotion = field(default_factory=MockEmotion)
    tone: MockTone = field(default_factory=MockTone)
    uniform: MockUniform = field(default_factory=MockUniform)
    is_danger: bool = True
    danger_reasons: list = field(default_factory=lambda: ["weapon:gun:0.82", "emotion:angry:0.75"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("  Cipher_NAISC – SwarmReasoningAgent Test")
    print("=" * 60)

    mock_result = MockPerceptionResult()
    learning_context = ""  # no historical incidents for this test

    print("\n[TEST] Initialising SwarmReasoningAgent…")
    agent = SwarmReasoningAgent()

    print("[TEST] Running parallel agent analysis…")
    result = agent.analyse(mock_result, learning_context)

    print("\n" + "=" * 60)
    print("  INDIVIDUAL AGENT REPORTS")
    print("=" * 60)

    agent_labels = {
        "threat_analyst": "Security Threat Analyst",
        "psychologist": "Behavioural Psychologist",
        "crowd_expert": "Crowd Safety Expert",
        "historian": "Historical Incident Analyst",
        "tactician": "Tactical Response Specialist",
    }
    for key, label in agent_labels.items():
        report = result.agent_reports.get(key, {})
        print(f"\n--- {label} ---")
        print(json.dumps(report, indent=2))

    print("\n" + "=" * 60)
    print("  COORDINATOR FINAL DECISION")
    print("=" * 60)
    print(json.dumps(result.to_dict(), indent=2, default=str))
    print("\n[TEST] Done.")


if __name__ == "__main__":
    main()
