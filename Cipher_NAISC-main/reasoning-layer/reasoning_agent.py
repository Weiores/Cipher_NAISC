"""
Groq-powered reasoning agent for Cipher_NAISC.

Converts raw perception detections into a human-readable incident summary
and recommends a course of action using llama-3.3-70b-versatile via the
Groq API.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# .env loader (standalone usage)
# ---------------------------------------------------------------------------

def _load_env() -> None:
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_env()

_GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
_GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
_GROQ_FALLBACK_MODEL = "mixtral-8x7b-32768"

try:
    from groq import Groq
    _GROQ_AVAILABLE = bool(_GROQ_API_KEY)
except ImportError:
    Groq = None  # type: ignore[assignment,misc]
    _GROQ_AVAILABLE = False
    logger.warning("[REASONING] groq library not installed; rule-based fallback will be used")

_SYSTEM_PROMPT = (
    "You are a security operations AI assistant for a surveillance system. "
    "You analyse threat detections from cameras and recommend clear, proportionate responses. "
    "Always respond in valid JSON format only. Be concise and professional."
)

_VALID_ACTIONS = {
    "DISPATCH_OFFICERS",
    "INCREASE_SURVEILLANCE",
    "ISSUE_VERBAL_WARNING",
    "REVIEW_FOOTAGE",
    "FALSE_ALARM",
}


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class ReasoningResult:
    """Output from the reasoning agent for a single incident."""

    incident_id: str
    summary: str
    course_of_action: str
    confidence: float
    reasoning: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "summary": self.summary,
            "course_of_action": self.course_of_action,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class ReasoningAgent:
    """Uses Groq LLM to summarise incidents and recommend actions."""

    def __init__(self) -> None:
        self._client = Groq(api_key=_GROQ_API_KEY) if (_GROQ_AVAILABLE and Groq) else None
        mode = "Groq" if self._client else "rule-based fallback"
        logger.info("[REASONING] Agent initialised with %s backend", mode)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def summarise(self, perception_result: dict[str, Any]) -> str:
        """Convert raw detection dict into a human-readable incident summary.

        Args:
            perception_result: Output dict from PerceptionLayer.process_frame.

        Returns:
            A concise incident summary string.
        """
        prompt = self._build_summary_prompt(perception_result)
        if self._client:
            raw = self._call_groq(prompt, expect_key="summary")
            if raw:
                return str(raw)
        return self._fallback_summary(perception_result)

    def get_learning_input(self, incident_history: list[dict[str, Any]]) -> str:
        """Format past incidents into a prompt-friendly learning context string.

        Args:
            incident_history: List of past incident dicts from the database.

        Returns:
            A formatted context string for inclusion in reasoning prompts.
        """
        if not incident_history:
            return "No historical incidents available."

        lines = ["Recent similar incidents and their outcomes:"]
        for i, inc in enumerate(incident_history[:5], 1):
            action = inc.get("officer_action") or inc.get("recommended_action") or "N/A"
            outcome = inc.get("final_outcome") or "N/A"
            weapon = "unknown"
            detections = inc.get("detections")
            if isinstance(detections, list) and detections:
                weapon = detections[0].get("label", "unknown")
            elif isinstance(detections, dict):
                weapon = detections.get("label", "unknown")

            lines.append(f"  {i}. Weapon={weapon}, Action={action}, Outcome={outcome}")

        return "\n".join(lines)

    def determine_course_of_action(
        self,
        summary: str,
        learning_input: str,
    ) -> dict[str, Any]:
        """Recommend a course of action using Groq LLM.

        Args:
            summary:        Human-readable incident summary.
            learning_input: Historical context from the learning agent.

        Returns:
            Dict with keys: action, reasoning, confidence.
        """
        prompt = self._build_action_prompt(summary, learning_input)
        if self._client:
            result = self._call_groq_json(prompt)
            if result:
                return self._validate_action_result(result)
        return self._fallback_action(summary)

    def process(
        self,
        perception_result: dict[str, Any],
        incident_history: list[dict[str, Any]] | None = None,
    ) -> ReasoningResult:
        """Full reasoning pipeline: summarise → context → action.

        Args:
            perception_result: Output from PerceptionLayer.
            incident_history:  Past incidents from the learning agent.

        Returns:
            :class:`ReasoningResult` with all reasoning outputs.
        """
        incident_id = f"INC-{uuid.uuid4().hex[:8].upper()}"
        summary = self.summarise(perception_result)
        learning_input = self.get_learning_input(incident_history or [])
        action_result = self.determine_course_of_action(summary, learning_input)

        result = ReasoningResult(
            incident_id=incident_id,
            summary=summary,
            course_of_action=action_result.get("action", "REVIEW_FOOTAGE"),
            confidence=float(action_result.get("confidence", 0.5)),
            reasoning=action_result.get("reasoning", ""),
        )

        logger.info(
            "[REASONING] %s → action=%s conf=%.2f",
            incident_id, result.course_of_action, result.confidence,
        )
        return result

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------

    def _build_summary_prompt(self, perception: dict[str, Any]) -> str:
        weapon = perception.get("weapon", {})
        emotion = perception.get("emotion", {})
        tone = perception.get("tone", {})
        uniform = perception.get("uniform", {})
        reasons = perception.get("danger_reasons", [])

        return (
            f"Analyse the following security camera detection data and produce a concise "
            f"incident summary.\n\n"
            f"Detection data:\n"
            f"- Weapon: {weapon.get('label','none')} (confidence {weapon.get('confidence',0):.0%})\n"
            f"- Emotion: {emotion.get('label','unknown')} (confidence {emotion.get('confidence',0):.0%})\n"
            f"- Audio tone: {tone.get('label','unknown')} (confidence {tone.get('confidence',0):.0%})\n"
            f"- Uniform present: {'yes' if uniform.get('present') else 'no'}\n"
            f"- Danger triggers: {', '.join(reasons) if reasons else 'none'}\n"
            f"- Timestamp: {perception.get('timestamp','unknown')}\n\n"
            f'Respond in JSON: {{"summary": "your incident summary here"}}'
        )

    def _build_action_prompt(self, summary: str, learning_input: str) -> str:
        actions_list = ", ".join(sorted(_VALID_ACTIONS))
        return (
            f"You are a security operations expert. Based on the incident summary and "
            f"historical context below, recommend the most appropriate action.\n\n"
            f"Incident summary:\n{summary}\n\n"
            f"Historical context:\n{learning_input}\n\n"
            f"Available actions: {actions_list}\n\n"
            f"Respond ONLY in JSON:\n"
            f'{{"action": "ONE_OF_THE_ACTIONS", "reasoning": "brief justification", "confidence": 0.0_to_1.0}}'
        )

    # ------------------------------------------------------------------
    # Groq API calls
    # ------------------------------------------------------------------

    def _call_groq(self, prompt: str, expect_key: str) -> Any:
        """Call Groq and extract a specific key from the JSON response."""
        raw = self._call_groq_json(prompt)
        if raw and expect_key in raw:
            return raw[expect_key]
        return None

    def _call_groq_json(self, prompt: str) -> dict[str, Any] | None:
        """Call Groq and return parsed JSON; try fallback model on failure."""
        for model in (_GROQ_MODEL, _GROQ_FALLBACK_MODEL):
            try:
                response = self._client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.2,
                    max_tokens=512,
                )
                content = response.choices[0].message.content
                return json.loads(content)
            except Exception as exc:
                logger.warning("[REASONING] Groq call failed (model=%s): %s", model, exc)
        return None

    # ------------------------------------------------------------------
    # Fallbacks (no LLM)
    # ------------------------------------------------------------------

    def _fallback_summary(self, perception: dict[str, Any]) -> str:
        weapon = perception.get("weapon", {}).get("label", "unknown")
        emotion = perception.get("emotion", {}).get("label", "unknown")
        tone = perception.get("tone", {}).get("label", "unknown")
        uniform = "uniformed" if perception.get("uniform", {}).get("present") else "civilian"
        ts = perception.get("timestamp", "unknown time")
        return (
            f"Security alert at {ts}. "
            f"Detected: weapon={weapon}, emotion={emotion}, tone={tone}. "
            f"Subject appears to be {uniform}. Immediate review recommended."
        )

    def _fallback_action(self, summary: str) -> dict[str, Any]:
        summary_lower = summary.lower()
        if any(w in summary_lower for w in ("gun", "rifle", "shotgun")):
            action, conf = "DISPATCH_OFFICERS", 0.90
        elif any(w in summary_lower for w in ("knife", "blade")):
            action, conf = "DISPATCH_OFFICERS", 0.80
        elif any(w in summary_lower for w in ("angry", "threat", "panic")):
            action, conf = "INCREASE_SURVEILLANCE", 0.65
        else:
            action, conf = "REVIEW_FOOTAGE", 0.55
        return {"action": action, "reasoning": "Heuristic fallback", "confidence": conf}

    def _validate_action_result(self, result: dict[str, Any]) -> dict[str, Any]:
        """Ensure returned action is one of the valid options."""
        action = str(result.get("action", "")).strip().upper()
        if action not in _VALID_ACTIONS:
            action = "REVIEW_FOOTAGE"
        confidence = float(result.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))
        return {
            "action": action,
            "reasoning": str(result.get("reasoning", "")),
            "confidence": confidence,
        }


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    from datetime import datetime, timezone
    logging.basicConfig(level=logging.DEBUG)

    agent = ReasoningAgent()

    mock_perception = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "danger_reasons": ["weapon:gun:0.85"],
        "weapon": {"label": "gun", "confidence": 0.85},
        "emotion": {"label": "angry", "confidence": 0.78},
        "tone": {"label": "threat", "confidence": 0.70},
        "uniform": {"present": False},
    }

    result = agent.process(mock_perception)
    print(json.dumps(result.to_dict(), indent=2))
