"""
AgentScope multi-agent swarm for Cipher_NAISC threat assessment.

Five specialist agents analyse a perception event in parallel;
a coordinator aggregates their JSON reports into a final decision.

Compatible with agentscope >= 1.0.18 (OpenAIChatModel / async API).
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import json
import logging
import os
import re
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_swarm_ml_model: Any = None


def _get_swarm_ml_model() -> Any:
    global _swarm_ml_model
    if _swarm_ml_model is None:
        try:
            _learning_path = Path(__file__).resolve().parent.parent / "learning-layer"
            import sys
            if str(_learning_path) not in sys.path:
                sys.path.insert(0, str(_learning_path))
            from ml_model import CipherMLModel  # type: ignore[import]
            _swarm_ml_model = CipherMLModel()
            logger.info("[SWARM] CipherMLModel loaded (samples=%d)", _swarm_ml_model._samples_seen)
        except Exception as exc:
            logger.debug("[SWARM] ML model unavailable: %s", exc)
    return _swarm_ml_model

# ---------------------------------------------------------------------------
# Agent metadata — shared by format_swarm_output and the frontend helper
# ---------------------------------------------------------------------------

AGENT_META: dict[str, tuple[str, str, str]] = {
    "threat_analyst": ("🔫", "SECURITY THREAT ANALYST",       "threat_level"),
    "psychologist":   ("🧠", "BEHAVIOURAL PSYCHOLOGIST",      "behaviour_risk"),
    "crowd_expert":   ("👥", "CROWD SAFETY EXPERT",           "crowd_risk"),
    "historian":      ("📋", "HISTORICAL INCIDENT ANALYST",   "pattern_match"),
    "tactician":      ("🎯", "TACTICAL RESPONSE SPECIALIST",  "urgency"),
}

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

# ---------------------------------------------------------------------------
# agentscope init (no model_configs in v1.0.18)
# ---------------------------------------------------------------------------

_AGENTSCOPE_INITED = False


def _ensure_agentscope_init() -> None:
    global _AGENTSCOPE_INITED
    if _AGENTSCOPE_INITED:
        return
    try:
        import agentscope
        agentscope.init()
        logger.info("[SWARM] agentscope.init() OK")
    except Exception as exc:
        logger.debug("[SWARM] agentscope.init() skipped: %s", exc)
    _AGENTSCOPE_INITED = True


def _make_model() -> Any:
    """Create a fresh OpenAIChatModel instance (each has its own httpx client)."""
    _ensure_agentscope_init()
    from agentscope.model import OpenAIChatModel
    return OpenAIChatModel(
        model_name=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        api_key=os.getenv("GROQ_API_KEY"),
        stream=False,
        client_kwargs={"base_url": "https://api.groq.com/openai/v1"},
        generate_kwargs={"temperature": 0.3, "max_tokens": 512},
    )


# ---------------------------------------------------------------------------
# Thin agent shim — each agent owns its own model/httpx-client instance
# ---------------------------------------------------------------------------

class _Agent:
    """Minimal agent: holds a system prompt and its own model instance."""

    def __init__(self, name: str, sys_prompt: str) -> None:
        self.name = name
        self.sys_prompt = sys_prompt
        # Each agent gets its own OpenAIChatModel so httpx connection pools
        # are isolated — avoids "Event loop is closed" errors across threads.
        self._model = _make_model()

    def call(self, user_content: str) -> str:
        """Call the model synchronously — safe to use from a thread pool."""
        messages = [
            {"role": "system", "content": self.sys_prompt},
            {"role": "user", "content": user_content},
        ]
        response = self._model(messages)
        return response.text or ""


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SwarmReasoningResult:
    """Aggregated output from the 5-agent swarm."""

    incident_id: str
    incident_summary: str
    overall_threat_level: str
    final_action: str
    secondary_action: Optional[str]
    urgency: str
    confidence: float
    false_positive_likelihood: float
    agent_reports: dict
    timestamp: str
    ml_prediction: dict = dataclasses.field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "summary": self.incident_summary,
            "course_of_action": self.final_action,
            "confidence": self.confidence,
            "reasoning": f"{self.overall_threat_level} – {self.urgency}",
            "overall_threat_level": self.overall_threat_level,
            "secondary_action": self.secondary_action,
            "urgency": self.urgency,
            "false_positive_likelihood": self.false_positive_likelihood,
            "agent_reports": self.agent_reports,
            "timestamp": self.timestamp,
            "ml_prediction": self.ml_prediction,
        }


# ---------------------------------------------------------------------------
# Formatting helper (module-level so alert_manager can import it directly)
# ---------------------------------------------------------------------------

def format_swarm_output(result: "SwarmReasoningResult") -> str:
    """Return a war-room style plain-text breakdown."""
    divider = "━" * 36
    lines: list[str] = [
        divider,
        "🚨 CIPHER NAISC — INCIDENT ANALYSIS",
        f"Incident: {result.incident_id[:8]}  |  {result.timestamp[11:19]}",
        divider,
        "",
    ]

    for key, (icon, label, risk_field) in AGENT_META.items():
        report = result.agent_reports.get(key, {})
        if "error" in report:
            lines.append(f"{icon} {label}   [UNAVAILABLE]")
            lines.append(f'"{report["error"]}"')
        else:
            risk_value = report.get(risk_field, "—")
            findings = (
                report.get("findings")
                or report.get("reasoning")
                or "No findings returned."
            )
            if key == "tactician":
                action_line = f" Primary: {report.get('primary_action', '—')}"
                if report.get("secondary_action"):
                    action_line += f"\n Secondary: {report['secondary_action']}"
                findings = action_line + "\n " + str(findings)
            lines.append(f"{icon} {label:<32} [{risk_value}]")
            lines.append(f'"{findings}"')
        lines.append("")

    ml_pred = result.ml_prediction or {}
    ml_line = ""
    if ml_pred.get("based_on_samples", 0) > 0:
        prob_pct = int(ml_pred["is_threat_probability"] * 100)
        n_samp   = ml_pred["based_on_samples"]
        ml_line  = f"🤖 ML Model: {prob_pct}% threat probability ({n_samp} training samples)"

    lines += [
        divider,
        "📊 COORDINATOR VERDICT",
        f"Overall threat: {result.overall_threat_level}  |  Confidence: {int(result.confidence * 100)}%",
        f"False positive likelihood: {int(result.false_positive_likelihood * 100)}%",
        "",
        "SUMMARY:",
        result.incident_summary,
        "",
        f"ACTION: {result.final_action}",
    ]
    if result.secondary_action:
        lines.append(f"SECONDARY: {result.secondary_action}")
    lines.append(f"URGENCY: {result.urgency}")
    if ml_line:
        lines.append("")
        lines.append(ml_line)
    lines.append(divider)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Swarm
# ---------------------------------------------------------------------------

class SwarmReasoningAgent:
    """Five specialist agents + one coordinator, run in parallel via threads."""

    def __init__(self) -> None:
        _ensure_agentscope_init()

        self.threat_analyst = _Agent(
            name="ThreatAnalyst",
            sys_prompt=(
                "You are a security threat analyst with 20 years of law enforcement experience.\n"
                "Analyse ONLY weapon and physical danger signals from the detection data given.\n"
                "Respond ONLY in valid JSON — no preamble, no markdown.\n"
                'Schema: {"threat_level":"CRITICAL|HIGH|MEDIUM|LOW|NONE","findings":"str",'
                '"confidence":0.0,"weapon_detected":false,"weapon_type":null}'
            ),
        )

        self.psychologist = _Agent(
            name="Psychologist",
            sys_prompt=(
                "You are a forensic behavioural psychologist specialising in threat assessment.\n"
                "Analyse ONLY emotion labels and tone classification from the detection data.\n"
                "Respond ONLY in valid JSON.\n"
                'Schema: {"behaviour_risk":"CRITICAL|HIGH|MEDIUM|LOW|NONE","findings":"str",'
                '"confidence":0.0,"escalation_likely":false,"dominant_emotion":"str"}'
            ),
        )

        self.crowd_expert = _Agent(
            name="CrowdExpert",
            sys_prompt=(
                "You are a crowd safety expert. Analyse bystander risk and escalation potential.\n"
                "Consider number of people, uniform presence, and spatial context.\n"
                "Respond ONLY in valid JSON.\n"
                'Schema: {"crowd_risk":"CRITICAL|HIGH|MEDIUM|LOW|NONE","findings":"str",'
                '"confidence":0.0,"bystander_count_estimate":0,"evacuation_needed":false}'
            ),
        )

        self.historian = _Agent(
            name="Historian",
            sys_prompt=(
                "You are a security data analyst specialising in incident pattern recognition.\n"
                "You receive current detections AND a summary of similar past incidents.\n"
                "Identify patterns and false positive signals.\n"
                "Respond ONLY in valid JSON.\n"
                'Schema: {"pattern_match":"STRONG|MODERATE|WEAK|NONE","findings":"str",'
                '"confidence":0.0,"similar_incident_count":0,"false_positive_likelihood":0.0}'
            ),
        )

        self.tactician = _Agent(
            name="Tactician",
            sys_prompt=(
                "You are a tactical response specialist for a security operations centre.\n"
                "Recommend the most proportionate response. Choose from:\n"
                "DISPATCH_OFFICERS, INCREASE_SURVEILLANCE, ISSUE_VERBAL_WARNING, REVIEW_FOOTAGE, "
                "LOCKDOWN_AREA, CONTACT_EMERGENCY_SERVICES, FALSE_ALARM, MONITOR_ONLY\n"
                "Respond ONLY in valid JSON.\n"
                'Schema: {"primary_action":"str","secondary_action":null,"reasoning":"str",'
                '"confidence":0.0,"urgency":"IMMEDIATE|URGENT|ROUTINE"}'
            ),
        )

        self.coordinator = _Agent(
            name="Coordinator",
            sys_prompt=(
                "You are the Security Operations Coordinator.\n"
                "You receive JSON reports from 5 specialist agents. Weigh all inputs — give more "
                "weight to higher confidence scores. If agents strongly disagree, err on the side "
                "of caution.\n"
                "Respond ONLY in valid JSON.\n"
                'Schema: {"incident_summary":"str","overall_threat_level":"CRITICAL|HIGH|MEDIUM|LOW|NONE",'
                '"final_action":"str","secondary_action":null,"urgency":"IMMEDIATE|URGENT|ROUTINE",'
                '"confidence":0.0,"false_positive_likelihood":0.0,"dissenting_views":null}'
            ),
        )

        logger.info("[SWARM] SwarmReasoningAgent ready: 5 specialists + coordinator")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyse(self, perception_result: Any, learning_context: str = "") -> SwarmReasoningResult:
        # Get ML prediction before spinning up agents
        ml_prediction = self._get_ml_prediction(perception_result)

        perception_msg = self._build_perception_message(perception_result)
        timeout = int(os.getenv("SWARM_TIMEOUT_SECONDS", "30"))

        # Append ML hint to historian context so Agent 4 sees it
        enriched_context = learning_context
        n_samples = ml_prediction.get("based_on_samples", 0)
        if n_samples > 0:
            prob_pct = ml_prediction["is_threat_probability"] * 100
            enriched_context += (
                f"\nML Model prediction: {prob_pct:.0f}% threat probability "
                f"based on {n_samples} past confirmed incidents."
            )

        agent_reports = self._run_agents_parallel(perception_msg, enriched_context, timeout)

        # Build coordinator message with ML hint injected
        ml_prob = ml_prediction.get("is_threat_probability", 0.5)
        n_samples = ml_prediction.get("based_on_samples", 0)
        ml_coord_hint = ""
        if n_samples > 0:
            if ml_prob >= 0.80:
                ml_coord_hint = (
                    "\n⚠️ HIGH ML CONFIDENCE — historical data strongly suggests "
                    "this is a real threat"
                )
            elif ml_prob <= 0.20:
                ml_coord_hint = (
                    "\nℹ️ LOW ML CONFIDENCE — historical data suggests this may be "
                    "a false alarm, consider REVIEW_FOOTAGE before escalating"
                )

        coordinator_msg = self._build_coordinator_message(agent_reports)
        if ml_coord_hint:
            coordinator_msg += "\n\nML Model Assessment:" + ml_coord_hint
        coordinator_raw = self.coordinator.call(coordinator_msg)
        final = self._parse_json_safe(coordinator_raw)
        agent_reports["coordinator"] = final

        return SwarmReasoningResult(
            incident_id=str(uuid.uuid4()),
            incident_summary=final.get("incident_summary", ""),
            overall_threat_level=final.get("overall_threat_level", "MEDIUM"),
            final_action=final.get("final_action", "REVIEW_FOOTAGE"),
            secondary_action=final.get("secondary_action"),
            urgency=final.get("urgency", "ROUTINE"),
            confidence=float(final.get("confidence", 0.5)),
            false_positive_likelihood=float(final.get("false_positive_likelihood", 0.0)),
            agent_reports=agent_reports,
            timestamp=datetime.utcnow().isoformat(),
            ml_prediction=ml_prediction,
        )

    def format_swarm_output(self, result: SwarmReasoningResult) -> str:
        return format_swarm_output(result)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_ml_prediction(self, perception_result: Any) -> dict:
        """Return ML threat probability (best-effort; returns safe defaults on failure)."""
        _default = {"is_threat_probability": 0.5, "suggested_action": "REVIEW_FOOTAGE",
                    "confidence": 0.0, "based_on_samples": 0}
        try:
            ml = _get_swarm_ml_model()
            if ml is None:
                return _default
            data = perception_result
            if dataclasses.is_dataclass(perception_result) and not isinstance(perception_result, type):
                data = dataclasses.asdict(perception_result)
            elif not isinstance(perception_result, dict):
                data = {"raw": str(perception_result)}
            return ml.predict(data)
        except Exception as exc:
            logger.debug("[SWARM] ML prediction skipped: %s", exc)
            return _default

    def _run_agents_parallel(
        self, perception_msg: str, learning_context: str, timeout: int
    ) -> dict:
        historian_content = perception_msg
        if learning_context:
            historian_content = perception_msg + "\n\nPAST INCIDENTS:\n" + learning_context

        def _call(agent: _Agent, content: str) -> dict:
            try:
                raw = agent.call(content)
                return self._parse_json_safe(raw)
            except Exception as exc:
                logger.warning("[SWARM] Agent %s failed: %s", agent.name, exc)
                return {"error": str(exc)}

        tasks = {
            "threat_analyst": (self.threat_analyst, perception_msg),
            "psychologist":   (self.psychologist,   perception_msg),
            "crowd_expert":   (self.crowd_expert,   perception_msg),
            "historian":      (self.historian,       historian_content),
            "tactician":      (self.tactician,       perception_msg),
        }

        results: dict[str, dict] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = {
                key: executor.submit(_call, agent, content)
                for key, (agent, content) in tasks.items()
            }
            for key, future in futures.items():
                try:
                    results[key] = future.result(timeout=timeout)
                except concurrent.futures.TimeoutError:
                    logger.warning("[SWARM] Agent %s timed out after %ds", key, timeout)
                    results[key] = {"error": "Agent timed out"}
                except Exception as exc:
                    results[key] = {"error": str(exc)}

        return results

    def _build_perception_message(self, perception_result: Any) -> str:
        if dataclasses.is_dataclass(perception_result) and not isinstance(perception_result, type):
            data = dataclasses.asdict(perception_result)
        elif isinstance(perception_result, dict):
            data = perception_result
        else:
            data = {"raw": str(perception_result)}
        return json.dumps(data, default=str)

    def _build_coordinator_message(self, agent_reports: dict) -> str:
        labels = {
            "threat_analyst": "Security Threat Analyst",
            "psychologist":   "Behavioural Psychologist",
            "crowd_expert":   "Crowd Safety Expert",
            "historian":      "Historical Incident Analyst",
            "tactician":      "Tactical Response Specialist",
        }
        lines = ["Agent reports:"]
        for i, (key, label) in enumerate(labels.items(), 1):
            report = agent_reports.get(key, {})
            lines.append(f"{i}. {label}:\n{json.dumps(report, indent=2)}")
        return "\n".join(lines)

    def _parse_json_safe(self, content: str) -> dict:
        if not content:
            return {}
        stripped = re.sub(r"^```(?:json)?\s*", "", content.strip(), flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            logger.warning("[SWARM] Failed to parse JSON response: %s", content[:300])
            return {"raw_response": content}
