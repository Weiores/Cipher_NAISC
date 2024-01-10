"""
Learning agent for Cipher_NAISC.

Analyses past incidents to:
  1. Find similar historical incidents (TF-IDF cosine similarity).
  2. Compute recommendation accuracy statistics.
  3. Generate a learning context string for the reasoning agent.
  4. Optionally use Groq to produce an enriched learning summary.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# .env loader
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

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False
    logger.warning("[LEARNING] scikit-learn not installed; keyword-overlap similarity will be used")

# ---------------------------------------------------------------------------
# ML model — lazy singleton so it isn't loaded until first use
# ---------------------------------------------------------------------------

_ml_model: Any = None


def _get_ml_model() -> Any:
    global _ml_model
    if _ml_model is None:
        try:
            _here = Path(__file__).resolve().parent
            import sys
            if str(_here) not in sys.path:
                sys.path.insert(0, str(_here))
            from ml_model import CipherMLModel
            _ml_model = CipherMLModel()
            logger.info("[LEARNING] CipherMLModel loaded (samples=%d)", _ml_model._samples_seen)
        except Exception as exc:
            logger.warning("[LEARNING] CipherMLModel unavailable: %s", exc)
    return _ml_model

try:
    from groq import Groq
    _GROQ_AVAILABLE = bool(_GROQ_API_KEY)
except ImportError:
    Groq = None  # type: ignore[assignment,misc]
    _GROQ_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _incident_to_text(incident: dict[str, Any]) -> str:
    """Convert an incident record to a plain-text description for TF-IDF."""
    parts: list[str] = []
    detections = incident.get("detections")
    if isinstance(detections, list):
        for d in detections:
            parts.append(str(d.get("label", "")))
    elif isinstance(detections, dict):
        parts.append(str(detections.get("label", "")))
    elif isinstance(detections, str):
        try:
            parsed = json.loads(detections)
            if isinstance(parsed, list):
                parts += [str(d.get("label", "")) for d in parsed]
        except json.JSONDecodeError:
            parts.append(detections[:50])

    parts.append(str(incident.get("perception_summary", "")))
    parts.append(str(incident.get("recommended_action", "")))
    parts.append(str(incident.get("officer_action", "")))
    return " ".join(filter(None, parts))


def _perception_to_text(perception: dict[str, Any]) -> str:
    """Convert a current perception result to query text."""
    weapon = perception.get("weapon", {}).get("label", "")
    emotion = perception.get("emotion", {}).get("label", "")
    tone = perception.get("tone", {}).get("label", perception.get("tone", {}).get("tone", ""))
    return f"{weapon} {emotion} {tone}"


def _keyword_similarity(query: str, doc: str) -> float:
    """Simple word-overlap Jaccard similarity as fallback."""
    q_words = set(query.lower().split())
    d_words = set(doc.lower().split())
    if not q_words or not d_words:
        return 0.0
    intersection = q_words & d_words
    union = q_words | d_words
    return len(intersection) / len(union)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class LearningAgent:
    """Retrieves and analyses historical incident data to improve reasoning."""

    def __init__(self, db: Any | None = None) -> None:
        """Initialise the learning agent.

        Args:
            db: An optional :class:`IncidentDatabase` instance. If supplied,
                the agent will auto-load incident history from the database.
        """
        self._db = db
        self._groq_client = (
            Groq(api_key=_GROQ_API_KEY) if (_GROQ_AVAILABLE and Groq) else None
        )
        logger.info("[LEARNING] Learning agent initialised")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_similar_incidents(
        self,
        current_perception: dict[str, Any],
        top_k: int = 5,
        incident_history: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Find the most similar past incidents to the current detection.

        Args:
            current_perception: Current perception result dict.
            top_k:              Maximum number of similar incidents to return.
            incident_history:   If not provided, uses the database if available.

        Returns:
            List of matching incident dicts, ordered by similarity descending.
        """
        history = incident_history or self._load_history()
        if not history:
            return []

        query = _perception_to_text(current_perception)

        if _SKLEARN_AVAILABLE and len(history) >= 2:
            return self._tfidf_similarity(query, history, top_k)
        return self._keyword_similarity_search(query, history, top_k)

    def get_recommendation_stats(
        self, incident_history: list[dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        """Compute statistics on how well recommendations matched officer actions.

        Returns:
            Dict with: total, responded, accuracy, action_counts.
        """
        history = incident_history or self._load_history()
        total = len(history)
        responded = sum(1 for i in history if i.get("officer_action"))
        matches = sum(
            1 for i in history
            if i.get("officer_action")
            and i.get("recommended_action", "").upper()
               == i.get("officer_action", "").upper()
        )
        accuracy = round(matches / responded, 4) if responded else 0.0

        action_counts: dict[str, int] = {}
        for inc in history:
            action = inc.get("recommended_action", "unknown") or "unknown"
            action_counts[action] = action_counts.get(action, 0) + 1

        return {
            "total_incidents": total,
            "responded": responded,
            "recommendation_accuracy": accuracy,
            "matched_actions": matches,
            "action_distribution": action_counts,
        }

    def generate_learning_context(
        self,
        similar_incidents: list[dict[str, Any]],
        use_llm: bool = False,
    ) -> str:
        """Format similar incidents into a prompt-friendly context string.

        Args:
            similar_incidents: Output of :meth:`get_similar_incidents`.
            use_llm:           If True, enrich the context summary using Groq.

        Returns:
            A plain-text context string for injection into the reasoning prompt.
        """
        if not similar_incidents:
            return "No similar historical incidents found."

        lines = ["Historical context from similar incidents:"]
        for i, inc in enumerate(similar_incidents[:5], 1):
            rec = inc.get("recommended_action") or "N/A"
            actual = inc.get("officer_action") or "N/A"
            outcome = inc.get("final_outcome") or "N/A"
            summary = inc.get("perception_summary") or ""
            fp = "FALSE POSITIVE" if inc.get("is_false_positive") else "confirmed"
            lines.append(
                f"  {i}. [{fp}] Recommended={rec}, Actual={actual}, "
                f"Outcome={outcome}. Summary: {summary[:80]}"
            )

        base_context = "\n".join(lines)

        if use_llm and self._groq_client:
            return self._enrich_with_llm(base_context, similar_incidents)
        return base_context

    def update_from_feedback(self, incident_id: str) -> None:
        """Pull latest Telegram feedback for an incident and update the ML model."""
        if self._db is None:
            logger.warning("[LEARNING] No DB — cannot update from feedback for %s", incident_id)
            return

        incident = self._db.get_incident(incident_id)
        if not incident:
            logger.warning("[LEARNING] Incident %s not found for feedback update", incident_id)
            return

        feedback_type = incident.get("telegram_feedback")
        label = {"confirmed": 1, "false_alarm": 0}.get(feedback_type or "", -1)
        if label == -1:
            logger.debug("[LEARNING] Skipping ambiguous feedback '%s' for ML", feedback_type)
            return

        ml = _get_ml_model()
        if ml is None:
            return

        ml.update(incident, {"label": label})

        # Every 5 samples, do a full retrain for a clean accuracy estimate
        if ml._samples_seen > 10 and ml._samples_seen % 5 == 0:
            training_data = self._db.get_training_data(min_samples=10)
            if training_data:
                result = ml.train_initial(training_data)
                acc = result.get("accuracy", 0.0)
                self._db.log_ml_accuracy(acc, ml._samples_seen)
                logger.info(f"[LEARNING] Model retrained. Accuracy: {acc:.1%}")

        stats = ml.get_stats()
        logger.info(f"[LEARNING] Model updated. Samples: {stats['samples_seen']}  Accuracy: {stats['accuracy']:.1%}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_history(self) -> list[dict[str, Any]]:
        if self._db is None:
            return []
        try:
            return self._db.get_incidents_for_learning(limit=200)
        except Exception as exc:
            logger.warning("[LEARNING] Could not load from DB: %s", exc)
            return []

    def _tfidf_similarity(
        self,
        query: str,
        history: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        docs = [_incident_to_text(i) for i in history]
        corpus = [query] + docs
        try:
            vec = TfidfVectorizer(min_df=1, stop_words="english")
            tfidf_matrix = vec.fit_transform(corpus)
            scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
            indices = scores.argsort()[::-1][:top_k]
            return [history[int(idx)] for idx in indices if scores[int(idx)] > 0]
        except Exception as exc:
            logger.debug("[LEARNING] TF-IDF failed: %s – falling back to keyword", exc)
            return self._keyword_similarity_search(query, history, top_k)

    def _keyword_similarity_search(
        self,
        query: str,
        history: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        scored = [
            (i, _keyword_similarity(query, _incident_to_text(i)))
            for i in history
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [item for item, score in scored[:top_k] if score > 0]

    def _enrich_with_llm(
        self,
        base_context: str,
        incidents: list[dict[str, Any]],
    ) -> str:
        prompt = (
            f"Summarise the following historical security incident patterns in 2-3 "
            f"sentences to help a reasoning agent make a better decision:\n\n"
            f"{base_context}\n\n"
            f'Respond in JSON: {{"learning_summary": "your summary"}}'
        )
        try:
            response = self._groq_client.chat.completions.create(
                model=_GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.2,
                max_tokens=256,
            )
            data = json.loads(response.choices[0].message.content)
            enriched = data.get("learning_summary", "")
            if enriched:
                return f"{base_context}\n\nLearning insight: {enriched}"
        except Exception as exc:
            logger.debug("[LEARNING] LLM enrichment failed: %s", exc)
        return base_context


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.DEBUG)

    mock_history = [
        {
            "id": "INC-001",
            "detections": [{"label": "gun"}],
            "perception_summary": "Armed individual at entrance",
            "recommended_action": "DISPATCH_OFFICERS",
            "officer_action": "DISPATCH_OFFICERS",
            "final_outcome": "RESOLVED",
            "is_false_positive": False,
        },
        {
            "id": "INC-002",
            "detections": [{"label": "knife"}],
            "perception_summary": "Knife threat in corridor",
            "recommended_action": "DISPATCH_OFFICERS",
            "officer_action": "ISSUE_VERBAL_WARNING",
            "final_outcome": "DE_ESCALATED",
            "is_false_positive": False,
        },
    ]

    agent = LearningAgent()

    current = {"weapon": {"label": "gun"}, "emotion": {"label": "angry"}, "tone": {"tone": "threat"}}
    similar = agent.get_similar_incidents(current, top_k=3, incident_history=mock_history)
    print("Similar incidents:", len(similar))

    context = agent.generate_learning_context(similar)
    print(context)

    stats = agent.get_recommendation_stats(incident_history=mock_history)
    print(json.dumps(stats, indent=2))
