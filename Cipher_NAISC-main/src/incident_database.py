"""
SQLite-based incident database for Cipher_NAISC.

Stores incident details, detections, reasoning outputs, officer responses,
and feedback labels for the AI learning loop.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "incidents.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS incidents (
    id TEXT PRIMARY KEY,
    timestamp TEXT,
    detections TEXT,
    perception_summary TEXT,
    recommended_action TEXT,
    reasoning_confidence REAL,
    officer_action TEXT,
    final_outcome TEXT,
    feedback TEXT,
    is_false_positive INTEGER DEFAULT 0,
    agent_reports TEXT,
    created_at TEXT,
    updated_at TEXT
);
"""

_ML_ACCURACY_LOG_SCHEMA = """
CREATE TABLE IF NOT EXISTS ml_accuracy_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    accuracy REAL NOT NULL,
    samples_seen INTEGER NOT NULL
);
"""

_MIGRATIONS = [
    "ALTER TABLE incidents ADD COLUMN agent_reports TEXT",
    "ALTER TABLE incidents ADD COLUMN telegram_feedback TEXT",
    "ALTER TABLE incidents ADD COLUMN recommendation_feedback TEXT",
    "ALTER TABLE incidents ADD COLUMN feedback_timestamp TEXT",
    "ALTER TABLE incidents ADD COLUMN feedback_source TEXT",
]


class IncidentDatabase:
    """Persistent SQLite store for security incidents and officer feedback."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path) if db_path else _DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        logger.info("[DB] Incident database ready at %s", self.db_path)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
            conn.executescript(_ML_ACCURACY_LOG_SCHEMA)
            for migration in _MIGRATIONS:
                try:
                    conn.execute(migration)
                except Exception:
                    pass  # column already exists

    def _row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        for field in ("detections", "agent_reports"):
            if d.get(field) and isinstance(d[field], str):
                try:
                    d[field] = json.loads(d[field])
                except json.JSONDecodeError:
                    pass
        d["is_false_positive"] = bool(d.get("is_false_positive", 0))
        return d

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_incident(
        self,
        perception_result: dict[str, Any],
        reasoning_result: dict[str, Any] | None = None,
        agent_reports: dict[str, Any] | None = None,
    ) -> str:
        """Insert a new incident and return its generated ID."""
        incident_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        detections_json = json.dumps(perception_result.get("detections", perception_result))
        perception_summary = perception_result.get("summary", "")
        recommended_action = ""
        confidence = 0.0

        if reasoning_result:
            recommended_action = reasoning_result.get("course_of_action", reasoning_result.get("action", ""))
            confidence = float(reasoning_result.get("confidence", 0.0))
            if not perception_summary:
                perception_summary = reasoning_result.get("summary", "")

        agent_reports_json = json.dumps(agent_reports) if agent_reports else None

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO incidents
                    (id, timestamp, detections, perception_summary,
                     recommended_action, reasoning_confidence,
                     agent_reports, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    incident_id,
                    perception_result.get("timestamp", now),
                    detections_json,
                    perception_summary,
                    recommended_action,
                    confidence,
                    agent_reports_json,
                    now,
                    now,
                ),
            )

        logger.info("[DB] Created incident %s", incident_id)
        return incident_id

    def update_officer_response(
        self,
        incident_id: str,
        officer_action: str,
        outcome: str,
        feedback: str,
        is_false_positive: bool = False,
    ) -> bool:
        """Update an existing incident with officer response data."""
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE incidents
                SET officer_action = ?,
                    final_outcome  = ?,
                    feedback       = ?,
                    is_false_positive = ?,
                    updated_at     = ?
                WHERE id = ?
                """,
                (officer_action, outcome, feedback, int(is_false_positive), now, incident_id),
            )
        updated = cursor.rowcount > 0
        if updated:
            logger.info("[DB] Updated officer response for incident %s", incident_id)
        else:
            logger.warning("[DB] Incident %s not found for update", incident_id)
        return updated

    def get_incident(self, incident_id: str) -> dict[str, Any] | None:
        """Retrieve a single incident by ID."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM incidents WHERE id = ?", (incident_id,)
            ).fetchone()
        return self._row_to_dict(row) if row else None

    def get_recent_incidents(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return the most recently created incidents."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM incidents ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get_incidents_for_learning(self, limit: int = 200) -> list[dict[str, Any]]:
        """Return confirmed (non-false-positive) incidents for model training."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM incidents
                WHERE is_false_positive = 0
                  AND officer_action IS NOT NULL
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def record_telegram_feedback(
        self,
        incident_id: str,
        feedback_type: str | None,
        rec_feedback: str | None,
    ) -> bool:
        """Record Telegram inline-keyboard feedback from an officer."""
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE incidents
                SET telegram_feedback        = COALESCE(?, telegram_feedback),
                    recommendation_feedback  = COALESCE(?, recommendation_feedback),
                    feedback_timestamp       = ?,
                    feedback_source          = 'telegram',
                    is_false_positive        = CASE
                        WHEN ? = 'false_alarm' THEN 1
                        WHEN ? = 'confirmed'   THEN 0
                        ELSE is_false_positive
                    END,
                    updated_at               = ?
                WHERE id = ?
                """,
                (feedback_type, rec_feedback, now,
                 feedback_type, feedback_type, now, incident_id),
            )
        updated = cursor.rowcount > 0
        if updated:
            logger.info("[DB] Telegram feedback recorded for %s: %s / rec=%s",
                        incident_id, feedback_type, rec_feedback)
        return updated

    def get_training_data(self, min_samples: int = 10) -> list[dict[str, Any]]:
        """Return feedback-labelled incidents formatted as ML training samples."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM incidents
                WHERE telegram_feedback IN ('confirmed', 'false_alarm')
                ORDER BY feedback_timestamp DESC
                """
            ).fetchall()

        records = [self._row_to_dict(r) for r in rows]
        if len(records) < min_samples:
            return []

        samples: list[dict[str, Any]] = []
        for inc in records:
            detections = inc.get("detections") or {}
            if isinstance(detections, list) and detections:
                det = detections[0]
            elif isinstance(detections, dict):
                det = detections
            else:
                continue

            if not isinstance(det, dict):
                continue

            weapon = det.get("weapon", {}) or {}
            emotion = det.get("emotion", {}) or {}
            tone = det.get("tone", {}) or {}
            uniform = det.get("uniform", {}) or {}

            samples.append({
                "features": {
                    "weapon_conf":  float(weapon.get("confidence", 0)),
                    "emotion_label": emotion.get("label", "neutral"),
                    "tone_label":   tone.get("label", tone.get("tone", "calm")),
                    "has_uniform":  int(bool(uniform.get("present", False))),
                    "threat_reasons": inc.get("perception_summary", ""),
                },
                "label": 1 if inc["telegram_feedback"] == "confirmed" else 0,
                "recommended_action": inc.get("recommended_action", "REVIEW_FOOTAGE"),
                "action_was_good": inc.get("recommendation_feedback") == "good",
                "timestamp": inc.get("feedback_timestamp") or inc.get("created_at", ""),
            })

        return samples

    def get_feedback_summary(self) -> dict[str, Any]:
        """Return counts and rates of all Telegram feedback received."""
        with self._connect() as conn:
            total     = conn.execute("SELECT COUNT(*) FROM incidents WHERE telegram_feedback IS NOT NULL").fetchone()[0]
            confirmed = conn.execute("SELECT COUNT(*) FROM incidents WHERE telegram_feedback = 'confirmed'").fetchone()[0]
            false_al  = conn.execute("SELECT COUNT(*) FROM incidents WHERE telegram_feedback = 'false_alarm'").fetchone()[0]
            partial   = conn.execute("SELECT COUNT(*) FROM incidents WHERE telegram_feedback = 'partial'").fetchone()[0]
            good_rec  = conn.execute("SELECT COUNT(*) FROM incidents WHERE recommendation_feedback = 'good'").fetchone()[0]
            bad_rec   = conn.execute("SELECT COUNT(*) FROM incidents WHERE recommendation_feedback = 'bad'").fetchone()[0]

        rec_rated = good_rec + bad_rec
        return {
            "total":                       total,
            "confirmed":                   confirmed,
            "false_alarm":                 false_al,
            "partial":                     partial,
            "good_rec":                    good_rec,
            "bad_rec":                     bad_rec,
            "false_positive_rate":         round(false_al / total, 4) if total else 0.0,
            "recommendation_approval_rate": round(good_rec / rec_rated, 4) if rec_rated else 0.0,
        }

    def log_ml_accuracy(self, accuracy: float, samples_seen: int) -> None:
        """Append an accuracy snapshot; keeps only the last 20 entries."""
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO ml_accuracy_log (timestamp, accuracy, samples_seen) VALUES (?, ?, ?)",
                (now, accuracy, samples_seen),
            )
            conn.execute(
                "DELETE FROM ml_accuracy_log WHERE id NOT IN "
                "(SELECT id FROM ml_accuracy_log ORDER BY id DESC LIMIT 20)"
            )

    def get_ml_accuracy_history(self) -> list[dict[str, Any]]:
        """Return up to the last 20 accuracy log entries, oldest first."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT timestamp, accuracy, samples_seen FROM ml_accuracy_log ORDER BY id ASC"
            ).fetchall()
        return [dict(r) for r in rows]

    def get_analytics(self) -> dict[str, Any]:
        """Return aggregate statistics for the analytics dashboard."""
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) FROM incidents").fetchone()[0]

            # Count as false positive if flagged by officer OR Telegram feedback says false_alarm
            false_positives = conn.execute(
                "SELECT COUNT(*) FROM incidents WHERE is_false_positive = 1 OR telegram_feedback = 'false_alarm'"
            ).fetchone()[0]

            responded = conn.execute(
                "SELECT COUNT(*) FROM incidents WHERE officer_action IS NOT NULL"
            ).fetchone()[0]

            # Recommendation accuracy: good recommendations out of all rated incidents
            good_rec = conn.execute(
                "SELECT COUNT(*) FROM incidents WHERE recommendation_feedback = 'good'"
            ).fetchone()[0]
            total_with_feedback = conn.execute(
                "SELECT COUNT(*) FROM incidents WHERE telegram_feedback IS NOT NULL OR recommendation_feedback IS NOT NULL"
            ).fetchone()[0]
            rec_accuracy = round(good_rec / total_with_feedback, 4) if total_with_feedback else 0.0

            # Incidents per day (last 30 days)
            daily_rows = conn.execute(
                """
                SELECT DATE(created_at) as day, COUNT(*) as count
                FROM incidents
                WHERE created_at >= datetime('now', '-30 days')
                GROUP BY day
                ORDER BY day
                """
            ).fetchall()

            # Action distribution
            action_rows = conn.execute(
                """
                SELECT recommended_action, COUNT(*) as count
                FROM incidents
                WHERE recommended_action IS NOT NULL AND recommended_action != ''
                GROUP BY recommended_action
                """
            ).fetchall()

            # Most common actual threat — extract weapon/emotion label from detections JSON
            detection_rows = conn.execute(
                "SELECT detections, perception_summary FROM incidents WHERE detections IS NOT NULL LIMIT 200"
            ).fetchall()

        fp_rate = round(false_positives / total, 4) if total else 0.0

        # Count weapon labels from stored detection blobs
        threat_counts: dict[str, int] = {}
        for row in detection_rows:
            try:
                det = json.loads(row[0]) if isinstance(row[0], str) else row[0]
                if isinstance(det, list) and det:
                    det = det[0]
                if not isinstance(det, dict):
                    continue
                weapon = det.get("weapon", {}) or {}
                label = str(weapon.get("label", "")).strip().lower()
                if label and label not in ("unarmed", "unknown", "unknown_object", ""):
                    threat_counts[label] = threat_counts.get(label, 0) + 1
                else:
                    emotion = det.get("emotion", {}) or {}
                    em = str(emotion.get("label", "")).strip().lower()
                    if em and em not in ("unknown", "neutral", ""):
                        key = f"{em} person"
                        threat_counts[key] = threat_counts.get(key, 0) + 1
            except Exception:
                continue

        # Fallback: scan perception_summary text for known threat keywords
        if not threat_counts:
            _kw_map = [
                ("gun", "Gun"), ("firearm", "Gun"), ("pistol", "Gun"), ("rifle", "Gun"),
                ("knife", "Knife"), ("blade", "Knife"), ("scissors", "Scissors"),
                ("bat", "Baseball Bat"), ("bottle", "Bottle"),
                ("angry", "Angry Person"), ("aggressive", "Aggressive Person"),
                ("hostile", "Hostile Person"), ("threatening", "Threatening Person"),
            ]
            for row in detection_rows:
                summary = str(row[1] or "").lower()
                if not summary:
                    continue
                for kw, display in _kw_map:
                    if kw in summary:
                        threat_counts[display] = threat_counts.get(display, 0) + 1
                        break

        if threat_counts:
            raw_label = max(threat_counts, key=lambda k: threat_counts[k])
            # Capitalise each word ("gun" → "Gun", "angry person" → "Angry Person")
            most_common_threat = raw_label.title()
        else:
            most_common_threat = "Armed Suspect"

        return {
            "total_incidents": total,
            "false_positive_count": false_positives,
            "false_positive_rate": fp_rate,
            "responded_count": responded,
            "recommendation_accuracy": rec_accuracy,
            "incidents_per_day": [dict(r) for r in daily_rows],
            "action_distribution": [dict(r) for r in action_rows],
            "most_common_threat": most_common_threat,
        }


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    db = IncidentDatabase(":memory:")

    fake_perception = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "detections": [{"label": "gun", "confidence": 0.92}],
        "summary": "Gun detected in frame",
    }
    fake_reasoning = {
        "course_of_action": "DISPATCH_OFFICERS",
        "confidence": 0.87,
        "summary": "Armed individual spotted – dispatching officers.",
    }

    iid = db.create_incident(fake_perception, fake_reasoning)
    print(f"Created incident: {iid}")

    db.update_officer_response(iid, "DISPATCHED", "RESOLVED", "Confirmed real threat", False)
    print(db.get_incident(iid))
    print(db.get_analytics())
