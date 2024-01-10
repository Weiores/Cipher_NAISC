"""
Seed 75 synthetic training samples into the incidents DB, then train the ML model.

Run from project root:
    python learning-layer/seed_training_data.py
"""
from __future__ import annotations

import json
import logging
import random
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "learning-layer"))

from incident_database import IncidentDatabase  # noqa: E402
from ml_model import CipherMLModel              # noqa: E402

random.seed(42)


def _random_ts(days_ago_max: int = 30) -> str:
    delta = timedelta(
        days=random.randint(0, days_ago_max),
        hours=random.randint(0, 23),
        minutes=random.randint(0, 59),
    )
    return (datetime.now(timezone.utc) - delta).isoformat()


def _make_detection(
    weapon_conf: float,
    weapon_label: str,
    emotion: str,
    tone: str,
    has_uniform: bool,
) -> dict:
    return {
        "weapon": {
            "label": weapon_label,
            "confidence": round(weapon_conf, 3),
            "bbox": [],
        },
        "emotion": {
            "label": emotion,
            "confidence": round(random.uniform(0.60, 0.90), 3),
            "face_count": 1,
        },
        "tone": {
            "label": tone,
            "tone": tone,
            "confidence": round(random.uniform(0.50, 0.85), 3),
            "speech_present": tone not in ("calm", "neutral"),
        },
        "uniform": {
            "present": has_uniform,
            "confidence": round(random.uniform(0.70, 0.95), 3),
            "label": "uniform" if has_uniform else "civilian",
        },
    }


# ---------------------------------------------------------------------------
# Sample definitions  (total = 50)
# ---------------------------------------------------------------------------

def _build_samples() -> list[dict]:
    samples: list[dict] = []

    # ── HIGH THREAT — 25 confirmed ──────────────────────────────────────────

    # Group A: gun/knife + angry/fearful + aggressive (10)
    for _ in range(10):
        wc = round(random.uniform(0.70, 0.95), 3)
        samples.append({
            "weapon_conf":  wc,
            "weapon_label": random.choice(["gun", "knife"]),
            "emotion":      random.choice(["angry", "fearful"]),
            "tone":         "aggressive",
            "has_uniform":  False,
            "feedback":     "confirmed",
            "rec_feedback": "good",
            "action":       "DISPATCH_OFFICERS",
            "summary":      f"Armed civilian, aggressive tone — weapon conf {wc:.0%}",
        })

    # Group B: weapon + angry, civilian (8)
    for _ in range(8):
        wc = round(random.uniform(0.60, 0.80), 3)
        samples.append({
            "weapon_conf":  wc,
            "weapon_label": random.choice(["gun", "knife", "bat"]),
            "emotion":      "angry",
            "tone":         random.choice(["aggressive", "threat"]),
            "has_uniform":  False,
            "feedback":     "confirmed",
            "rec_feedback": "good",
            "action":       "DISPATCH_OFFICERS",
            "summary":      f"Angry civilian with weapon — conf {wc:.0%}",
        })

    # Group C: high-conf weapon + fearful, civilian (7)
    for _ in range(7):
        wc = round(random.uniform(0.80, 0.90), 3)
        samples.append({
            "weapon_conf":  wc,
            "weapon_label": random.choice(["gun", "knife"]),
            "emotion":      "fearful",
            "tone":         random.choice(["threat", "tense"]),
            "has_uniform":  False,
            "feedback":     "confirmed",
            "rec_feedback": "good",
            "action":       "DISPATCH_OFFICERS",
            "summary":      f"High-conf weapon + fearful person — conf {wc:.0%}",
        })

    # ── FALSE ALARM — 15 false_alarm ────────────────────────────────────────

    # Group D: low conf, neutral emotion/tone (6)
    for _ in range(6):
        wc = round(random.uniform(0.15, 0.35), 3)
        samples.append({
            "weapon_conf":  wc,
            "weapon_label": "unknown_object",
            "emotion":      "neutral",
            "tone":         "neutral",
            "has_uniform":  random.choice([True, False]),
            "feedback":     "false_alarm",
            "rec_feedback": "bad",
            "action":       "REVIEW_FOOTAGE",
            "summary":      f"Low-conf detection, likely bag/shadow — conf {wc:.0%}",
        })

    # Group E: very low conf, unknown emotion (5)
    for _ in range(5):
        wc = round(random.uniform(0.20, 0.40), 3)
        samples.append({
            "weapon_conf":  wc,
            "weapon_label": "unknown_object",
            "emotion":      "unknown",
            "tone":         "neutral",
            "has_uniform":  False,
            "feedback":     "false_alarm",
            "rec_feedback": "bad",
            "action":       "MONITOR_ONLY",
            "summary":      f"Ambiguous object, no threatening behaviour — conf {wc:.0%}",
        })

    # Group F: very low conf, happy/neutral (4)
    for _ in range(4):
        wc = round(random.uniform(0.10, 0.30), 3)
        samples.append({
            "weapon_conf":  wc,
            "weapon_label": "unknown_object",
            "emotion":      random.choice(["neutral", "happy"]),
            "tone":         "neutral",
            "has_uniform":  False,
            "feedback":     "false_alarm",
            "rec_feedback": "bad",
            "action":       "MONITOR_ONLY",
            "summary":      f"Very low risk — probable false detection — conf {wc:.0%}",
        })

    # ── LOW THREAT — 10 partial ─────────────────────────────────────────────

    # Group G: moderate weapon + angry, no uniform (5)
    for _ in range(5):
        wc = round(random.uniform(0.40, 0.60), 3)
        samples.append({
            "weapon_conf":  wc,
            "weapon_label": random.choice(["bat", "unknown_object"]),
            "emotion":      "angry",
            "tone":         "neutral",
            "has_uniform":  False,
            "feedback":     "partial",
            "rec_feedback": "good",
            "action":       "INCREASE_SURVEILLANCE",
            "summary":      f"Moderate risk — angry civilian with possible object — conf {wc:.0%}",
        })

    # Group H: moderate weapon + fearful, uniformed (5)
    for _ in range(5):
        wc = round(random.uniform(0.30, 0.50), 3)
        samples.append({
            "weapon_conf":  wc,
            "weapon_label": random.choice(["bat", "unknown_object"]),
            "emotion":      "fearful",
            "tone":         "tense",
            "has_uniform":  True,
            "feedback":     "partial",
            "rec_feedback": "good",
            "action":       "ISSUE_VERBAL_WARNING",
            "summary":      f"Officer with unclear object — conf {wc:.0%}",
        })

    # ── EDGED WEAPONS — HIGH THREAT (20) ────────────────────────────────────

    # Group I: scissors/knife/blade — confirmed high-threat (20)
    edged_labels = ["scissors", "knife", "blade", "scissors", "knife"]  # weighted toward scissors
    for _ in range(20):
        wl = random.choice(edged_labels)
        wc = round(random.uniform(0.70, 0.90), 3)
        samples.append({
            "weapon_conf":  wc,
            "weapon_label": wl,
            "emotion":      random.choice(["angry", "fearful", "distressed"]),
            "tone":         random.choice(["aggressive", "threat", "tense"]),
            "has_uniform":  False,
            "feedback":     "confirmed",
            "rec_feedback": "good",
            "action":       "DISPATCH_OFFICERS",
            "summary":      f"Edged weapon ({wl}) confirmed — civilian, aggressive — conf {wc:.0%}",
        })

    # ── EDGED WEAPONS — FALSE ALARM (5) ─────────────────────────────────────

    # Group J: low-conf scissors in benign context (5)
    for _ in range(5):
        wc = round(random.uniform(0.30, 0.45), 3)
        samples.append({
            "weapon_conf":  wc,
            "weapon_label": "scissors",
            "emotion":      random.choice(["neutral", "happy"]),
            "tone":         "neutral",
            "has_uniform":  False,
            "feedback":     "false_alarm",
            "rec_feedback": "bad",
            "action":       "MONITOR_ONLY",
            "summary":      f"Scissors likely craft/office scissors in benign context — conf {wc:.0%}",
        })

    assert len(samples) == 75, f"Expected 75 samples, got {len(samples)}"
    return samples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    db = IncidentDatabase()
    samples = _build_samples()

    logger.info("Seeding %d synthetic training samples…", len(samples))

    inserted = 0
    for s in samples:
        iid = str(uuid.uuid4())
        ts = _random_ts(30)
        feedback_ts = _random_ts(15)
        detection = _make_detection(
            weapon_conf=s["weapon_conf"],
            weapon_label=s["weapon_label"],
            emotion=s["emotion"],
            tone=s["tone"],
            has_uniform=s["has_uniform"],
        )
        is_fp = 1 if s["feedback"] == "false_alarm" else 0
        reasoning_conf = round(min(s["weapon_conf"] * 1.05, 1.0), 3)

        with db._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO incidents
                    (id, timestamp, detections, perception_summary, recommended_action,
                     reasoning_confidence, telegram_feedback, recommendation_feedback,
                     feedback_timestamp, feedback_source, is_false_positive,
                     created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    iid, ts, json.dumps(detection), s["summary"], s["action"],
                    reasoning_conf,
                    s["feedback"], s["rec_feedback"],
                    feedback_ts, "synthetic_seed", is_fp,
                    ts, ts,
                ),
            )
        inserted += 1

    logger.info("Inserted %d rows into incidents table", inserted)

    # ── Pull labeled samples (confirmed + false_alarm only) and train ───────
    training_data = db.get_training_data(min_samples=1)

    if not training_data:
        logger.error(
            "get_training_data() returned empty — "
            "check that telegram_feedback='confirmed'/'false_alarm' rows exist"
        )
        sys.exit(1)

    label_counts = {0: 0, 1: 0}
    for t in training_data:
        label_counts[t["label"]] = label_counts.get(t["label"], 0) + 1
    logger.info(
        "Training data: %d samples (confirmed=%d, false_alarm=%d)",
        len(training_data), label_counts.get(1, 0), label_counts.get(0, 0),
    )

    ml = CipherMLModel()
    result = ml.train_initial(training_data)

    # ── Confusion matrix + report ────────────────────────────────────────────
    try:
        import numpy as np
        from sklearn.metrics import confusion_matrix, classification_report  # noqa: PLC0415

        X_list, y_list = [], []
        for sample in training_data:
            try:
                X_list.append(ml._extract_features_from_sample(sample)[0])
                y_list.append(int(sample["label"]))
            except Exception:
                continue

        X = np.array(X_list)
        y = np.array(y_list)
        X_scaled = ml._scaler.transform(X)
        preds = ml._classifier.predict(X_scaled)

        cm = confusion_matrix(y, preds, labels=[0, 1])

        print()
        print("=" * 55)
        print("  CIPHER ML SEED TRAINING COMPLETE")
        print("=" * 55)
        print(f"  Samples trained : {result['samples']}")
        print(f"  Final accuracy  : {result['accuracy']:.4f}  ({result['accuracy'] * 100:.1f}%)")
        print()
        print("  Confusion matrix  (rows=actual, cols=predicted)")
        print("                  false_alarm  confirmed")
        print(f"  false_alarm     {cm[0, 0]:5d}        {cm[0, 1]:5d}")
        print(f"  confirmed       {cm[1, 0]:5d}        {cm[1, 1]:5d}")
        print()
        report = classification_report(
            y, preds,
            target_names=["false_alarm", "confirmed"],
            zero_division=0,
        )
        print(report)

    except ImportError as exc:
        print(f"\nConfusion matrix skipped (missing dependency: {exc})")
        print(f"Accuracy: {result['accuracy']:.4f}")

    # Log accuracy snapshot to DB
    db.log_ml_accuracy(result["accuracy"], result["samples"])
    logger.info("Accuracy snapshot logged to ml_accuracy_log")

    print("DONE. Seed complete. ML model saved to data/cipher_ml_model.joblib")
    print("  Run the API and check /ml-stats to confirm.")


if __name__ == "__main__":
    main()
