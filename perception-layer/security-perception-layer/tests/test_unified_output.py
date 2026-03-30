from datetime import datetime, timezone

from app.schemas import AudioDetection, EmotionDetection, PerceptionRequest, UniformDetection, WeaponDetection
from app.services.fusion import build_fused_event, build_unified_output


def test_unified_output_builds_high_signal_decision() -> None:
    request = PerceptionRequest(
        source_id="bodycam_01",
        timestamp=datetime.now(timezone.utc),
    )
    weapon = WeaponDetection(label="gun", confidence=0.99, bounding_boxes=[], backend="tf2_saved_model")
    uniform = UniformDetection(uniform_present=False, confidence=0.08, bounding_boxes=[], backend="tf_frozen_graph")
    emotion = EmotionDetection(label="neutral", confidence=0.87, face_count=2)
    audio = AudioDetection(
        tone="abnormal",
        confidence=0.68,
        speech_present=True,
        acoustic_events=["speech_like_activity"],
        transcript="someone pulls a gun",
        keyword_flags=["threat_language"],
    )

    fused = build_fused_event(request, weapon, uniform, emotion, audio)
    unified = build_unified_output(request, weapon, uniform, emotion, audio, fused)

    assert unified.decision.threat_level in {"high", "critical"}
    assert "weapon_presence" in unified.decision.anomaly_type
    assert "dispatch_alert" in unified.decision.recommended_response


def test_uniform_presence_suppresses_weapon_decision() -> None:
    request = PerceptionRequest(
        source_id="bodycam_02",
        timestamp=datetime.now(timezone.utc),
    )
    weapon = WeaponDetection(label="gun", confidence=0.91, bounding_boxes=[], backend="ultralytics_yolo")
    uniform = UniformDetection(
        uniform_present=True,
        confidence=0.88,
        bounding_boxes=[],
        backend="tf_frozen_graph",
        class_evidence=[{"label": "uniform", "confidence": 0.88, "frame_index": 12}],
    )
    emotion = EmotionDetection(label="neutral", confidence=0.7, face_count=1)
    audio = AudioDetection(
        tone="calm",
        confidence=0.4,
        speech_present=False,
        acoustic_events=[],
        transcript=None,
        keyword_flags=[],
    )

    fused = build_fused_event(request, weapon, uniform, emotion, audio)
    unified = build_unified_output(request, weapon, uniform, emotion, audio, fused)

    assert fused.weapon_detected == "unarmed"
    assert "weapon_suppressed_uniformed_personnel" in fused.risk_hints
    assert unified.traits.weapon_suppressed_due_to_uniform is True
    assert unified.traits.raw_weapon_detected == "gun"
    assert unified.decision.threat_level == "low"
