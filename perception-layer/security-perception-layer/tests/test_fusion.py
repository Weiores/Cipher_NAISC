from datetime import datetime, timezone

from app.schemas import AudioDetection, EmotionDetection, PerceptionRequest, UniformDetection, WeaponDetection
from app.services.fusion import build_fused_event


def test_fusion_adds_expected_risk_hints() -> None:
    request = PerceptionRequest(
        source_id="bodycam_07",
        timestamp=datetime.now(timezone.utc),
        incident_context="knife panic",
    )
    weapon = WeaponDetection(label="knife", confidence=0.9, bounding_boxes=[])
    uniform = UniformDetection(uniform_present=False, confidence=0.05, bounding_boxes=[])
    emotion = EmotionDetection(label="fearful", confidence=0.8, face_count=1)
    audio = AudioDetection(
        tone="panic",
        confidence=0.85,
        speech_present=True,
        acoustic_events=["shouting"],
        transcript="help help",
        keyword_flags=["distress_language"],
    )

    fused = build_fused_event(request, weapon, uniform, emotion, audio)

    assert "visible_weapon" in fused.risk_hints
    assert "emotional_escalation" in fused.risk_hints
    assert "audio_escalation" in fused.risk_hints
