from app.services.models.audio import AudioThreatDetectionAdapter


def test_audio_tone_rules_cover_expected_cases() -> None:
    adapter = AudioThreatDetectionAdapter()

    assert adapter._classify_tone(
        speech_ratio=0.0,
        burst_ratio=0.0,
        p95_rms=0.01,
        mean_zcr=0.01,
        peak=0.05,
    )[0] == "unknown"
    assert adapter._classify_tone(
        speech_ratio=0.3,
        burst_ratio=0.2,
        p95_rms=0.3,
        mean_zcr=0.15,
        peak=0.98,
    )[0] == "panic"
