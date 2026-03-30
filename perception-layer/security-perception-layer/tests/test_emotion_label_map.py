from app.services.models.emotion import EmotionDetectionAdapter


def test_emotion_labels_are_normalized() -> None:
    adapter = EmotionDetectionAdapter()

    assert adapter._normalize_label("angry") == "angry"
    assert adapter._normalize_label("fear") == "fearful"
    assert adapter._normalize_label("sad") == "distressed"
    assert adapter._normalize_label("neutral") == "neutral"
