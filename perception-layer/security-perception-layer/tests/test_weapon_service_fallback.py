from app.schemas import PerceptionRequest, VideoInput
from app.services.models.weapon import WeaponDetectionAdapter


def test_placeholder_is_used_when_remote_and_local_models_are_unavailable() -> None:
    adapter = WeaponDetectionAdapter()
    adapter.legacy_service_url = "http://127.0.0.1:9999"
    adapter.model = None

    result = adapter._infer_placeholder(
        PerceptionRequest(
            source_id="cam_01",
            incident_context="gun reported",
            video=VideoInput(stream_type="cctv", uri="C:\\missing.jpg"),
        )
    )

    assert result.label == "gun"
    assert result.backend == "placeholder"
