from app.schemas import PerceptionRequest, VideoInput
from app.services.models.weapon import WeaponDetectionAdapter


def test_placeholder_backend_is_used_without_model_path() -> None:
    adapter = WeaponDetectionAdapter()

    result = adapter._infer_placeholder(
        PerceptionRequest(
            source_id="cam_01",
            incident_context="knife seen near entrance",
            video=VideoInput(stream_type="cctv"),
        )
    )

    assert result.label == "knife"
    assert result.backend == "placeholder"
