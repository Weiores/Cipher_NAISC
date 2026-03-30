from app.services.models.weapon import WeaponDetectionAdapter


def test_repo_class_ids_map_to_normalized_labels() -> None:
    adapter = WeaponDetectionAdapter()

    assert adapter._map_repo_class_id(1) == "gun"
    assert adapter._map_repo_class_id(5) == "gun"
    assert adapter._map_repo_class_id(6) == "knife"
    assert adapter._map_repo_class_id(99) == "unknown_object"
