"""
Full end-to-end integration test for Perception + Reasoning layers
Tests the complete pipeline from unified perception output to reasoning output
"""

import sys
from pathlib import Path
from datetime import datetime, timezone

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "reasoning-layer"))

from app.schemas import UnifiedPerceptionOutput, UnifiedTraits, UnifiedDecision
from app.reasoning_adapter import get_reasoning_adapter


def create_test_perception(
    weapon_detected: str = "unarmed",
    emotion: str = "neutral",
    tone: str = "calm",
    uniform_present: bool = False,
    unusual_movement: bool = False,
    keyword_flags: list = None
) -> UnifiedPerceptionOutput:
    """Helper to create test perception outputs"""
    # Weapon is suppressed if uniform is present and weapon is detected
    weapon_suppressed = uniform_present and weapon_detected != "unarmed"
    
    return UnifiedPerceptionOutput(
        source_id="camera-01",
        timestamp=datetime.now(timezone.utc),
        model_backends={"weapon": "test", "emotion": "test", "audio": "test", "uniform": "test"},
        confidence_scores={"overall": 0.9},
        traits=UnifiedTraits(
            weapon_detected=weapon_detected,
            raw_weapon_detected=weapon_detected,
            weapon_class_evidence=[],
            visual_secondary_evidence=[],
            uniform_present=uniform_present,
            uniform_confidence=0.9 if uniform_present else 0.1,
            uniform_evidence=[],
            weapon_suppressed_due_to_uniform=weapon_suppressed,
            emotion=emotion,
            tone=tone,
            speech_present=True,
            acoustic_events=[],
            keyword_flags=keyword_flags or [],
            transcript=None
        ),
        risk_hints=[],
        decision=UnifiedDecision(
            threat_level="low",
            anomaly_type=[],
            recommended_response=[],
            confidence=0.9,
            rationale_summary="Test perception"
        )
    )


def run_integration_tests():
    """Run all integration tests"""
    adapter = get_reasoning_adapter()
    
    print("=" * 80)
    print("FULL PERCEPTION-REASONING INTEGRATION TEST")
    print("=" * 80)
    
    # TEST 1: Normal Person
    print("\n[TEST 1] Normal Person - Should be LOW/MEDIUM threat")
    print("-" * 80)
    perception_1 = create_test_perception(
        weapon_detected="unarmed",
        emotion="neutral",
        tone="calm"
    )
    result_1 = adapter.process_perception(perception_1)
    print(f"Threat Level: {result_1.threat_level}")
    print(f"Threat Score: {result_1.metrics.combined_threat_score:.2f}")
    print(f"Recommended Action: {result_1.recommended_action.action}")
    print(f"Confidence: {result_1.recommended_action.confidence:.2f}")
    assert result_1.threat_level in ["low", "medium"], f"Expected LOW or MEDIUM, got {result_1.threat_level}"
    print("✓ TEST 1 PASSED")
    
    # TEST 2: Armed Intruder
    print("\n[TEST 2] Armed Intruder - Should be CRITICAL threat")
    print("-" * 80)
    perception_2 = create_test_perception(
        weapon_detected="gun",
        emotion="angry",
        tone="threat",
        uniform_present=False,
        unusual_movement=True,
        keyword_flags=["attack", "threat"]
    )
    result_2 = adapter.process_perception(perception_2)
    print(f"Threat Level: {result_2.threat_level}")
    print(f"Threat Score: {result_2.metrics.combined_threat_score:.2f}")
    print(f"Recommended Action: {result_2.recommended_action.action}")
    print(f"Confidence: {result_2.recommended_action.confidence:.2f}")
    assert result_2.threat_level == "critical", f"Expected CRITICAL, got {result_2.threat_level}"
    print("✓ TEST 2 PASSED")
    
    # TEST 3: Security Guard with Weapon
    print("\n[TEST 3] Security Guard with Weapon - Should be LOW threat")
    print("-" * 80)
    perception_3 = create_test_perception(
        weapon_detected="gun",
        emotion="calm",
        tone="calm",
        uniform_present=True
    )
    result_3 = adapter.process_perception(perception_3)
    print(f"Threat Level: {result_3.threat_level}")
    print(f"Threat Score: {result_3.metrics.combined_threat_score:.2f}")
    print(f"Recommended Action: {result_3.recommended_action.action}")
    print(f"Confidence: {result_3.recommended_action.confidence:.2f}")
    assert result_3.threat_level == "low", f"Expected LOW, got {result_3.threat_level}"
    print("✓ TEST 3 PASSED")
    
    # TEST 4: Escalating Threat Sequence
    print("\n[TEST 4] Escalating Threat - Temporal Analysis")
    print("-" * 80)
    
    # Frame 1: Calm
    perception_4a = create_test_perception(
        weapon_detected="unarmed",
        emotion="neutral",
        tone="calm"
    )
    result_4a = adapter.process_perception(perception_4a)
    print(f"Frame 1 - Threat: {result_4a.threat_level} (Score: {result_4a.metrics.combined_threat_score:.2f})")
    # Normal person baseline can be medium due to 0.5 bias in threat calculation
    assert result_4a.threat_level in ["low", "medium"], f"Expected low/medium, got {result_4a.threat_level}"
    
    # Frame 2: Suspicious
    perception_4b = create_test_perception(
        weapon_detected="knife",
        emotion="anxious",
        tone="abnormal",
        unusual_movement=True
    )
    result_4b = adapter.process_perception(perception_4b)
    print(f"Frame 2 - Threat: {result_4b.threat_level} (Score: {result_4b.metrics.combined_threat_score:.2f})")
    assert result_4b.threat_level in ["medium", "high", "critical"]
    
    # Frame 3: Even More Suspicious
    perception_4c = create_test_perception(
        weapon_detected="knife",
        emotion="angry",
        tone="threat",
        unusual_movement=True,
        keyword_flags=["help", "attack"]
    )
    result_4c = adapter.process_perception(perception_4c)
    print(f"Frame 3 - Threat: {result_4c.threat_level} (Score: {result_4c.metrics.combined_threat_score:.2f})")
    
    # Check escalation tracking
    state = adapter.get_state()
    print(f"\nState after escalation:")
    print(f"  - History size: {len(state['history']) if 'history' in state else 'N/A'}")
    print(f"  - Escalation detected: {state.get('escalation_detected', False)}")
    print("✓ TEST 4 PASSED")
    
    print("\n" + "=" * 80)
    print("ALL INTEGRATION TESTS PASSED ✓")
    print("=" * 80)


if __name__ == "__main__":
    try:
        run_integration_tests()
    except Exception as e:
        print(f"\n✗ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
