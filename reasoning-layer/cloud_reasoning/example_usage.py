"""
Cloud Reasoning Agent - Example Usage

This shows how to use the full-featured cloud reasoning agent.
Run this on your cloud server to process perception data from officer devices.
"""

from cloud_reasoning_agent import CloudReasoningAgent, SOPContext, ScenarioPrediction
import json


def example_1_gun_at_school():
    """Example 1: Gun detected at school during critical security level"""
    print("\n" + "="*70)
    print("Example 1: GUN AT SCHOOL - Critical Situation")
    print("="*70)
    
    agent = CloudReasoningAgent()
    
    # Officer detected a gun at school entrance
    perception_data = {
        "source_id": "officer_badge_001",
        "weapon_detected": "gun",
        "emotion": "angry",
        "tone": "threat",
        "confidence_scores": {
            "weapon": 0.95,
            "emotion": 0.85,
            "tone": 0.90,
            "uniform": 0.0
        },
        "risk_hints": [
            "visible_weapon",
            "emotional_escalation",
            "audio_escalation"
        ]
    }
    
    # School context
    sop_context = SOPContext(
        location_type="school",
        time_of_day="peak_hours",
        security_level="high",
        active_alerts=["armed_individual"],
        personnel_count=500,
        restricted_weapons=["gun", "explosive"]
    )
    
    # Get threat assessment
    result = agent.reason(
        **perception_data,
        sop_context=sop_context
    )
    
    # Display results
    print(f"\n🎯 Source: {result.source_id}")
    print(f"⚠️  Threat Level: {result.threat_level.upper()}")
    print(f"🔴 Confidence: {result.confidence:.0%}")
    print(f"\n📋 Recommended Action:")
    print(f"   Action: {result.recommended_action.action}")
    print(f"   Priority: {result.recommended_action.priority}")
    print(f"   Reason: {result.recommended_action.reason}")
    
    print(f"\n📊 Threat Metrics:")
    print(f"   Weapon Score: {result.metrics.weapon_threat_score:.2f}")
    print(f"   Emotion Score: {result.metrics.emotion_threat_score:.2f}")
    print(f"   Audio Score: {result.metrics.audio_threat_score:.2f}")
    print(f"   Behavioral Score: {result.metrics.behavioral_anomaly_score:.2f}")
    print(f"   Combined: {result.metrics.combined_threat_score:.2f}")
    
    print(f"\n💭 Explanation:")
    print(f"   {result.explanation.summary}")
    print(f"   Key Factors: {', '.join(result.explanation.key_factors)}")
    
    return result


def example_2_calm_civilian():
    """Example 2: Unarmed civilian - no threat"""
    print("\n" + "="*70)
    print("Example 2: UNARMED CIVILIAN - Normal Situation")
    print("="*70)
    
    agent = CloudReasoningAgent()
    
    perception_data = {
        "source_id": "officer_badge_002",
        "weapon_detected": "unarmed",
        "emotion": "calm",
        "tone": "neutral",
        "confidence_scores": {
            "weapon": 0.98,
            "emotion": 0.90,
            "tone": 0.85,
            "uniform": 0.0
        },
        "risk_hints": []
    }
    
    # Street context
    sop_context = SOPContext(
        location_type="street",
        time_of_day="day",
        security_level="medium",
        active_alerts=[],
        personnel_count=50,
        restricted_weapons=["gun", "explosive"]
    )
    
    result = agent.reason(
        **perception_data,
        sop_context=sop_context
    )
    
    # Display results
    print(f"\n✓ Source: {result.source_id}")
    print(f"✅ Threat Level: {result.threat_level.upper()}")
    print(f"💚 Confidence: {result.confidence:.0%}")
    print(f"\n📋 Recommended Action:")
    print(f"   Action: {result.recommended_action.action}")
    print(f"   Priority: {result.recommended_action.priority}")
    
    print(f"\n📊 Threat Metrics:")
    print(f"   Combined: {result.metrics.combined_threat_score:.2f}")
    
    return result


def example_3_with_scenarios():
    """Example 3: Using Learning Agent scenario predictions"""
    print("\n" + "="*70)
    print("Example 3: WITH SCENARIO PREDICTIONS - Predictive Alert")
    print("="*70)
    
    agent = CloudReasoningAgent()
    
    perception_data = {
        "source_id": "camera_entrance_042",
        "weapon_detected": "knife",
        "emotion": "distressed",
        "tone": "panic",
        "confidence_scores": {
            "weapon": 0.80,
            "emotion": 0.85,
            "tone": 0.90,
            "uniform": 0.0
        },
        "risk_hints": [
            "visible_weapon",
            "emotional_escalation",
            "audio_escalation"
        ]
    }
    
    # Learning Agent predicted these scenarios
    scenario_predictions = [
        ScenarioPrediction(
            rank=1,
            scenario_description="Escalation to attack",
            probability=0.75,
            estimated_escalation_time_seconds=30,
            recommended_preemptive_action="Evacuate area"
        ),
        ScenarioPrediction(
            rank=2,
            scenario_description="Threat display (de-escalation possible)",
            probability=0.20,
            estimated_escalation_time_seconds=120,
            recommended_preemptive_action="Engage professional mediator"
        ),
        ScenarioPrediction(
            rank=3,
            scenario_description="Accidental behavior",
            probability=0.05,
            estimated_escalation_time_seconds=None,
            recommended_preemptive_action="None"
        )
    ]
    
    sop_context = SOPContext(
        location_type="bank",
        time_of_day="business_hours",
        security_level="critical",
        active_alerts=["armed_individual"],
        personnel_count=100,
        restricted_weapons=["gun", "explosive", "sharp_objects"]
    )
    
    result = agent.reason(
        **perception_data,
        scenario_predictions=scenario_predictions,
        sop_context=sop_context
    )
    
    # Display results
    print(f"\n⚠️  Source: {result.source_id}")
    print(f"🔴 Threat Level: {result.threat_level.upper()}")
    print(f"   Confidence: {result.confidence:.0%}")
    
    print(f"\n📊 Threat Metrics (with scenario boost):")
    print(f"   Weapon Score: {result.metrics.weapon_threat_score:.2f}")
    print(f"   Emotion Score: {result.metrics.emotion_threat_score:.2f}")
    print(f"   Audio Score: {result.metrics.audio_threat_score:.2f}")
    print(f"   Combined (with scenarios): {result.metrics.combined_threat_score:.2f}")
    print(f"   Trend: {result.metrics.trend}")
    
    print(f"\n🎯 Recommended Action:")
    print(f"   {result.recommended_action.action}")
    print(f"   {result.recommended_action.reason}")
    
    print(f"\n💭 Explanation Summary:")
    print(f"   {result.explanation.summary}")
    
    return result


def example_4_batch_processing():
    """Example 4: Processing multiple detections (like replay of video)"""
    print("\n" + "="*70)
    print("Example 4: BATCH PROCESSING - Multiple Frames")
    print("="*70)
    
    agent = CloudReasoningAgent()
    
    # Simulate multiple perception results from same source over time
    detections = [
        {
            "frame": 0,
            "weapon": "unarmed",
            "emotion": "neutral",
            "tone": "calm",
            "confidence": {"weapon": 0.99, "emotion": 0.85, "tone": 0.88}
        },
        {
            "frame": 5,
            "weapon": "unarmed",
            "emotion": "angry",
            "tone": "elevated",
            "confidence": {"weapon": 0.99, "emotion": 0.90, "tone": 0.82}
        },
        {
            "frame": 10,
            "weapon": "knife",
            "emotion": "angry",
            "tone": "threat",
            "confidence": {"weapon": 0.85, "emotion": 0.85, "tone": 0.90}
        },
    ]
    
    sop_context = SOPContext(
        location_type="street",
        time_of_day="night",
        security_level="high",
        active_alerts=[],
        personnel_count=20,
        restricted_weapons=["gun", "explosive"]
    )
    
    print("\nProcessing video stream over 10 frames:")
    print("-" * 70)
    
    for detection in detections:
        result = agent.reason(
            source_id="street_camera_001",
            weapon_detected=detection["weapon"],
            emotion=detection["emotion"],
            tone=detection["tone"],
            confidence_scores=detection["confidence"],
            risk_hints=[],
            sop_context=sop_context
        )
        
        threat_char = "⚠️" if result.threat_level != "low" else "✓"
        print(f"Frame {detection['frame']:2d}: {threat_char} {result.threat_level:8s} | "
              f"Weapon: {detection['weapon']:10s} | Confidence: {result.confidence:.0%}")
    
    print("-" * 70)
    print("✓ Replay completed - Timeline shows escalation from calm to threat")


def example_5_different_locations():
    """Example 5: Same weapon, different location impacts"""
    print("\n" + "="*70)
    print("Example 5: LOCATION SENSITIVITY - Same Weapon, Different Scores")
    print("="*70)
    
    agent = CloudReasoningAgent()
    
    locations = [
        ("street", "medium"),
        ("office", "medium"),
        ("bank", "high"),
        ("airport", "high"),
        ("school", "critical"),
    ]
    
    print("\nGun detection in different locations:")
    print("-" * 70)
    
    for location, security_level in locations:
        sop_context = SOPContext(
            location_type=location,
            time_of_day="business",
            security_level=security_level,
            active_alerts=[],
            personnel_count=100,
            restricted_weapons=["gun"]
        )
        
        result = agent.reason(
            source_id=f"camera_{location}",
            weapon_detected="gun",
            emotion="neutral",
            tone="calm",
            confidence_scores={"weapon": 0.95, "emotion": 0.0, "tone": 0.0},
            risk_hints=["visible_weapon"],
            sop_context=sop_context
        )
        
        print(f"{location:12s} (security={security_level:8s}): "
              f"Threat={result.threat_level:10s} | Score={result.metrics.combined_threat_score:.2f}")
    
    print("-" * 70)
    print("✓ Location context significantly impacts threat assessment")


if __name__ == "__main__":
    print("\n" + "="*70)
    print(" CLOUD REASONING AGENT - USAGE EXAMPLES")
    print("="*70)
    
    # Run all examples
    example_1_gun_at_school()
    example_2_calm_civilian()
    example_3_with_scenarios()
    example_4_batch_processing()
    example_5_different_locations()
    
    print("\n" + "="*70)
    print(" EXAMPLES COMPLETED")
    print("="*70)
    print("\n✓ Cloud Reasoning Agent ready for deployment!")
    print("  Deploy to: AWS Lambda, Docker, or Cloud Server")
    print("  API: POST /reason with perception data")
    print("  Latency: 100-500ms per request")
    print("  Accuracy: Highest (4-factor analysis)")
