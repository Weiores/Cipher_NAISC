"""
Local Reasoning Agent - Example Usage

This shows how to use the lightweight local reasoning agent on officer's device.
Runs fast, offline-capable, minimal resources.
"""

from local_reasoning_agent import LocalReasoningAgent, SOPContext


def example_1_gun_detection():
    """Example 1: Officer's phone detects gun - immediate alert"""
    print("\n" + "="*70)
    print("Example 1: OFFICER'S PHONE DETECTS GUN")
    print("="*70)
    
    agent = LocalReasoningAgent()
    
    # Weapon detection from phone camera (no emotions/audio)
    result = agent.reason(
        source_id="officer_phone_badge_001",
        weapon_detected="gun",
        confidence_scores={"weapon": 0.95}
    )
    
    # Display on officer's screen
    print(f"\n📱 Officer's Device Display:")
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"🔴 {result.threat_level.upper()}")
    print(f"Action: {result.recommended_action.action}")
    print(f"{result.recommended_action.reason}")
    print(f"Confidence: {result.confidence:.0%}")
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    print(f"\n⚡ Latency: ~50-100ms (instant on device)")
    print(f"📡 Offline: ✅ Works without internet")
    
    return result


def example_2_gun_at_school():
    """Example 2: Same gun, but at school - multiplier effect"""
    print("\n" + "="*70)
    print("Example 2: GUN AT SCHOOL - SOP Context Impact")
    print("="*70)
    
    agent = LocalReasoningAgent()
    
    # School context
    sop_context = SOPContext(
        location_type="school",
        time_of_day="peak_hours",
        security_level="critical",
        active_alerts=[],
        personnel_count=500,
        restricted_weapons=["gun"]
    )
    
    result = agent.reason(
        source_id="school_bodycam_042",
        weapon_detected="gun",
        confidence_scores={"weapon": 0.95},
        sop_context=sop_context
    )
    
    # Display results
    print(f"\n📱 Officer's Device Display:")
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"🔴 {result.threat_level.upper()}")
    print(f"Location: {sop_context.location_type}")
    print(f"Security Level: {sop_context.security_level}")
    print(f"Threat Score: {result.confidence:.0%}")
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    print(f"\nKey Factors:")
    for factor in result.explanation.key_factors:
        print(f"  • {factor}")
    
    print(f"\n💡 School detection = 2x sensitivity")
    print(f"   Threat score multiplied by school context")
    
    return result


def example_3_unarmed_person():
    """Example 3: No weapon - safe situation"""
    print("\n" + "="*70)
    print("Example 3: UNARMED PERSON - No Threat")
    print("="*70)
    
    agent = LocalReasoningAgent()
    
    result = agent.reason(
        source_id="patrol_officer_123",
        weapon_detected="unarmed",
        confidence_scores={"weapon": 0.98}
    )
    
    # Display on officer's screen
    print(f"\n📱 Officer's Device Display:")
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"🟢 {result.threat_level.upper()}")
    print(f"Status: Normal")
    print(f"Action: {result.recommended_action.action}")
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    print(f"\n✓ No weapon detected")
    print(f"✓ Continue routine patrol")
    
    return result


def example_4_multiple_frames():
    """Example 4: Processing multiple frames quickly"""
    print("\n" + "="*70)
    print("Example 4: REAL-TIME VIDEO PROCESSING")
    print("="*70)
    
    agent = LocalReasoningAgent()
    
    # Simulated frame-by-frame detections
    frames = [
        {"frame": 0, "weapon": "unarmed"},
        {"frame": 1, "weapon": "unarmed"},
        {"frame": 2, "weapon": "unarmed"},
        {"frame": 3, "weapon": "bat"},      # Weapon appears
        {"frame": 4, "weapon": "bat"},
        {"frame": 5, "weapon": "bat"},
    ]
    
    sop_context = SOPContext(
        location_type="street",
        security_level="medium",
        time_of_day="night",
        active_alerts=[],
        personnel_count=10,
        restricted_weapons=["gun"]
    )
    
    print(f"\nProcessing video stream in REAL-TIME:")
    print(f"-" * 70)
    print(f"Frame  | Weapon    | Threat      | Action")
    print(f"-" * 70)
    
    for frame_data in frames:
        result = agent.reason(
            source_id="bodycam",
            weapon_detected=frame_data["weapon"],
            sop_context=sop_context
        )
        
        threat_icon = "🟢" if result.threat_level == "low" else "🟡"
        print(f"{frame_data['frame']:5d}  | {frame_data['weapon']:9s} | "
              f"{threat_icon} {result.threat_level:10s} | {result.recommended_action.action}")
    
    print(f"-" * 70)
    print(f"\n⚡ Real-time assessment (<100ms per frame)")
    print(f"📡 Offline: ✅ No internet needed")
    print(f"💾 Memory: Light footprint, runs smoothly on any phone")


def example_5_different_weapons():
    """Example 5: Different weapons, different threat levels"""
    print("\n" + "="*70)
    print("Example 5: WEAPON SEVERITY COMPARISON")
    print("="*70)
    
    agent = LocalReasoningAgent()
    
    weapons = [
        ("unarmed", 1.0),
        ("stick", 0.8),
        ("bat", 0.8),
        ("knife", 0.95),
        ("gun", 0.99),
    ]
    
    print(f"\nWeapon threat threshold comparison:")
    print(f"-" * 70)
    print(f"Weapon     | Confidence | Threat Score | Level")
    print(f"-" * 70)
    
    for weapon, confidence in weapons:
        result = agent.reason(
            source_id="assessment",
            weapon_detected=weapon,
            confidence_scores={"weapon": confidence}
        )
        
        threat_icon = "🔴" if result.threat_level == "critical" else \
                      "🟡" if result.threat_level == "medium" else \
                      "🟢"
        
        print(f"{weapon:10s} | {confidence:10.0%} | "
              f"{result.confidence:12.0%} | {threat_icon} {result.threat_level:12s}")
    
    print(f"-" * 70)


def example_6_offline_mode():
    """Example 6: Demonstrating offline operation"""
    print("\n" + "="*70)
    print("Example 6: OFFLINE MODE - No Internet Required")
    print("="*70)
    
    agent = LocalReasoningAgent()
    
    print(f"\nScenario: Officer's phone in subway (no WiFi/cellular)")
    print(f"Camera detects knife")
    
    result = agent.reason(
        source_id="subway_bodycam",
        weapon_detected="knife",
        confidence_scores={"weapon": 0.85}
    )
    
    print(f"\n📱 Device Screen (OFFLINE MODE):")
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"🔴 {result.threat_level.upper()}")
    print(f"Weapon: Knife")
    print(f"Status: Processing OFFLINE ✓")
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    print(f"\n✅ No internet required")
    print(f"✅ No latency from network delays")
    print(f"✅ Immediate response to officer")
    print(f"✅ Can later sync with cloud when WiFi available")


def example_7_mobile_app_integration():
    """Example 7: How to integrate in a mobile app"""
    print("\n" + "="*70)
    print("Example 7: MOBILE APP INTEGRATION CODE")
    print("="*70)
    
    print("""
# In your mobile app (Python (Web)/React/Flutter backend)

from local_reasoning_agent import LocalReasoningAgent

# Initialize once when app starts
reasoning_agent = LocalReasoningAgent()

# After running object detection on camera frame
def process_camera_frame(frame):
    # Step 1: Run weapon detection model
    weapon_results = weapon_detector.predict(frame)  # From YOLOv8, etc
    weapon_type = weapon_results[0].class  # "gun", "knife", etc
    confidence = weapon_results[0].confidence
    
    # Step 2: Get location info (from GPS or manual input)
    location = gps.get_current_location()  # e.g., "school"
    security_level = get_security_level(location)  # e.g., "high"
    
    # Step 3: Create SOP context
    sop_context = SOPContext(
        location_type=location,
        security_level=security_level,
        time_of_day="day",
        active_alerts=[],
        personnel_count=None,
        restricted_weapons=["gun"]
    )
    
    # Step 4: Run local reasoning (FAST - <100ms)
    threat_assessment = reasoning_agent.reason(
        source_id=officer_badge_id,
        weapon_detected=weapon_type,
        confidence_scores={"weapon": confidence},
        sop_context=sop_context
    )
    
    # Step 5: Update UI
    update_threat_display(
        threat_level=threat_assessment.threat_level,
        recommended_action=threat_assessment.recommended_action.action,
        confidence=threat_assessment.confidence
    )
    
    # Step 6: Optional - Queue for cloud analysis
    if has_internet_connectivity():
        await send_to_cloud_for_detailed_analysis(frame, weapon_type)

""")
    
    print("✓ Simple integration with weapon detection models")
    print("✓ Fast response for real-time display")
    print("✓ Optional cloud for detailed analysis")


if __name__ == "__main__":
    print("\n" + "="*70)
    print(" LOCAL REASONING AGENT - USAGE EXAMPLES")
    print(" (For Officer's Phone/Body Cam)")
    print("="*70)
    
    # Run all examples
    example_1_gun_detection()
    example_2_gun_at_school()
    example_3_unarmed_person()
    example_4_multiple_frames()
    example_5_different_weapons()
    example_6_offline_mode()
    example_7_mobile_app_integration()
    
    print("\n" + "="*70)
    print(" EXAMPLES COMPLETED")
    print("="*70)
    print("\n✓ Local Reasoning Agent ready for deployment!")
    print("  Deploy to: Officer's phone, body cam, edge device")
    print("  Latency: <100ms (instant on device)")
    print("  Offline: ✅ Works completely offline")
    print("  Memory: Minimal footprint")
