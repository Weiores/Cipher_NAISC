from app.schemas import (
    AudioDetection,
    EmotionDetection,
    FusedEvent,
    PerceptionRequest,
    UniformDetection,
    UnifiedDecision,
    UnifiedPerceptionOutput,
    UnifiedTraits,
    WeaponDetection,
)

UNIFORM_SUPPRESSION_MIN_CONFIDENCE = 0.8


def build_fused_event(
    request: PerceptionRequest,
    weapon: WeaponDetection,
    uniform: UniformDetection,
    emotion: EmotionDetection,
    audio: AudioDetection,
) -> FusedEvent:
    effective_weapon_label = _effective_weapon_label(weapon, uniform)
    suppression_reason = _build_suppression_reason(weapon, uniform)
    risk_hints: list[str] = []

    if effective_weapon_label in {"gun", "knife", "bat", "stick"}:
        risk_hints.append("visible_weapon")
    if suppression_reason is not None:
        risk_hints.append("weapon_suppressed_uniformed_personnel")
    if emotion.label in {"angry", "fearful", "distressed"}:
        risk_hints.append("emotional_escalation")
    if audio.tone in {"panic", "threat", "abnormal"}:
        risk_hints.append("audio_escalation")
    if audio.keyword_flags:
        risk_hints.append("speech_flags_present")

    rationale = (
        f"Weapon={effective_weapon_label} ({weapon.confidence:.2f}), "
        f"raw_weapon={weapon.label}, "
        f"uniform_present={uniform.uniform_present} ({uniform.confidence:.2f}), "
        f"emotion={emotion.label} ({emotion.confidence:.2f}), "
        f"tone={audio.tone} ({audio.confidence:.2f})."
    )

    return FusedEvent(
        source_id=request.source_id,
        timestamp=request.timestamp,
        weapon_detected=effective_weapon_label,
        emotion=emotion.label,
        tone=audio.tone,
        confidence_scores={
            "weapon": weapon.confidence,
            "uniform": uniform.confidence,
            "emotion": emotion.confidence,
            "tone": audio.confidence,
        },
        risk_hints=risk_hints,
        rationale_summary=rationale,
        suppression_reason=suppression_reason,
    )


def build_unified_output(
    request: PerceptionRequest,
    weapon: WeaponDetection,
    uniform: UniformDetection,
    emotion: EmotionDetection,
    audio: AudioDetection,
    fused: FusedEvent,
) -> UnifiedPerceptionOutput:
    effective_weapon_label = fused.weapon_detected
    threat_level = _compute_threat_level(effective_weapon_label, emotion, audio)
    anomaly_type = _build_anomaly_types(effective_weapon_label, emotion, audio, uniform)
    recommended_response = _build_recommended_response(threat_level, effective_weapon_label)
    confidence = max(weapon.confidence, uniform.confidence, emotion.confidence, audio.confidence)

    return UnifiedPerceptionOutput(
        source_id=request.source_id,
        timestamp=request.timestamp,
        model_backends={
            "weapon": weapon.backend,
            "uniform": uniform.backend,
            "emotion": "fer" if emotion.face_count > 0 and emotion.confidence > 0 else "placeholder",
            "audio": "waveform_heuristics_plus_whisper" if audio.transcript is not None else "waveform_heuristics",
        },
        confidence_scores=fused.confidence_scores,
        traits=UnifiedTraits(
            weapon_detected=effective_weapon_label,
            raw_weapon_detected=weapon.label,
            weapon_class_evidence=weapon.class_evidence,
            visual_secondary_evidence=_build_visual_secondary_evidence(weapon),
            uniform_present=uniform.uniform_present,
            uniform_confidence=uniform.confidence,
            uniform_evidence=uniform.class_evidence,
            weapon_suppressed_due_to_uniform=fused.suppression_reason is not None,
            suppression_reason=fused.suppression_reason,
            emotion=emotion.label,
            tone=audio.tone,
            speech_present=audio.speech_present,
            acoustic_events=audio.acoustic_events,
            transcript=audio.transcript,
            keyword_flags=audio.keyword_flags,
        ),
        risk_hints=fused.risk_hints,
        decision=UnifiedDecision(
            threat_level=threat_level,
            anomaly_type=anomaly_type,
            recommended_response=recommended_response,
            confidence=round(confidence, 4),
            rationale_summary=fused.rationale_summary,
        ),
    )


def _compute_threat_level(
    effective_weapon_label: str,
    emotion: EmotionDetection,
    audio: AudioDetection,
) -> str:
    score = 0
    if effective_weapon_label in {"gun", "knife", "bat", "stick"}:
        score += 3
    if emotion.label in {"angry", "fearful", "distressed"}:
        score += 1
    if audio.tone in {"panic", "threat", "abnormal"}:
        score += 1
    if "threat_language" in audio.keyword_flags:
        score += 1
    if "distress_language" in audio.keyword_flags:
        score += 1

    if effective_weapon_label == "gun" and audio.speech_present:
        score += 1

    if score >= 5:
        return "critical"
    if score >= 3:
        return "high"
    if score >= 2:
        return "medium"
    return "low"


def _build_anomaly_types(
    effective_weapon_label: str,
    emotion: EmotionDetection,
    audio: AudioDetection,
    uniform: UniformDetection,
) -> list[str]:
    anomaly_type: list[str] = []
    if effective_weapon_label in {"gun", "knife", "bat", "stick"}:
        anomaly_type.append("weapon_presence")
    if uniform.uniform_present:
        anomaly_type.append("uniformed_personnel_present")
    if emotion.label in {"angry", "fearful", "distressed"}:
        anomaly_type.append("emotional_escalation")
    if audio.tone in {"panic", "threat", "abnormal"}:
        anomaly_type.append("audio_escalation")
    if audio.keyword_flags:
        anomaly_type.append("speech_risk_signal")
    return anomaly_type


def _build_recommended_response(
    threat_level: str,
    effective_weapon_label: str,
) -> list[str]:
    if threat_level == "critical":
        return ["dispatch_alert", "escalate", "lockdown", "preserve_evidence"]
    if threat_level == "high":
        actions = ["dispatch_alert", "escalate", "monitor_live"]
        if effective_weapon_label in {"gun", "knife"}:
            actions.append("isolate_area")
        return actions
    if threat_level == "medium":
        return ["monitor_live", "operator_review"]
    return ["monitor"]


def _build_visual_secondary_evidence(weapon: WeaponDetection) -> list[dict[str, float | int | str]]:
    secondary: list[dict[str, float | int | str]] = []
    for item in weapon.class_evidence:
        label = str(item.get("label", ""))
        if label in {weapon.label, "unarmed", "unknown_object"}:
            continue
        secondary.append(item)
    return secondary


def _effective_weapon_label(weapon: WeaponDetection, uniform: UniformDetection) -> str:
    if _uniform_suppression_applies(weapon, uniform):
        return "unarmed"
    return weapon.label


def _build_suppression_reason(weapon: WeaponDetection, uniform: UniformDetection) -> str | None:
    if _uniform_suppression_applies(weapon, uniform):
        return (
            "Weapon alert suppressed because strong uniformed-personnel evidence "
            f"was detected (confidence >= {UNIFORM_SUPPRESSION_MIN_CONFIDENCE:.2f})."
        )
    return None


def _uniform_suppression_applies(weapon: WeaponDetection, uniform: UniformDetection) -> bool:
    return (
        uniform.uniform_present
        and uniform.confidence >= UNIFORM_SUPPRESSION_MIN_CONFIDENCE
        and weapon.label in {"gun", "knife", "bat", "stick"}
    )
