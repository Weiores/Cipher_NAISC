"""
Local Reasoning Agent - Fast, lightweight reasoning for edge devices (officer's phone/bodycam).

DEPLOYMENT: This agent is designed to run on EDGE DEVICES (officer's phone, body cam, etc).

This is a simplified, fast version of the reasoning agent that:
- Does NOT require scenario predictions
- Uses only SOP context for decision making
- Minimal computation (device-friendly)
- Fast latency (<100ms even on mobile devices)
- Works offline without cloud connectivity

Features:
- Simple threat scoring (weapon primary indicator)
- SOP-based context awareness
- Location and security level multipliers
- Basic explanation generation
- Offline-capable
- Low memory footprint

Use Case:
- Deploy on officer's phone/body cam
- Gets perception results from local models
- Makes fast threat assessment
- Shows recommendation on device screen
- Falls back from cloud agent if connectivity lost
"""

from datetime import datetime, timezone
from enum import Enum
import logging

from ..schemas import (
    ReasoningOutput,
    RecommendedAction,
    ThreatMetrics,
    ReasoningExplanation,
)


logger = logging.getLogger(__name__)


class RiskLevel(str, Enum):
    """Threat levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class LocalReasoningAgent:
    """
    Lightweight local reasoning agent for edge devices.
    
    EDGE DEPLOYMENT: Runs on officer's phone/body cam with minimal resources.
    
    Features:
    - Fast threat assessment (~50ms)
    - Offline operation (no cloud dependency)
    - Minimal memory footprint
    - SOP-aware decision making
    - Simple but effective scoring
    
    Latency: <100ms even on mobile devices
    Accuracy: Good (simplified but still effective)
    Resources: Minimal (runs on any device)
    
    DIFFERENCE from CloudReasoningAgent:
    - Does NOT use emotion threat (emotion detection may fail on device)
    - Does NOT use audio threat (requires processing pipeline)
    - Does NOT integrate scenario predictions (requires ML model)
    - Does NOT use behavioral anomalies (complex heuristics)
    - Uses ONLY weapon threat + SOP context
    
    This ensures:
    ✅ Works completely offline
    ✅ Fast even on slow phones
    ✅ Reliable baseline assessment
    ✅ Graceful fallback from cloud
    """
    
    def __init__(self):
        """Initialize the local reasoning agent"""
        logger.info("[LOCAL_AGENT] Local Reasoning Agent initialized (edge device)")
    
    def reason(
        self,
        source_id: str,
        weapon_detected: str,
        emotion: str = None,
        tone: str = None,
        confidence_scores: dict[str, float] = None,
        risk_hints: list[str] = None,
        sop_context = None,
    ) -> ReasoningOutput:
        """
        Fast reasoning method for edge devices.
        
        SIMPLIFIED from cloud version:
        - Only uses weapon detection (most reliable)
        - Ignores emotion/audio (may not be available on device)
        - Ignores scenario predictions (not on device)
        - Uses SOP context for multipliers
        
        Args:
            source_id: Camera/sensor identifier
            weapon_detected: "gun", "knife", "unarmed", etc.
            emotion: OPTIONAL (may not be available on device)
            tone: OPTIONAL (may not be available on device)
            confidence_scores: OPTIONAL weapon confidence minimum required
            risk_hints: OPTIONAL
            sop_context: OPTIONAL location and security level
        
        Returns:
            ReasoningOutput with threat level and fast recommendation
        """
        logger.info(f"[LOCAL_AGENT] Fast reasoning for source={source_id}, weapon={weapon_detected}")
        
        # Step 1: Simple weapon-based threat score
        weapon_threat = self._score_weapon_threat(
            weapon_detected, 
            confidence_scores.get("weapon", 0.0) if confidence_scores else 0.0
        )
        
        # Step 2: Apply SOP context multipliers
        threat_score = weapon_threat
        if sop_context:
            threat_score = self._apply_sop_context(threat_score, sop_context)
            logger.info(f"[LOCAL_AGENT] SOP adjusted: {sop_context.location_type}")
        
        # Step 3: Classify threat level
        threat_level = self._classify_threat_level(threat_score)
        
        # Step 4: Generate recommendation
        recommended_action = self._generate_recommendation(threat_level, weapon_detected)
        
        # Step 5: Build simple explanation
        explanation = self._build_explanation(weapon_detected, threat_score, sop_context)
        
        # Build metrics object
        metrics = ThreatMetrics(
            weapon_threat_score=weapon_threat,
            emotion_threat_score=0.0,  # Not computed on device
            audio_threat_score=0.0,     # Not computed on device
            behavioral_anomaly_score=0.0,
            combined_threat_score=threat_score,
            trend="stable",
            frames_in_history=1,
            context_anomaly_flag=False
        )
        
        # Build output
        output = ReasoningOutput(
            source_id=source_id,
            timestamp=datetime.now(timezone.utc),
            threat_level=threat_level,
            confidence=threat_score,
            recommended_action=recommended_action,
            explanation=explanation,
            anomaly_types=self._detect_anomalies(weapon_detected),
            metrics=metrics,
            reasoning_version="local_v1.0"  # Different version to distinguish from cloud
        )
        
        logger.info(f"[LOCAL_AGENT] Decision: threat={threat_level}, action={recommended_action.action}, score={threat_score:.2f}")
        return output
    
    def _score_weapon_threat(self, weapon: str, confidence: float) -> float:
        """
        Score weapon threat - SIMPLE VERSION for edge devices.
        
        Only weapon detection counts, confidence-weighted.
        """
        weapon_scores = {
            "gun": 1.0,        # Highest/immediate threat
            "rifle": 1.0,
            "shotgun": 1.0,
            "knife": 0.8,      # High threat
            "blade": 0.8,
            "bat": 0.5,        # Medium threat
            "stick": 0.3,      # Lower threat
            "unarmed": 0.0,    # No threat
        }
        
        base_score = weapon_scores.get(weapon.lower(), 0.2)
        # Confidence-weighted
        return base_score * min(confidence, 1.0)
    
    def _apply_sop_context(self, threat_score: float, sop_context) -> float:
        """
        Apply SOP multipliers - SIMPLE VERSION.
        
        Adjusts threat based on location and security level.
        """
        # Location-based multipliers
        location_mult = {
            "airport": 1.5,
            "bank": 1.3,
            "school": 2.0,     # Schools = highest sensitivity
            "street": 1.0,
            "office": 1.0,
        }.get(sop_context.location_type, 1.0)
        
        # Security level multipliers
        security_mult = {
            "critical": 1.5,
            "high": 1.2,
            "medium": 1.0,
            "low": 0.8,
        }.get(sop_context.security_level, 1.0)
        
        # Apply and cap at 1.0
        adjusted = min(1.0, threat_score * location_mult * security_mult)
        logger.debug(f"[LOCAL_AGENT] SOP: location_mult={location_mult}, security_mult={security_mult}, adjusted={adjusted:.2f}")
        return adjusted
    
    def _classify_threat_level(self, score: float) -> RiskLevel:
        """Map threat score to threat level - SIMPLE THRESHOLD"""
        if score >= 0.75:
            return RiskLevel.CRITICAL
        elif score >= 0.50:
            return RiskLevel.HIGH
        elif score >= 0.25:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def _generate_recommendation(self, threat_level: RiskLevel, weapon: str) -> RecommendedAction:
        """Generate simple, fast recommendation"""
        
        actions = {
            RiskLevel.CRITICAL: RecommendedAction(
                action="immediate_alert",
                priority="critical",
                reason=f"⚠️ CRITICAL: {weapon} detected - immediate action required",
                confidence=1.0
            ),
            RiskLevel.HIGH: RecommendedAction(
                action="escalate",
                priority="high",
                reason=f"🔴 HIGH: {weapon} detected - escalate to authorities",
                confidence=0.95
            ),
            RiskLevel.MEDIUM: RecommendedAction(
                action="elevated_monitoring",
                priority="medium",
                reason=f"🟡 MEDIUM: {weapon} detected - increase monitoring",
                confidence=0.85
            ),
            RiskLevel.LOW: RecommendedAction(
                action="monitor",
                priority="low",
                reason="🟢 LOW: Continue routine monitoring",
                confidence=0.7
            ),
        }
        
        return actions[threat_level]
    
    def _build_explanation(self, weapon: str, threat_score: float, sop_context = None) -> ReasoningExplanation:
        """Build simple explanation - for edge device display"""
        
        summary = f"Weapon: {weapon}"
        if sop_context:
            summary += f" | Location: {sop_context.location_type}"
        summary += f" | Threat Score: {threat_score:.1%}"
        
        key_factors = []
        if weapon != "unarmed":
            key_factors.append(f"Weapon detected: {weapon}")
        if sop_context and sop_context.location_type == "school":
            key_factors.append("School zone - heightened sensitivity")
        if sop_context and sop_context.security_level == "critical":
            key_factors.append("Critical security level")
        
        evidence = {
            "weapon_detection": weapon,
            "threat_score": threat_score,
        }
        
        temporal = "Immediate assessment from edge device"
        
        return ReasoningExplanation(
            summary=summary,
            key_factors=key_factors,
            evidence=evidence,
            anomalies_detected=[],
            temporal_analysis=temporal,
            confidence_reasoning=f"Based on weapon detection confidence: {threat_score:.0%}",
        )
    
    def _detect_anomalies(self, weapon: str) -> list[str]:
        """Detect anomalies - SIMPLE VERSION"""
        anomalies = []
        if weapon != "unarmed":
            anomalies.append("weapon_present")
        return anomalies
