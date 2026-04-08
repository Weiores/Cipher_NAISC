"""
Cloud Reasoning Agent - Enhanced decision-making with SOP integration and learning feedback.

DEPLOYMENT: This agent is designed to run on CLOUD SERVERS.

This agent receives fused traits from the perception layer and performs weighted reasoning
to generate recommendations. It can integrate with:
- SOP/Rule Base for structured decision logic
- Learning Agent outputs (top 3 scenario predictions)
- Historical incident data
- Operator feedback (for future training)

Features:
- Full multi-factor threat analysis (weapon 50%, emotion 25%, audio 20%, behavioral 5%)
- SOP-aware decision making with context multipliers
- Learning Agent integration for scenario predictions
- Comprehensive explanation generation
- High accuracy, higher latency acceptable (~500ms)

Use Case:
- Deploy on cloud server
- Called from officer's device (perception results sent to cloud)
- Returns threat decision + recommended action
- Suitable for offline replay and detailed analysis
"""

from datetime import datetime, timezone
from typing import Literal
from dataclasses import dataclass
from enum import Enum
import json
import logging

from schemas import (
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


@dataclass
class ScenarioPrediction:
    """Top 3 likely scenario progressions from Learning Agent"""
    rank: int  # 1, 2, 3
    scenario_description: str
    probability: float  # 0.0-1.0
    estimated_escalation_time_seconds: int | None
    recommended_preemptive_action: str | None


@dataclass
class SOPContext:
    """Standard Operating Procedure context"""
    location_type: str  # e.g., "airport", "bank", "school", "street"
    time_of_day: str  # e.g., "peak_hours", "night", "weekend"
    security_level: str  # e.g., "low", "medium", "high", "critical"
    active_alerts: list[str]  # e.g., ["active_shooter", "bomb_threat"]
    personnel_count: int | None
    restricted_weapons: list[str]  # e.g., ["gun", "explosive"]


class CloudReasoningAgent:
    """
    Main cloud reasoning agent. This is the brain of the decision-making system.
    
    CLOUD DEPLOYMENT: Runs on centralized server with full computing resources.
    
    Features:
    - Receives fused threat traits from perception layer
    - Integrates with Learning Agent for scenario prediction
    - Uses SOP/rule base for structured decision-making
    - Generates confidence-scored recommendations
    - Supports weighted multi-factor analysis
    - Returns detailed explanations with evidence
    
    Latency: ~100-500ms per decision (acceptable for cloud)
    Accuracy: Highest (uses all modalities)
    Resources: Requires cloud server
    """
    
    def __init__(self, sop_base_path: str = "./sop_rules"):
        """
        Initialize the cloud reasoning agent.
        
        Args:
            sop_base_path: Path to SOP/rule base files
        """
        self.sop_base_path = sop_base_path
        self.sop_rules = self._load_sop_rules()
        self.threat_thresholds = self._init_threat_thresholds()
        logger.info("[CLOUD_AGENT] Cloud Reasoning Agent initialized")
    
    def reason(
        self,
        source_id: str,
        weapon_detected: str,
        emotion: str,
        tone: str,
        confidence_scores: dict[str, float],
        risk_hints: list[str],
        scenario_predictions: list[ScenarioPrediction] | None = None,
        sop_context: SOPContext | None = None,
    ) -> ReasoningOutput:
        """
        Main reasoning method. Performs weighted analysis and generates recommendation.
        
        Args:
            source_id: Camera/sensor identifier
            weapon_detected: "gun", "knife", "unarmed", etc.
            emotion: "neutral", "angry", "fearful", "distressed", etc.
            tone: "calm", "panic", "threat", "abnormal", etc.
            confidence_scores: dict with keys: weapon, emotion, tone, uniform
            risk_hints: Extracted risk indicators from perception layer
            scenario_predictions: Top 3 likely scenarios from Learning Agent
            sop_context: Current operating context and constraints
        
        Returns:
            ReasoningOutput with threat level, recommendation, and explanation
        """
        logger.info(f"[CLOUD_AGENT] Reasoning for source={source_id}, weapon={weapon_detected}, emotion={emotion}, tone={tone}")
        
        # Step 1: Compute individual threat scores
        threat_metrics = self._compute_threat_metrics(
            weapon_detected, 
            emotion, 
            tone, 
            confidence_scores,
            risk_hints
        )
        
        # Step 2: Apply SOP/context weighting
        if sop_context:
            threat_metrics = self._apply_sop_context(threat_metrics, sop_context)
            logger.info(f"[CLOUD_AGENT] Applied SOP context adjustment: {sop_context.location_type}")
        
        # Step 3: Integrate scenario predictions from Learning Agent
        if scenario_predictions:
            threat_metrics = self._integrate_scenario_predictions(
                threat_metrics, 
                scenario_predictions
            )
            logger.info(f"[CLOUD_AGENT] Integrated {len(scenario_predictions)} scenario predictions")
        
        # Step 4: Generate threat level and recommendation
        threat_level = self._classify_threat_level(threat_metrics.combined_threat_score)
        
        # FIREARM OVERRIDE: Any firearm detection = at least HIGH threat (CRITICAL if threatening emotion/tone)
        # This is a hard security rule - guns are inherently high-threat weapons
        if weapon_detected and weapon_detected.lower() in {"gun", "rifle", "shotgun"}:
            original_threat = threat_level
            
            # Guns are minimum HIGH threat
            if threat_level in {RiskLevel.LOW, RiskLevel.MEDIUM}:
                threat_level = RiskLevel.HIGH
                logger.info(f"[CLOUD_AGENT] ⚠️ FIREARM DETECTED: Forcing threat_level to HIGH (was {original_threat})")
            
            # Escalate to CRITICAL if there's also threatening emotion or tone
            if threat_level == RiskLevel.HIGH:
                if emotion and emotion.lower() in {"angry", "fearful", "distressed", "panicked"}:
                    threat_level = RiskLevel.CRITICAL
                    logger.info(f"[CLOUD_AGENT] 🚨 FIREARM + THREATENING EMOTION: Forcing threat_level to CRITICAL (emotion={emotion})")
                elif tone and tone.lower() in {"threat", "panic", "abnormal", "distressed"}:
                    threat_level = RiskLevel.CRITICAL
                    logger.info(f"[CLOUD_AGENT] 🚨 FIREARM + THREATENING TONE: Forcing threat_level to CRITICAL (tone={tone})")
        
        recommended_action = self._generate_recommendation(
            threat_level,
            weapon_detected,
            emotion,
            threat_metrics,
            scenario_predictions,
            sop_context
        )
        
        # Step 5: Build explanation
        explanation = self._build_explanation(
            weapon_detected,
            emotion,
            tone,
            threat_metrics,
            risk_hints,
            scenario_predictions,
            sop_context
        )
        
        # Compute final confidence
        final_confidence = self._compute_confidence_score(
            threat_metrics,
            confidence_scores,
            recommended_action
        )
        
        # Build and return output
        output = ReasoningOutput(
            source_id=source_id,
            timestamp=datetime.now(timezone.utc),
            threat_level=threat_level,
            confidence=final_confidence,
            recommended_action=recommended_action,
            explanation=explanation,
            anomaly_types=self._detect_anomalies(weapon_detected, emotion, tone, risk_hints),
            metrics=threat_metrics,
            reasoning_version="cloud_v1.0"
        )
        
        logger.info(f"[CLOUD_AGENT] Decision: threat_level={threat_level}, action={recommended_action.action}, confidence={final_confidence:.2f}")
        return output
    
    def _compute_threat_metrics(
        self,
        weapon_detected: str,
        emotion: str,
        tone: str,
        confidence_scores: dict[str, float],
        risk_hints: list[str],
    ) -> ThreatMetrics:
        """Compute individual threat scores from each modality"""
        
        # Weapon threat score
        weapon_threat = self._score_weapon_threat(weapon_detected, confidence_scores.get("weapon", 0.0))
        
        # Emotion threat score
        emotion_threat = self._score_emotion_threat(emotion, confidence_scores.get("emotion", 0.0))
        
        # Audio threat score
        audio_threat = self._score_audio_threat(tone, confidence_scores.get("tone", 0.0))
        
        # Behavioral anomaly (risk hints)
        behavioral_anomaly = self._score_behavioral_anomaly(risk_hints)
        
        # Combined threat (weighted average)
        # Weights: weapon 50%, emotion 25%, audio 20%, behavioral 5%
        combined = (
            weapon_threat * 0.50 +
            emotion_threat * 0.25 +
            audio_threat * 0.20 +
            behavioral_anomaly * 0.05
        )
        
        # Determine trend (would normally come from history, defaulting to stable)
        trend = "stable"  # This would be computed from frame history in production
        
        metrics = ThreatMetrics(
            weapon_threat_score=weapon_threat,
            emotion_threat_score=emotion_threat,
            audio_threat_score=audio_threat,
            behavioral_anomaly_score=behavioral_anomaly,
            combined_threat_score=combined,
            trend=trend,
            frames_in_history=1,  # Would be 10+ in production
            context_anomaly_flag=False
        )
        
        logger.debug(f"[CLOUD_AGENT] Threat metrics: weapon={weapon_threat:.2f}, emotion={emotion_threat:.2f}, audio={audio_threat:.2f}, combined={combined:.2f}")
        return metrics
    
    def _score_weapon_threat(self, weapon: str, confidence: float) -> float:
        """Score weapon threat on 0-1 scale"""
        weapon_scores = {
            "gun": 0.95,
            "rifle": 0.95,
            "shotgun": 0.95,
            "knife": 0.85,
            "blade": 0.85,
            "bat": 0.60,
            "stick": 0.40,
            "others": 0.50,
            "unarmed": 0.0,
        }
        
        base_score = weapon_scores.get(weapon.lower(), 0.30)
        
        # CRITICAL WEAPONS: Gun, Rifle, Shotgun - ALWAYS MAXIMUM THREAT
        # Even if detected in few frames, guns are inherently critical
        if weapon.lower() in {"gun", "rifle", "shotgun"}:
            # Gun detected = IMMEDIATE CRITICAL THREAT (minimum 0.95)
            # Confidence only boosts above baseline
            return 0.95 + (0.05 * confidence)  # Range: 0.95-1.0
        
        # HIGH-THREAT WEAPONS (knife, blade) - HIGH baseline threat
        elif base_score >= 0.85:
            # Knife/blade: minimum 0.85, boosted by confidence
            return min(1.0, 0.85 + (0.15 * confidence))
        
        # MEDIUM-THREAT WEAPONS - Confidence weighted
        else:
            return base_score * (0.5 + 0.5 * confidence)
    
    def _score_emotion_threat(self, emotion: str, confidence: float) -> float:
        """Score emotional threat on 0-1 scale"""
        emotion_scores = {
            "distressed": 0.80,
            "fearful": 0.60,
            "angry": 0.75,
            "panicked": 0.80,
            "neutral": 0.0,
            "calm": 0.0,
            "happy": 0.0,
        }
        
        base_score = emotion_scores.get(emotion.lower(), 0.0)
        return base_score * confidence
    
    def _score_audio_threat(self, tone: str, confidence: float) -> float:
        """Score audio/tone threat on 0-1 scale"""
        tone_scores = {
            "panic": 0.85,
            "threat": 0.80,
            "abnormal": 0.60,
            "distressed": 0.70,
            "calm": 0.0,
            "neutral": 0.0,
        }
        
        base_score = tone_scores.get(tone.lower(), 0.0)
        return base_score * confidence
    
    def _score_behavioral_anomaly(self, risk_hints: list[str]) -> float:
        """Score behavioral anomalies"""
        anomaly_weights = {
            "visible_weapon": 0.40,
            "emotional_escalation": 0.30,
            "audio_escalation": 0.30,
            "speech_flags_present": 0.20,
            "weapon_suppressed_uniformed_personnel": -0.20,  # Negative = reduces threat
        }
        
        score = 0.0
        for hint in risk_hints:
            score += anomaly_weights.get(hint, 0.0)
        
        # Clamp to 0-1
        return max(0.0, min(1.0, score))
    
    def _apply_sop_context(self, metrics: ThreatMetrics, context: SOPContext) -> ThreatMetrics:
        """Apply SOP-based context adjustments to threat metrics"""
        
        # Location-based adjustments
        location_multipliers = {
            "airport": 1.3,  # Higher sensitivity at airports
            "bank": 1.2,
            "school": 1.4,  # Highest sensitivity at schools
            "street": 1.0,
            "office": 0.9,
        }
        
        # Security level multipliers
        security_multipliers = {
            "critical": 1.5,
            "high": 1.3,
            "medium": 1.0,
            "low": 0.8,
        }
        
        location_mult = location_multipliers.get(context.location_type, 1.0)
        security_mult = security_multipliers.get(context.security_level, 1.0)
        
        # Apply multipliers
        combined = metrics.combined_threat_score * location_mult * security_mult
        combined = min(1.0, combined)  # Cap at 1.0
        
        # Check for active alerts matching detections
        anomaly_flag = False
        for alert in context.active_alerts:
            if alert.lower() in ["active_shooter", "armed_individual"]:
                anomaly_flag = True
        
        logger.debug(f"[CLOUD_AGENT] SOP adjustment: location_mult={location_mult}, security_mult={security_mult}, combined_adjusted={combined:.2f}")
        
        return ThreatMetrics(
            weapon_threat_score=metrics.weapon_threat_score,
            emotion_threat_score=metrics.emotion_threat_score,
            audio_threat_score=metrics.audio_threat_score,
            behavioral_anomaly_score=metrics.behavioral_anomaly_score,
            combined_threat_score=combined,
            trend=metrics.trend,
            frames_in_history=metrics.frames_in_history,
            context_anomaly_flag=anomaly_flag,
        )
    
    def _integrate_scenario_predictions(
        self,
        metrics: ThreatMetrics,
        predictions: list[ScenarioPrediction]
    ) -> ThreatMetrics:
        """Integrate Learning Agent's scenario predictions into threat assessment"""
        
        if not predictions:
            return metrics
        
        # Top scenario's probability increases our confidence
        top_scenario = predictions[0]
        scenario_weight = top_scenario.probability * 0.1  # Max 10% impact
        
        # If top scenario is escalating and likely soon, increase threat
        escalation_boost = 0.0
        if "escalat" in top_scenario.scenario_description.lower():
            if top_scenario.estimated_escalation_time_seconds and top_scenario.estimated_escalation_time_seconds < 60:
                escalation_boost = top_scenario.probability * 0.15  # Up to 15% boost
        
        adjusted_combined = min(1.0, metrics.combined_threat_score + scenario_weight + escalation_boost)
        
        logger.debug(f"[CLOUD_AGENT] Scenario integration: top_prob={top_scenario.probability:.2f}, boost={escalation_boost:.2f}, adjusted_combined={adjusted_combined:.2f}")
        
        return ThreatMetrics(
            weapon_threat_score=metrics.weapon_threat_score,
            emotion_threat_score=metrics.emotion_threat_score,
            audio_threat_score=metrics.audio_threat_score,
            behavioral_anomaly_score=metrics.behavioral_anomaly_score,
            combined_threat_score=adjusted_combined,
            trend="escalating" if escalation_boost > 0 else metrics.trend,
            frames_in_history=metrics.frames_in_history,
            context_anomaly_flag=metrics.context_anomaly_flag,
        )
    
    def _classify_threat_level(self, score: float) -> RiskLevel:
        """Map threat score to threat level"""
        if score >= 0.75:
            return RiskLevel.CRITICAL
        elif score >= 0.50:
            return RiskLevel.HIGH
        elif score >= 0.25:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def _generate_recommendation(
        self,
        threat_level: RiskLevel,
        weapon: str,
        emotion: str,
        metrics: ThreatMetrics,
        scenarios: list[ScenarioPrediction] | None,
        sop_context: SOPContext | None,
    ) -> RecommendedAction:
        """Generate recommended action based on all factors"""
        
        action_map = {
            RiskLevel.CRITICAL: {
                "action": "immediate_alert",
                "priority": "critical",
                "reason": "Critical threat detected - immediate action required",
            },
            RiskLevel.HIGH: {
                "action": "escalate",
                "priority": "high",
                "reason": "High threat level - escalate to security personnel",
            },
            RiskLevel.MEDIUM: {
                "action": "elevated_monitoring",
                "priority": "medium",
                "reason": "Medium threat - escalate monitoring and alert",
            },
            RiskLevel.LOW: {
                "action": "monitor",
                "priority": "low",
                "reason": "Low threat - continue routine monitoring",
            },
        }
        
        base_recommendation = action_map[threat_level]
        
        # Refine based on scenarios
        if scenarios and len(scenarios) > 0:
            top_scenario = scenarios[0]
            if top_scenario.recommended_preemptive_action:
                base_recommendation["reason"] += f" (Scenario: {top_scenario.scenario_description})"
        
        # Confidence score
        confidence = metrics.combined_threat_score if threat_level == RiskLevel.CRITICAL else 0.9
        
        return RecommendedAction(
            action=base_recommendation["action"],
            priority=base_recommendation["priority"],
            reason=base_recommendation["reason"],
            confidence=confidence,
        )
    
    def _build_explanation(
        self,
        weapon: str,
        emotion: str,
        tone: str,
        metrics: ThreatMetrics,
        risk_hints: list[str],
        scenarios: list[ScenarioPrediction] | None,
        sop_context: SOPContext | None,
    ) -> ReasoningExplanation:
        """Build human-readable explanation of reasoning"""
        
        # Summary
        summary = f"Threat assessment based on weapon detection ({weapon}), "
        summary += f"emotion analysis ({emotion}), and audio analysis ({tone}). "
        
        if metrics.context_anomaly_flag:
            summary += "Context anomalies detected. "
        
        if scenarios:
            summary += f"Learning agent predicts {len(scenarios)} potential scenarios. "
        
        # Key factors
        key_factors = []
        if metrics.weapon_threat_score > 0.5:
            key_factors.append(f"Potential weapon detected: {weapon}")
        if metrics.emotion_threat_score > 0.4:
            key_factors.append(f"Elevated emotional state: {emotion}")
        if metrics.audio_threat_score > 0.4:
            key_factors.append(f"Threatening audio tone: {tone}")
        if metrics.behavioral_anomaly_score > 0.3:
            key_factors.append("Behavioral anomalies present")
        
        # Evidence
        evidence = {
            "weapon_evidence": [f"Label: {weapon}"],
            "emotion_evidence": [f"Emotion: {emotion}"],
            "audio_evidence": [f"Tone: {tone}"],
            "risk_indicators": risk_hints,
        }
        
        # Anomalies detected
        anomalies = []
        for hint in risk_hints:
            if "escalation" in hint.lower():
                anomalies.append("Escalation detected")
            if "weapon" in hint.lower() and hint != "weapon_suppressed_uniformed_personnel":
                anomalies.append("Weapon presence")
        
        # Temporal analysis
        temporal = _build_temporal_analysis(metrics)
        
        # Confidence reasoning
        confidence_reason = f"Confidence based on combined threat score ({metrics.combined_threat_score:.2f}) "
        confidence_reason += f"from weapon ({metrics.weapon_threat_score:.2f}), "
        confidence_reason += f"emotion ({metrics.emotion_threat_score:.2f}), "
        confidence_reason += f"and audio ({metrics.audio_threat_score:.2f}) modalities."
        
        return ReasoningExplanation(
            summary=summary,
            key_factors=key_factors,
            evidence=evidence,
            anomalies_detected=anomalies,
            temporal_analysis=temporal,
            confidence_reasoning=confidence_reason,
        )
    
    def _compute_confidence_score(
        self,
        metrics: ThreatMetrics,
        confidence_scores: dict[str, float],
        recommendation: RecommendedAction,
    ) -> float:
        """Compute overall confidence in the recommendation"""
        # Average of individual model confidences
        model_confidence = sum(confidence_scores.values()) / len(confidence_scores) if confidence_scores else 0.5
        
        # Threat metric agreement
        metric_agreement = 1.0 - abs(
            metrics.weapon_threat_score - metrics.combined_threat_score
        ) / 2  # Simplified
        
        # Recommendation confidence is already set
        recommendation_confidence = recommendation.confidence
        
        # Weighted average
        final = (
            model_confidence * 0.4 +
            metric_agreement * 0.3 +
            recommendation_confidence * 0.3
        )
        
        return min(1.0, max(0.0, final))
    
    def _detect_anomalies(
        self,
        weapon: str,
        emotion: str,
        tone: str,
        risk_hints: list[str]
    ) -> list[str]:
        """Detect anomalies based on inputs"""
        anomalies = []
        
        if weapon != "unarmed":
            anomalies.append("weapon_present")
        
        if emotion in ["angry", "fearful", "distressed", "panicked"]:
            anomalies.append("emotional_escalation")
        
        if tone in ["panic", "threat", "abnormal"]:
            anomalies.append("audio_escalation")
        
        for hint in risk_hints:
            if hint not in anomalies and hint != "weapon_suppressed_uniformed_personnel":
                anomalies.append(hint)
        
        return anomalies
    
    def _load_sop_rules(self) -> dict:
        """Load SOP/rule base (stub for now)"""
        # In production, this would load from JSON/YAML files
        return {
            "escalation_rules": [],
            "de_escalation_rules": [],
            "restricted_locations": [],
            "restricted_weapons": [],
        }
    
    def _init_threat_thresholds(self) -> dict:
        """Initialize threat level thresholds"""
        return {
            "critical": 0.75,
            "high": 0.50,
            "medium": 0.25,
            "low": 0.0,
        }


def _build_temporal_analysis(metrics: ThreatMetrics) -> str:
    """Build temporal analysis string"""
    if metrics.trend == "escalating":
        return "Threat is escalating over time - immediate action recommended"
    elif metrics.trend == "de_escalating":
        return "Threat is de-escalating - monitor situation"
    else:
        return "Threat level is stable - maintain current monitoring"
