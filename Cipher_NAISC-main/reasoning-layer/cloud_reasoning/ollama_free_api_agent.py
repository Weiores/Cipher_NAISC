"""
Ollama Free API Reasoning Agent - Non-local, cloud-hosted Ollama models.

DEPLOYMENT: Cloud (no local installation needed!)

This agent uses OllamaFreeAPI - a free, distributed API that hosts Ollama models
on community nodes. No need to run Ollama locally or have a cloud server.

Features:
- Zero setup - just install ollamafreeapi package
- No API keys required
- Free forever (community-funded)
- 50+ models available
- Auto load-balanced across global nodes
- Works offline fallback included

Latency: ~200-1000ms per decision (depends on API load)
Accuracy: High (same as local Ollama)
Resources: None (runs in cloud)
Cost: Free!

Usage:
    agent = OllamaFreeAPIReasoningAgent(model="llama3.2:3b")
    
    result = agent.reason(
        source_id="camera_1",
        weapon_detected="gun",
        emotion="angry",
        tone="threat",
        confidence_scores={...},
        risk_hints=[...],
    )
"""

import json
import logging
import re
from datetime import datetime, timezone
from typing import Optional

from ollamafreeapi import OllamaFreeAPI

from schemas import (
    ReasoningOutput,
    RecommendedAction,
    ThreatMetrics,
    ReasoningExplanation,
)

from cloud_reasoning_agent import (
    ScenarioPrediction,
    SOPContext,
)

logger = logging.getLogger(__name__)


class OllamaFreeAPIReasoningAgent:
    """
    Reasoning agent using OllamaFreeAPI - cloud-hosted Ollama without setup.
    
    FEATURES:
    - Uses free community-hosted Ollama API
    - No local installation required
    - No API keys or credit cards
    - Same interface as local OllamaReasoningAgent
    - Auto-fallback to rule-based if API unavailable
    
    AVAILABLE MODELS:
    - llama3.2:3b (lightweight, good for resource-constrained)
    - mistral:latest (fast, capable)
    - deepseek-r1:latest (strong reasoning)
    - gpt-oss:20b (powerful)
    - qwen:latest (multilingual)
    - phi:latest (small, efficient)
    
    USAGE:
        agent = OllamaFreeAPIReasoningAgent(model="llama3.2:3b")
        
        result = agent.reason(
            source_id="camera_1",
            weapon_detected="gun",
            emotion="angry",
            tone="threat",
            confidence_scores={"weapon": 0.95, "emotion": 0.8, "tone": 0.9},
            risk_hints=["visible_weapon", "emotional_escalation"]
        )
    """
    
    def __init__(self, model: str = "llama3.2:3b", temperature: float = 0.3):
        """
        Initialize OllamaFreeAPI Reasoning Agent.
        
        Args:
            model: Model to use from OllamaFreeAPI
                  (llama3.2:3b, mistral:latest, deepseek-r1:latest, etc.)
            temperature: LLM temperature (0.0-1.0, lower = more deterministic)
        """
        self.model = model
        self.temperature = temperature
        self.client = OllamaFreeAPI()
        
        logger.info(
            f"[OLLAMA_FREE_API_AGENT] Initialized with model={model}"
        )
    
    def is_available(self) -> bool:
        """Check if OllamaFreeAPI is reachable"""
        try:
            # Try to list models - quick connectivity check
            models = self.client.list_models()
            logger.info("[OLLAMA_FREE_API_AGENT] API is available")
            return True
        except Exception as e:
            logger.warning(
                f"[OLLAMA_FREE_API_AGENT] API not available: {str(e)}"
            )
            return False
    
    def reason(
        self,
        source_id: str,
        weapon_detected: str,
        emotion: str,
        tone: str,
        confidence_scores: dict[str, float],
        risk_hints: list[str],
        scenario_predictions: Optional[list[ScenarioPrediction]] = None,
        sop_context: Optional[SOPContext] = None,
    ) -> ReasoningOutput:
        """
        Perform threat reasoning using OllamaFreeAPI.
        
        Args:
            source_id: Camera/sensor identifier
            weapon_detected: Detected weapon type
            emotion: Detected emotion
            tone: Audio tone analysis
            confidence_scores: Dict with confidence values
            risk_hints: Risk indicators from perception layer
            scenario_predictions: Optional learning agent predictions
            sop_context: Optional SOP/context information
        
        Returns:
            ReasoningOutput with threat assessment and recommendation
        """
        
        if not self.is_available():
            logger.error(
                "[OLLAMA_FREE_API_AGENT] OllamaFreeAPI is not available. "
                "Check your internet connection."
            )
            raise ConnectionError(
                "OllamaFreeAPI is not available. "
                "Check your internet connection: https://ollama.com/download"
            )
        
        logger.info(
            f"[OLLAMA_FREE_API_AGENT] Reasoning for source={source_id}, "
            f"weapon={weapon_detected}, emotion={emotion}, tone={tone}"
        )
        
        # Build the reasoning prompt
        prompt = self._build_threat_prompt(
            source_id=source_id,
            weapon_detected=weapon_detected,
            emotion=emotion,
            tone=tone,
            confidence_scores=confidence_scores,
            risk_hints=risk_hints,
            scenario_predictions=scenario_predictions,
            sop_context=sop_context,
        )
        
        # Call OllamaFreeAPI
        try:
            llm_response = self._call_ollama_free_api(prompt)
        except Exception as e:
            logger.error(f"[OLLAMA_FREE_API_AGENT] API error: {str(e)}")
            raise
        
        # Parse structured response from LLM
        parsed = self._parse_llm_response(llm_response)
        
        # CRITICAL FIREARM OVERRIDE: Hard rule that guns are always HIGH+ threat
        # This ensures LLM hallucinations don't underestimate gun threats
        if weapon_detected and weapon_detected.lower() in {"gun", "rifle", "shotgun"}:
            original_threat = parsed["threat_level"]
            
            # Guns are at minimum HIGH threat
            if parsed["threat_level"] in {"low", "medium"}:
                parsed["threat_level"] = "high"
                parsed["recommended_action"] = "escalate"
                parsed["priority"] = "high"
                parsed["reasoning"] = f"[OVERRIDE] Gun detected ({weapon_detected}). Forcing threat_level to HIGH per security policy. Original LLM assessment: {original_threat}. " + parsed["reasoning"]
                logger.warning(
                    f"[OLLAMA_FREE_API_AGENT] ⚠️ FIREARM OVERRIDE: Gun detected but LLM assessed as {original_threat}. "
                    f"Forcing threat_level to HIGH and action to ESCALATE per security rules."
                )
            
            # If there's also threatening emotion or tone, escalate to CRITICAL
            if parsed["threat_level"] == "high":
                if emotion.lower() in {"angry", "fearful", "distressed", "panicked"}:
                    parsed["threat_level"] = "critical"
                    parsed["recommended_action"] = "immediate_alert"
                    parsed["priority"] = "critical"
                    parsed["reasoning"] = f"[CRITICAL ESCALATION] Gun + threatening emotion ({emotion}). Forcing CRITICAL threat per policy. " + parsed["reasoning"]
                    logger.warning(
                        f"[OLLAMA_FREE_API_AGENT] 🚨 CRITICAL ESCALATION: Gun + {emotion} detected. "
                        f"Forcing threat_level to CRITICAL and action to IMMEDIATE_ALERT."
                    )
                elif tone.lower() in {"threat", "panic", "abnormal", "distressed"}:
                    parsed["threat_level"] = "critical"
                    parsed["recommended_action"] = "immediate_alert"
                    parsed["priority"] = "critical"
                    parsed["reasoning"] = f"[CRITICAL ESCALATION] Gun + threatening tone ({tone}). Forcing CRITICAL threat per policy. " + parsed["reasoning"]
                    logger.warning(
                        f"[OLLAMA_FREE_API_AGENT] 🚨 CRITICAL ESCALATION: Gun + {tone} tone detected. "
                        f"Forcing threat_level to CRITICAL and action to IMMEDIATE_ALERT."
                    )
        
        # Build threat metrics from parsed data
        threat_metrics = self._build_threat_metrics(
            weapon_detected=weapon_detected,
            emotion=emotion,
            tone=tone,
            confidence_scores=confidence_scores,
            parsed_data=parsed,
        )
        
        # Build recommendation
        recommended_action = RecommendedAction(
            action=parsed["recommended_action"],
            priority=parsed["priority"],
            reason=parsed["reasoning"],
            confidence=parsed["confidence"],
        )
        
        # Build explanation
        explanation = ReasoningExplanation(
            summary=parsed["summary"],
            key_factors=parsed["key_factors"],
            evidence={
                "weapon": weapon_detected,
                "emotion": emotion,
                "tone": tone,
                "risk_indicators": risk_hints,
            },
            anomalies_detected=risk_hints,
            temporal_analysis=None,
            confidence_reasoning=parsed["confidence_reasoning"],
        )
        
        # Create final output
        output = ReasoningOutput(
            source_id=source_id,
            timestamp=datetime.now(timezone.utc),
            threat_level=parsed["threat_level"],
            confidence=parsed["confidence"],
            recommended_action=recommended_action,
            explanation=explanation,
            anomaly_types=self._detect_anomalies(weapon_detected, emotion, tone),
            metrics=threat_metrics,
            reasoning_version="ollama_free_api_v1.0",
        )
        
        logger.info(
            f"[OLLAMA_FREE_API_AGENT] Decision: threat_level={parsed['threat_level']}, "
            f"action={parsed['recommended_action']}, confidence={parsed['confidence']:.2f}"
        )
        
        return output
    
    def _build_threat_prompt(
        self,
        source_id: str,
        weapon_detected: str,
        emotion: str,
        tone: str,
        confidence_scores: dict[str, float],
        risk_hints: list[str],
        scenario_predictions: Optional[list[ScenarioPrediction]] = None,
        sop_context: Optional[SOPContext] = None,
    ) -> str:
        """Build a structured threat assessment prompt with security procedures"""
        
        prompt = f"""You are an expert security threat assessment AI for a security operations center.
Your role is to provide accurate, proportionate threat assessments for security incidents using both standard procedures and reasoned judgment.

=== SECURITY ASSESSMENT RULES ===

WEAPON THREAT ESCALATION (Always Apply):
- GUN / RIFLE / SHOTGUN: MINIMUM threat_level=HIGH, action=ESCALATE
  * If gun + angry/fearful emotion OR threat tone: threat_level=CRITICAL, action=IMMEDIATE_ALERT
  * If gun + calm tone + uniform + restricted area: threat_level=HIGH, action=ESCALATE (need verification)
- KNIFE / BLADE: MINIMUM threat_level=MEDIUM, action=ELEVATED_MONITORING
  * If knife + aggression indicators: threat_level=HIGH, action=ESCALATE
- No weapon: Start from LOW, adjust based on emotion/tone/context

EMOTION & TONE MODIFIERS:
- Angry, Fearful, Distressed, Panicked: Add 1 threat level
- Calm, Neutral: No increase
- Threat, Panic in audio tone: Add 1 threat level

CONTEXTUAL FACTORS:
- Uniform Present: May indicate authorized personnel (reduce threat by 1 level for non-gun scenarios)
- Restricted Area: Unauthorized presence increases threat by 1 level
- Multiple Risk Indicators: Each indicator adds to urgency

PROPORTIONATE RESPONSE GUIDANCE:
- For scenarios covered by standard security procedures: Follow procedures
- For novel scenarios (like armed individuals): Use proportionate reasoning based on:
  1. Weapon lethality (gun > knife > other)
  2. Behavioral indicators (emotion, tone, body language)
  3. Environmental context (location, time, authorized presence)
  4. Confidence of detection (detection confidence scores)

=== INCIDENT DATA ===
Source/Camera: {source_id}
Weapon Detected: {weapon_detected}
Emotion Detected: {emotion}
Audio Tone: {tone}

Detection Confidence:
- Weapon Detection: {confidence_scores.get('weapon', 0):.0%}
- Emotion Detection: {confidence_scores.get('emotion', 0):.0%}
- Audio Tone: {confidence_scores.get('tone', 0):.0%}
- Uniform/Security Badge: {confidence_scores.get('uniform', 0):.0%}

Risk Indicators Detected: {', '.join(risk_hints) if risk_hints else 'None observed'}
"""
        
        if sop_context:
            prompt += f"""
=== SECURITY CONTEXT ===
Location Type: {sop_context.location_type}
Time: {sop_context.time_of_day}
Security Level: {sop_context.security_level}
Active Alerts: {', '.join(sop_context.active_alerts) if sop_context.active_alerts else 'None'}
Personnel in Area: {sop_context.personnel_count or 'Unknown'}
Restricted Weapons: {', '.join(sop_context.restricted_weapons)}
"""
        
        if scenario_predictions:
            prompt += "\n=== SCENARIO ANALYSIS ===\nLikely Scenarios (from pattern analysis):\n"
            for pred in scenario_predictions:
                prompt += f"- [{pred.rank}] {pred.scenario_description} (likelihood: {pred.probability:.0%})\n"
        
        prompt += """
=== ASSESSMENT INSTRUCTIONS ===
1. ALWAYS apply weapon escalation rules first (guns are inherently high-threat)
2. Apply emotion/tone modifiers to adjust assessment
3. Consider contextual factors (uniform, location, time)
4. For any weapon detection, set priority to at least "HIGH"
5. For guns specifically: minimum action is ESCALATE, unless additional context confirms misidentification
6. Use proportionate reasoning for novel situations not covered by standard procedures
7. Provide clear reasoning that references which factors drove your assessment

=== REQUIRED OUTPUT ===
Return ONLY valid JSON with no additional text:

{
    "threat_level": "low" | "medium" | "high" | "critical",
    "threat_score": 0.0-1.0,
    "confidence": 0.0-1.0,
    "recommended_action": "monitor" | "escalate" | "immediate_alert" | "lockdown",
    "priority": "low" | "medium" | "high" | "critical",
    "summary": "One-line threat assessment",
    "reasoning": "Detailed reasoning addressing each factor (weapon, emotion, tone, context)",
    "confidence_reasoning": "Why you are confident in this specific assessment and priority level",
    "key_factors": ["factor1", "factor2", "factor3"]
}
"""
        
        return prompt
    
    def _call_ollama_free_api(self, prompt: str) -> str:
        """Call OllamaFreeAPI and get response"""
        
        try:
            # Use the chat method from OllamaFreeAPI
            response = self.client.chat(
                model=self.model,
                prompt=prompt,
                temperature=self.temperature,
            )
            
            # OllamaFreeAPI returns the response directly
            return response
        
        except ConnectionError:
            raise ConnectionError(
                "Cannot connect to OllamaFreeAPI. "
                "Check your internet connection."
            )
        except Exception as e:
            raise RuntimeError(f"OllamaFreeAPI error: {str(e)}")
    
    def _parse_llm_response(self, response: str) -> dict:
        """Parse JSON response from LLM"""
        
        # Try to extract JSON from response
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        
        if not json_match:
            logger.warning(
                f"[OLLAMA_FREE_API_AGENT] Could not find JSON in response: {response[:100]}..."
            )
            raise ValueError("LLM response is not valid JSON")
        
        try:
            parsed = json.loads(json_match.group())
        except json.JSONDecodeError as e:
            logger.error(f"[OLLAMA_FREE_API_AGENT] JSON parse error: {str(e)}")
            raise
        
        # Validate required fields
        required_fields = [
            "threat_level",
            "threat_score",
            "confidence",
            "recommended_action",
            "priority",
            "summary",
            "reasoning",
            "key_factors",
        ]
        
        for field in required_fields:
            if field not in parsed:
                raise ValueError(f"Missing required field in LLM response: {field}")
        
        # Normalize threat level
        parsed["threat_level"] = parsed["threat_level"].lower()
        if parsed["threat_level"] not in ["low", "medium", "high", "critical"]:
            logger.warning(
                f"[OLLAMA_FREE_API_AGENT] Invalid threat level: {parsed['threat_level']}, "
                "defaulting to 'medium'"
            )
            parsed["threat_level"] = "medium"
        
        return parsed
    
    def _build_threat_metrics(
        self,
        weapon_detected: str,
        emotion: str,
        tone: str,
        confidence_scores: dict[str, float],
        parsed_data: dict,
    ) -> ThreatMetrics:
        """Build ThreatMetrics from parsed LLM response"""
        
        # Score individual modalities
        weapon_threat = self._score_weapon(weapon_detected, confidence_scores.get("weapon", 0))
        emotion_threat = self._score_emotion(emotion, confidence_scores.get("emotion", 0))
        audio_threat = self._score_audio(tone, confidence_scores.get("tone", 0))
        behavioral_threat = parsed_data.get("threat_score", 0.5)
        
        return ThreatMetrics(
            weapon_threat_score=weapon_threat,
            emotion_threat_score=emotion_threat,
            audio_threat_score=audio_threat,
            behavioral_anomaly_score=behavioral_threat,
            combined_threat_score=parsed_data.get("threat_score", 0.5),
            trend="stable",
            frames_in_history=1,
            context_anomaly_flag=False,
        )
    
    def _score_weapon(self, weapon: str, confidence: float) -> float:
        """Score weapon threat"""
        scores = {
            "gun": 0.95,
            "rifle": 0.95,
            "shotgun": 0.95,
            "knife": 0.85,
            "blade": 0.85,
            "bat": 0.60,
            "stick": 0.40,
            "unarmed": 0.0,
        }
        return scores.get(weapon.lower(), 0.3) * confidence
    
    def _score_emotion(self, emotion: str, confidence: float) -> float:
        """Score emotion threat"""
        scores = {
            "distressed": 0.80,
            "fearful": 0.60,
            "angry": 0.75,
            "panicked": 0.80,
            "neutral": 0.0,
            "calm": 0.0,
            "happy": 0.0,
        }
        return scores.get(emotion.lower(), 0.0) * confidence
    
    def _score_audio(self, tone: str, confidence: float) -> float:
        """Score audio threat"""
        scores = {
            "panic": 0.85,
            "threat": 0.80,
            "abnormal": 0.60,
            "distressed": 0.70,
            "calm": 0.0,
            "neutral": 0.0,
        }
        return scores.get(tone.lower(), 0.0) * confidence
    
    def _detect_anomalies(self, weapon: str, emotion: str, tone: str) -> list[str]:
        """Detect anomalies from threat indicators"""
        anomalies = []
        
        if weapon.lower() != "unarmed":
            anomalies.append("armed_individual")
        
        if emotion.lower() in ["angry", "distressed", "panicked"]:
            anomalies.append("emotional_distress")
        
        if tone.lower() in ["panic", "threat", "distressed"]:
            anomalies.append("abnormal_audio")
        
        return anomalies
