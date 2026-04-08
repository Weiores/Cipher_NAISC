"""
Ollama Reasoning Agent - LLM-powered decision-making using local Ollama.

DEPLOYMENT: Local machine running Ollama (supports cloud server too)

This agent uses a local Large Language Model (via Ollama) to perform reasoning
on threat assessment. It provides more nuanced decision-making compared to 
rule-based approaches, while maintaining interpretability.

Features:
- Uses Ollama local/cloud LLM for reasoning
- Graceful fallback when Ollama unavailable
- Structured output parsing (threat level, action, confidence)
- Same interface as CloudReasoningAgent
- Async support for production use

Latency: ~200-800ms per decision (depends on model size)
Accuracy: High (LLM-based reasoning)
Resources: Moderate (depends on model)
"""

import asyncio
import json
import logging
import re
from datetime import datetime, timezone
from typing import Optional

import requests

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


class OllamaReasoningAgent:
    """
    LLM-powered reasoning agent using Ollama.
    
    FEATURES:
    - Uses local Ollama for threat assessment reasoning
    - Compatible with various models (mistral, neural-chat, dolphin-mixtral, etc.)
    - Falls back gracefully if Ollama is unavailable
    - Parses structured threat recommendations from LLM output
    - Maintains compatibility with CloudReasoningAgent interface
    
    USAGE:
        agent = OllamaReasoningAgent(
            ollama_url="http://localhost:11434",
            model="mistral"
        )
        
        output = agent.reason(
            source_id="camera_1",
            weapon_detected="gun",
            emotion="angry",
            tone="threat",
            confidence_scores={"weapon": 0.95, "emotion": 0.8, "tone": 0.9},
            risk_hints=["visible_weapon", "emotional_escalation"]
        )
    """
    
    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        model: str = "mistral",
        #model: str = "gpt-oss:20b",
        timeout: int = 30,
        temperature: float = 0.3,
    ):
        """
        Initialize Ollama Reasoning Agent.
        
        Args:
            ollama_url: URL where Ollama is running (default: localhost)
            model: Ollama model to use (e.g., "mistral", "neural-chat")
            timeout: Timeout for Ollama API calls (seconds)
            temperature: LLM temperature for reasoning (0.0-1.0, lower = more deterministic)
        """
        self.ollama_url = ollama_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        
        self._is_available = None  # Cache availability check
        logger.info(
            f"[OLLAMA_AGENT] Initialized with model={model}, url={ollama_url}"
        )
    
    def is_available(self) -> bool:
        """Check if Ollama server is reachable"""
        if self._is_available is not None:
            return self._is_available
        
        try:
            response = requests.get(
                f"{self.ollama_url}/api/tags",
                timeout=self.timeout,
            )
            self._is_available = response.status_code == 200
            logger.info(f"[OLLAMA_AGENT] Availability check: {self._is_available}")
            return self._is_available
        except Exception as e:
            logger.warning(f"[OLLAMA_AGENT] Ollama not available: {str(e)}")
            self._is_available = False
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
        Perform threat reasoning using Ollama LLM.
        
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
            logger.warning(
                "[OLLAMA_AGENT] Ollama not available, cannot reason. "
                "Check Ollama is running: ollama serve"
            )
            raise ConnectionError(
                f"Ollama server not available at {self.ollama_url}. "
                "Is it running? Try: ollama serve"
            )
        
        logger.info(
            f"[OLLAMA_AGENT] Reasoning for source={source_id}, "
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
        
        # Call Ollama API
        try:
            llm_response = self._call_ollama(prompt)
        except Exception as e:
            logger.error(f"[OLLAMA_AGENT] Ollama API error: {str(e)}")
            raise
        
        # Parse structured response from LLM
        parsed = self._parse_llm_response(llm_response)
        
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
            reasoning_version="ollama_v1.0",
        )
        
        logger.info(
            f"[OLLAMA_AGENT] Decision: threat_level={parsed['threat_level']}, "
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
    
    def _call_ollama(self, prompt: str) -> str:
        """Call Ollama API and get response"""
        
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": self.temperature,
                },
                timeout=self.timeout,
            )
            
            response.raise_for_status()
            result = response.json()
            
            return result.get("response", "")
        
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.ollama_url}. "
                "Ensure Ollama is running: ollama serve"
            )
        except requests.exceptions.Timeout:
            raise TimeoutError(
                f"Ollama request timed out after {self.timeout}s. "
                "Try increasing timeout or using a smaller model."
            )
    
    def _parse_llm_response(self, response: str) -> dict:
        """Parse JSON response from LLM"""
        
        # Try to extract JSON from response
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        
        if not json_match:
            logger.warning(
                f"[OLLAMA_AGENT] Could not find JSON in response: {response[:100]}..."
            )
            raise ValueError("LLM response is not valid JSON")
        
        try:
            parsed = json.loads(json_match.group())
        except json.JSONDecodeError as e:
            logger.error(f"[OLLAMA_AGENT] JSON parse error: {str(e)}")
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
                f"[OLLAMA_AGENT] Invalid threat level: {parsed['threat_level']}, "
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
    
    def analyze_detection(self, detection_data):
        """Analyze detection data and provide recommended actions for real-time frame analysis"""
        try:
            # Extract detection information
            weapon_detected = detection_data.get('weapon_detected', False) or \
                            detection_data.get('Weapon Detected', False)
            threat_level = detection_data.get('threat_level', 'UNKNOWN') or \
                          detection_data.get('Threat Level', 'UNKNOWN')
            confidence = detection_data.get('confidence', 0)
            
            # Build a simple prompt for frame-by-frame analysis
            prompt = f"""You are a security threat assessment AI analyzing a video frame.

DETECTION: Weapon={weapon_detected}, Threat={threat_level}, Confidence={confidence}

Provide ONLY valid JSON response with these exact fields:
{{"action": "ACTION", "priority": "PRIORITY", "confidence": "PCT%", "reasoning": "BRIEF", "key_factors": ["FACTOR"], "detected_anomalies": ["ANOMALY"]}}"""
            
            # Try to call Ollama if available
            if self.is_available():
                llm_response = self._call_ollama(prompt)
                recommendation = self._parse_reasoning_response(llm_response, detection_data, threat_level)
            else:
                # Fallback without LLM
                recommendation = self._create_fallback_recommendation(detection_data, threat_level)
            
            return recommendation
            
        except Exception as e:
            logger.error(f"[OLLAMA_AGENT] Error analyzing detection: {str(e)}")
            return {
                'action': 'MONITOR',
                'priority': 'UNKNOWN',
                'confidence': '0%',
                'reasoning': f'Analysis error: {str(e)}',
                'key_factors': [],
                'detected_anomalies': []
            }
    
    def _parse_reasoning_response(self, response_text, detection_data, threat_level):
        """Parse LLM response into structured recommendation"""
        try:
            # Find JSON in response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                return {
                    'action': parsed.get('action', 'MONITOR'),
                    'priority': parsed.get('priority', 'UNKNOWN'),
                    'confidence': parsed.get('confidence', '—'),
                    'reasoning': parsed.get('reasoning', 'Analysis complete'),
                    'key_factors': parsed.get('key_factors', []),
                    'detected_anomalies': parsed.get('detected_anomalies', [])
                }
        except Exception as e:
            logger.warning(f"[OLLAMA_AGENT] Error parsing JSON: {str(e)}")
        
        # Fallback based on threat level
        return self._create_fallback_recommendation(detection_data, threat_level)
    
    def _create_fallback_recommendation(self, detection_data, threat_level):
        """Create a fallback recommendation based on threat level"""
        weapon_detected = detection_data.get('weapon_detected', False) or \
                        detection_data.get('Weapon Detected', False)
        confidence = detection_data.get('confidence', 0)
        
        if weapon_detected and threat_level in ['CRITICAL', 'HIGH']:
            return {
                'action': 'ALERT',
                'priority': threat_level,
                'confidence': f"{int(confidence * 100)}%" if isinstance(confidence, float) else "—",
                'reasoning': f'Weapon detected with {threat_level} threat level. Immediate response required.',
                'key_factors': [f'Weapon type: {weapon_detected}', f'Threat Level: {threat_level}'],
                'detected_anomalies': ['Armed threat detected']
            }
        else:
            return {
                'action': 'MONITOR',
                'priority': 'LOW',
                'confidence': '—',
                'reasoning': 'No critical threats detected',
                'key_factors': [],
                'detected_anomalies': []
            }
    
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


class OllamaReasoningService:
    """
    Wrapper service for Ollama reasoning with retry logic and caching.
    
    Provides with enhanced reliability:
    - Automatic retry on transient failures
    - Caching of Ollama availability
    - Graceful error messages
    """
    
    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        model: str = "mistral",
        timeout: int = 30,
        max_retries: int = 2,
    ):
        self.agent = OllamaReasoningAgent(
            ollama_url=ollama_url,
            model=model,
            timeout=timeout,
        )
        self.max_retries = max_retries
    
    def reason_with_retry(
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
        Perform reasoning with automatic retry on failure.
        
        Retries on transient errors but fails fast on connection errors
        (which indicate Ollama is not running).
        """
        
        for attempt in range(self.max_retries):
            try:
                return self.agent.reason(
                    source_id=source_id,
                    weapon_detected=weapon_detected,
                    emotion=emotion,
                    tone=tone,
                    confidence_scores=confidence_scores,
                    risk_hints=risk_hints,
                    scenario_predictions=scenario_predictions,
                    sop_context=sop_context,
                )
            except ConnectionError:
                # Connection errors are not retryable
                raise
            except Exception as e:
                if attempt < self.max_retries - 1:
                    logger.warning(
                        f"[OLLAMA_SERVICE] Attempt {attempt + 1} failed: {str(e)}, retrying..."
                    )
                    asyncio.sleep(0.5)  # Brief delay before retry
                else:
                    raise
