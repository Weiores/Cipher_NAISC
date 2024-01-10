"""
Cloud Reasoning Service - Integration layer between Perception and Reasoning layers.

DEPLOYMENT: Cloud Server or Cloud-Hosted LLM

This service:
1. Receives fused traits from the perception layer
2. Optionally integrates with Learning Agent (for scenario predictions)
3. Routes through orchestration layer (cloud vs local)
4. Returns enhanced reasoning output to the dashboard
5. Can be called via REST API or directly

Supports multiple cloud reasoning providers:
- CloudReasoningAgent (traditional rule-based)
- OllamaFreeAPI (free cloud LLM - no setup)
- Local Ollama (self-hosted)

Usage (Cloud LLM):
    service = CloudReasoningService(cloud_provider="ollama_free_api")
    
    # Or with custom options
    service = CloudReasoningService(
        cloud_provider="ollama_free_api",
        cloud_config={"model": "mistral:latest"}
    )
    
    reasoning_output = await service.reason(
        source_id="camera_1",
        weapon_detected="gun",
        emotion="angry",
        tone="threat",
        confidence_scores={...},
        risk_hints=[...],
    )
"""

import asyncio
import logging
from typing import Callable, Optional, Literal
from datetime import datetime, timezone

from schemas import ReasoningOutput, RecommendedAction, ThreatMetrics, ReasoningExplanation
from cloud_reasoning_agent import (
    CloudReasoningAgent,
    ScenarioPrediction,
    SOPContext,
)


logger = logging.getLogger(__name__)


class CloudReasoningService:
    """
    Main service for cloud reasoning. Acts as the interface between
    perception layer and decision output layer.
    
    Integrates:
    - Cloud Reasoning Providers (CloudReasoningAgent, OllamaFreeAPI, etc.)
    - Local Fallback Agent (for degraded mode)
    - Orchestration (routing logic)
    - Learning Agent integration (optional scenario predictions)
    
    Cloud Providers:
    - "cloud_agent" (default): Traditional rule-based CloudReasoningAgent
    - "ollama_free_api": Free cloud-hosted LLM (no setup needed)
    - "ollama_local": Local Ollama server
    """
    
    def __init__(
        self,
        learning_agent_callable: Optional[Callable] = None,
        sop_context_provider: Optional[Callable] = None,
        cloud_provider: Literal["cloud_agent", "ollama_free_api", "ollama_local"] = "cloud_agent",
        cloud_config: Optional[dict] = None,
    ):
        """
        Initialize the cloud reasoning service.
        
        Args:
            learning_agent_callable: Optional function to get scenario predictions
            sop_context_provider: Optional function to get SOP context
            cloud_provider: Which cloud provider to use
                - "cloud_agent": Rule-based reasoning (default)
                - "ollama_free_api": Free LLM API (no setup)
                - "ollama_local": Local Ollama server
            cloud_config: Configuration dict for the cloud provider
                - For "ollama_free_api": {"model": "llama3.2:3b"}
                - For "ollama_local": {"url": "http://localhost:11434", "model": "mistral"}
        """
        
        self.cloud_provider = cloud_provider
        self.cloud_config = cloud_config or {}
        self.learning_agent = learning_agent_callable
        self.sop_context_provider = sop_context_provider
        
        # Initialize the cloud reasoning agent
        self.cloud_agent = self._create_cloud_agent(cloud_provider, self.cloud_config)
        
        # Initialize fallback local agent
        self.local_agent = CloudReasoningAgent()
        
        logger.info(f"[CLOUD_REASONING_SERVICE] Initialized with cloud_provider={cloud_provider}")
    
    def _create_cloud_agent(self, provider: str, config: dict):
        """Create the cloud reasoning agent based on provider"""
        
        if provider == "cloud_agent":
            return CloudReasoningAgent()
        
        elif provider == "ollama_free_api":
            from ollama_free_api_agent import OllamaFreeAPIReasoningAgent
            model = config.get("model", "llama3.2:3b")
            temperature = config.get("temperature", 0.3)
            return OllamaFreeAPIReasoningAgent(model=model, temperature=temperature)
        
        elif provider == "ollama_local":
            from ollama_reasoning_agent import OllamaReasoningAgent
            ollama_url = config.get("url", "http://localhost:11434")
            model = config.get("model", "mistral")
            return OllamaReasoningAgent(ollama_url=ollama_url, model=model)
        
        else:
            raise ValueError(f"Unknown cloud provider: {provider}")
    

    
    async def reason(
        self,
        source_id: str,
        weapon_detected: str,
        emotion: str,
        tone: str,
        confidence_scores: dict,
        risk_hints: list,
        use_learning_agent: bool = False,
        use_sop_context: bool = False,
        force_local_only: bool = False,
    ) -> ReasoningOutput:
        """
        Main method to get reasoning decision.
        
        Args:
            source_id: Camera/sensor ID
            weapon_detected: Weapon type from perception
            emotion: Emotion from perception
            tone: Tone from perception
            confidence_scores: Confidence dict from perception
            risk_hints: Risk indicators from perception
            use_learning_agent: Whether to integrate learning agent predictions
            use_sop_context: Whether to apply SOP context
            force_local_only: Force use of local fallback agent
        
        Returns:
            ReasoningOutput with complete decision
        """
        
        logger.info(f"[CLOUD_REASONING_SERVICE] Reasoning request: source={source_id}, weapon={weapon_detected}, emotion={emotion}, tone={tone}")
        
        # Get optional scenario predictions
        scenario_predictions = None
        if use_learning_agent and self.learning_agent:
            try:
                scenario_predictions = await self._get_scenarios_from_learning_agent(
                    source_id, weapon_detected, emotion, tone, risk_hints
                )
            except Exception as e:
                logger.error(f"[CLOUD_REASONING_SERVICE] Error with learning agent: {e}")
        
        # Get optional SOP context
        sop_context = None
        if use_sop_context and self.sop_context_provider:
            try:
                sop_context = await self._get_sop_context(source_id)
            except Exception as e:
                logger.error(f"[CLOUD_REASONING_SERVICE] Error getting SOP context: {e}")
        
        # Try cloud reasoning first, fallback to local if needed
        reasoning_output = None
        used_provider = self.cloud_provider
        
        if not force_local_only:
            try:
                logger.info(f"[CLOUD_REASONING_SERVICE] ========== ATTEMPTING {self.cloud_provider.upper()} ==========")
                logger.info(f"[CLOUD_REASONING_SERVICE] Calling {self.cloud_provider} with 40s timeout")
                logger.info(f"[CLOUD_REASONING_SERVICE] Inputs: weapon={weapon_detected}, emotion={emotion}, tone={tone}")
                reasoning_output = await asyncio.wait_for(
                    asyncio.to_thread(
                        self.cloud_agent.reason,
                        source_id,
                        weapon_detected,
                        emotion,
                        tone,
                        confidence_scores,
                        risk_hints,
                        scenario_predictions,
                        sop_context,
                    ),
                    timeout=40.0  # 40 second timeout
                )
                logger.info(f"[CLOUD_REASONING_SERVICE] ✓ {self.cloud_provider} successful: threat_level={reasoning_output.threat_level}, action={reasoning_output.recommended_action.action}")
            except asyncio.TimeoutError:
                logger.warning(f"[CLOUD_REASONING_SERVICE] ✗ {self.cloud_provider} TIMEOUT (>40s), falling back to local")
                reasoning_output = None
            except Exception as e:
                logger.warning(f"[CLOUD_REASONING_SERVICE] ✗ {self.cloud_provider} failed: {str(e)[:200]}")
                logger.warning(f"[CLOUD_REASONING_SERVICE] Full error: {str(e)}")
                reasoning_output = None
        
        # Fallback to local agent if cloud failed
        if reasoning_output is None:
            try:
                logger.info(f"[CLOUD_REASONING_SERVICE] ========== FALLBACK to LOCAL AGENT ==========")
                logger.info("[CLOUD_REASONING_SERVICE] Using local rule-based CloudReasoningAgent")
                reasoning_output = await asyncio.to_thread(
                    self.local_agent.reason,
                    source_id,
                    weapon_detected,
                    emotion,
                    tone,
                    confidence_scores,
                    risk_hints,
                    scenario_predictions,
                    sop_context,
                )
                used_provider = "cloud_agent (fallback)"
                logger.info(f"[CLOUD_REASONING_SERVICE] ✓ Fallback agent succeeded: threat_level={reasoning_output.threat_level}")
            except Exception as e:
                logger.error(f"[CLOUD_REASONING_SERVICE] ✗ Local fallback also failed: {e}")
                # Last resort: return basic decision
                reasoning_output = self._create_basic_decision(
                    source_id, weapon_detected, emotion, tone, confidence_scores, risk_hints
                )
                used_provider = "basic_fallback"
                logger.warning(f"[CLOUD_REASONING_SERVICE] ✗ Using basic fallback decision")
        
        logger.info(f"[CLOUD_REASONING_SERVICE] ========== FINAL DECISION ==========")
        logger.info(f"[CLOUD_REASONING_SERVICE] Provider: {used_provider}")
        logger.info(f"[CLOUD_REASONING_SERVICE] Decision: threat_level={reasoning_output.threat_level}, action={reasoning_output.recommended_action.action}, confidence={reasoning_output.confidence:.2f}")
        
        return reasoning_output
    
    def _create_basic_decision(
        self,
        source_id: str,
        weapon_detected: str,
        emotion: str,
        tone: str,
        confidence_scores: dict,
        risk_hints: list,
    ) -> ReasoningOutput:
        """Create a basic decision when all providers fail"""
        
        threat_score = sum(confidence_scores.values()) / len(confidence_scores) if confidence_scores else 0.5
        
        if threat_score > 0.7:
            threat_level = "critical"
            action = "immediate_alert"
            priority = "critical"
        elif threat_score > 0.4:
            threat_level = "high"
            action = "escalate"
            priority = "high"
        else:
            threat_level = "medium"
            action = "monitor"
            priority = "medium"
        
        return ReasoningOutput(
            source_id=source_id,
            timestamp=datetime.now(timezone.utc),
            threat_level=threat_level,
            confidence=threat_score,
            recommended_action=RecommendedAction(
                action=action,
                priority=priority,
                reason="Basic fallback decision",
                confidence=threat_score,
            ),
            explanation=ReasoningExplanation(
                summary="All reasoning providers failed, using basic decision",
                key_factors=[f"Weapon: {weapon_detected}", f"Emotion: {emotion}", f"Tone: {tone}"],
                evidence={},
                anomalies_detected=[],
                temporal_analysis=None,
                confidence_reasoning="Using threat score from perception layer",
            ),
            anomaly_types=[],
            metrics=ThreatMetrics(
                weapon_threat_score=confidence_scores.get("weapon", 0),
                emotion_threat_score=confidence_scores.get("emotion", 0),
                audio_threat_score=confidence_scores.get("tone", 0),
                behavioral_anomaly_score=0.0,
                combined_threat_score=threat_score,
                trend="stable",
                frames_in_history=1,
            ),
            reasoning_version="fallback_v1.0",
        )
    
    async def _get_scenarios_from_learning_agent(
        self,
        source_id: str,
        weapon: str,
        emotion: str,
        tone: str,
        risk_hints: list,
    ) -> Optional[list]:
        """Query learning agent for scenario predictions"""
        
        if not self.learning_agent:
            return None
        
        try:
            learning_input = {
                "source_id": source_id,
                "weapon": weapon,
                "emotion": emotion,
                "tone": tone,
                "risk_hints": risk_hints,
            }
            
            result = await asyncio.to_thread(self.learning_agent, learning_input)
            return result if isinstance(result, list) else None
            
        except Exception as e:
            logger.error(f"[CLOUD_REASONING_SERVICE] Error getting scenarios: {e}")
            return None
    
    async def _get_sop_context(self, source_id: str) -> Optional[SOPContext]:
        """Get SOP context for the given source"""
        
        if not self.sop_context_provider:
            return None
        
        try:
            result = await asyncio.to_thread(self.sop_context_provider, source_id)
            return result if isinstance(result, SOPContext) else None
        except Exception as e:
            logger.error(f"[CLOUD_REASONING_SERVICE] Error getting SOP context: {e}")
            return None
    
    def get_cloud_provider_info(self) -> dict:
        """Get information about the current cloud provider"""
        
        descriptions = {
            "cloud_agent": "Rule-based reasoning (CloudReasoningAgent)",
            "ollama_free_api": "Free cloud-hosted LLM (OllamaFreeAPI)",
            "ollama_local": "Local Ollama server",
        }
        
        return {
            "provider": self.cloud_provider,
            "description": descriptions.get(self.cloud_provider, "Unknown"),
        }
