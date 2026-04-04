"""
Cloud Reasoning Service - Integration layer between Perception and Reasoning layers.

DEPLOYMENT: Cloud Server

This service:
1. Receives fused traits from the perception layer
2. Optionally integrates with Learning Agent (for scenario predictions)
3. Routes through orchestration layer (cloud vs local)
4. Returns enhanced reasoning output to the dashboard
5. Can be called via REST API or directly

Usage:
    service = CloudReasoningService()
    
    # From perception layer
    fused_traits = {...}
    
    # Optional: Get scenarios from learning agent
    scenarios = learning_agent.predict_scenarios(fused_traits)
    
    # Get reasoning decision
    reasoning_output = await service.reason(
        source_id="camera_1",
        weapon_detected="gun",
        emotion="angry",
        tone="threat",
        confidence_scores={...},
        risk_hints=[...],
        scenario_predictions=scenarios,
    )
"""

import asyncio
import logging
from typing import Callable, Optional

from schemas import ReasoningOutput
from orchestrator import (
    ReasoningOrchestrator,
    ConnectivityChecker,
    create_orchestrator_with_defaults,
)
from .cloud_reasoning_agent import (
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
    - Cloud Reasoning Agent (for enhanced decision-making)
    - Local Fallback Agent (for degraded mode)
    - Orchestration (routing logic)
    - Learning Agent integration (optional scenario predictions)
    """
    
    def __init__(
        self,
        orchestrator: Optional[ReasoningOrchestrator] = None,
        learning_agent_callable: Optional[Callable] = None,
        sop_context_provider: Optional[Callable] = None,
    ):
        """
        Initialize the cloud reasoning service.
        
        Args:
            orchestrator: Optional custom orchestrator (or uses default)
            learning_agent_callable: Optional function to get scenario predictions
            sop_context_provider: Optional function to get SOP context
        """
        
        if orchestrator:
            self.orchestrator = orchestrator
        else:
            # Create default local reasoning function
            def default_local_reasoning(
                source_id, weapon, emotion, tone, confidence_scores, risk_hints
            ):
                # Stub: just return a basic recommended action
                from ..schemas import ReasoningOutput, RecommendedAction, ThreatMetrics, ReasoningExplanation
                from datetime import datetime, timezone
                
                threat_score = sum(confidence_scores.values()) / len(confidence_scores) if confidence_scores else 0.5
                
                if threat_score > 0.7:
                    action = "immediate_alert"
                    priority = "critical"
                elif threat_score > 0.4:
                    action = "escalate"
                    priority = "high"
                else:
                    action = "monitor"
                    priority = "low"
                
                return ReasoningOutput(
                    source_id=source_id,
                    timestamp=datetime.now(timezone.utc),
                    threat_level="high" if threat_score > 0.4 else "low",
                    confidence=threat_score,
                    recommended_action=RecommendedAction(
                        action=action,
                        priority=priority,
                        reason="Local fallback reasoning (SOP-only)",
                        confidence=threat_score,
                    ),
                    explanation=ReasoningExplanation(
                        summary="Local fallback mode - cloud unavailable",
                        key_factors=[],
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
                    reasoning_version="local_fallback_v1.0",
                )
            
            self.orchestrator = create_orchestrator_with_defaults(
                local_reasoning_fn=default_local_reasoning,
            )
        
        self.learning_agent = learning_agent_callable
        self.sop_context_provider = sop_context_provider
        
        logger.info("[CLOUD_REASONING_SERVICE] Initialized")
    
    async def reason(
        self,
        source_id: str,
        weapon_detected: str,
        emotion: str,
        tone: str,
        confidence_scores: dict[str, float],
        risk_hints: list[str],
        use_learning_agent: bool = True,
        use_sop_context: bool = True,
        force_cloud: bool = False,
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
            force_cloud: Force cloud routing
        
        Returns:
            ReasoningOutput with complete decision
        """
        
        logger.info(f"[CLOUD_REASONING_SERVICE] Reasoning request: source={source_id}, weapon={weapon_detected}, emotion={emotion}, tone={tone}")
        
        # Get optional scenario predictions from learning agent
        scenario_predictions = None
        if use_learning_agent and self.learning_agent:
            try:
                logger.info(f"[CLOUD_REASONING_SERVICE] Querying learning agent for scenarios")
                scenario_predictions = await self._get_scenarios_from_learning_agent(
                    source_id, weapon_detected, emotion, tone, risk_hints
                )
                logger.info(f"[CLOUD_REASONING_SERVICE] Received {len(scenario_predictions) if scenario_predictions else 0} scenarios from learning agent")
            except Exception as e:
                logger.error(f"[CLOUD_REASONING_SERVICE] Error querying learning agent: {e}")
                scenario_predictions = None
        
        # Get optional SOP context
        sop_context = None
        if use_sop_context and self.sop_context_provider:
            try:
                logger.info(f"[CLOUD_REASONING_SERVICE] Getting SOP context")
                sop_context = await self._get_sop_context(source_id)
                logger.info(f"[CLOUD_REASONING_SERVICE] Got SOP context: location={sop_context.location_type if sop_context else None}")
            except Exception as e:
                logger.error(f"[CLOUD_REASONING_SERVICE] Error getting SOP context: {e}")
                sop_context = None
        
        # Route through orchestrator (cloud vs local)
        reasoning_output = await self.orchestrator.reason(
            source_id=source_id,
            weapon_detected=weapon_detected,
            emotion=emotion,
            tone=tone,
            confidence_scores=confidence_scores,
            risk_hints=risk_hints,
            scenario_predictions=scenario_predictions,
            sop_context=sop_context,
            force_cloud=force_cloud,
        )
        
        logger.info(f"[CLOUD_REASONING_SERVICE] Decision made: threat_level={reasoning_output.threat_level}, action={reasoning_output.recommended_action.action}, confidence={reasoning_output.confidence:.2f}")
        
        return reasoning_output
    
    async def _get_scenarios_from_learning_agent(
        self,
        source_id: str,
        weapon: str,
        emotion: str,
        tone: str,
        risk_hints: list[str],
    ) -> Optional[list[ScenarioPrediction]]:
        """
        Query learning agent for scenario predictions.
        
        This would be called when:
        - Threat level is MEDIUM or above
        - We have enough historical data
        - Cloud connection is healthy enough
        
        Returns:
            List of top 3 scenario predictions, or None if unavailable
        """
        
        if not self.learning_agent:
            return None
        
        try:
            # Prepare input for learning agent
            learning_input = {
                "source_id": source_id,
                "weapon": weapon,
                "emotion": emotion,
                "tone": tone,
                "risk_hints": risk_hints,
            }
            
            # Call learning agent (async or sync depending on implementation)
            result = await asyncio.to_thread(
                self.learning_agent,
                learning_input
            )
            
            return result if isinstance(result, list) else None
            
        except Exception as e:
            logger.error(f"[CLOUD_REASONING_SERVICE] Error getting scenarios: {e}")
            return None
    
    async def _get_sop_context(self, source_id: str) -> Optional[SOPContext]:
        """
        Get SOP context for the given source.
        
        Would lookup current location, security level, active alerts, etc.
        """
        
        if not self.sop_context_provider:
            return None
        
        try:
            context = await asyncio.to_thread(
                self.sop_context_provider,
                source_id
            )
            
            return context if isinstance(context, SOPContext) else None
            
        except Exception as e:
            logger.error(f"[CLOUD_REASONING_SERVICE] Error getting SOP context: {e}")
            return None
    
    def get_connectivity_status(self) -> dict:
        """Get current cloud connectivity status"""
        return self.orchestrator.connectivity_checker.get_status()
    
    def get_recent_decisions(self, limit: int = 50) -> list[dict]:
        """Get recent orchestration decisions (audit trail)"""
        return self.orchestrator.get_decision_log(limit=limit)
