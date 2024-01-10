"""
Reasoning Layer Adapter - Bridges Perception and Cloud Reasoning layers

This module integrates the new Cloud Reasoning Agent with the perception layer,
handling perception output conversion and reasoning result formatting.
"""

import sys
import logging
import asyncio
from pathlib import Path
from datetime import datetime, timezone

# Setup paths for reasoning layer imports
CURRENT_DIR = Path(__file__).parent.absolute()
# Go up 3 levels: app -> security-perception-layer -> perception-layer -> Cipher_NAISC
REASONING_LAYER = CURRENT_DIR.parent.parent.parent / "reasoning-layer"
CLOUD_REASONING = REASONING_LAYER / "cloud_reasoning"

# Add reasoning layer to path
sys.path.insert(0, str(REASONING_LAYER))
sys.path.insert(0, str(CLOUD_REASONING))

logger = logging.getLogger(__name__)

# Import reasoning components
try:
    from cloud_reasoning_service import CloudReasoningService
    from cloud_reasoning_agent import SOPContext
    from schemas import ReasoningOutput
    REASONING_AVAILABLE = True
    logger.info("[ADAPTER] Successfully imported Cloud Reasoning Service")
except ImportError as e:
    logger.error(f"[ADAPTER] Could not import reasoning layer (attempt 1): {e}")
    # Try alternate import path
    try:
        import sys
        sys.path.insert(0, str(CLOUD_REASONING))
        from cloud_reasoning_service import CloudReasoningService
        from cloud_reasoning_agent import SOPContext
        from schemas import ReasoningOutput
        REASONING_AVAILABLE = True
        logger.info("[ADAPTER] Successfully imported Cloud Reasoning Service (alternate path)")
    except ImportError as e2:
        logger.error(f"[ADAPTER] Could not import reasoning layer (attempt 2): {e2}")
        REASONING_AVAILABLE = False
        CloudReasoningService = None


class ReasoningAdapter:
    """
    Adapter for integrating Cloud Reasoning Agent with perception layer.
    
    Converts perception output to reasoning input format and returns
    threat assessment with recommendations.
    """
    
    def __init__(self, cloud_provider="ollama_free_api"):
        """Initialize the Cloud Reasoning Service
        
        Args:
            cloud_provider: Which cloud provider to use:
                - "ollama_free_api": Free cloud LLM (recommended - no setup needed)
                - "ollama_local": Local Ollama server (requires: ollama serve)
                - "cloud_agent": Rule-based fallback (fast, reliable)
        """
        if not REASONING_AVAILABLE:
            raise RuntimeError("Cloud Reasoning Service not available")
        
        # Initialize with specified cloud provider (with automatic fallback)
        self.service = CloudReasoningService(cloud_provider=cloud_provider)
        self.is_ready = True
        self.call_count = 0
        self.last_reasoning = None
        
        logger.info(f"[ADAPTER] Cloud Reasoning Adapter initialized with provider: {cloud_provider}")
    
    async def process_perception_async(self, perception_output):
        """
        Process perception output through cloud reasoning layer (async).
        
        Args:
            perception_output: UnifiedPerceptionOutput from perception layer
        
        Returns:
            ReasoningOutput with threat assessment and recommendations
        """
        if not self.is_ready:
            raise RuntimeError("Reasoning adapter not initialized")
        
        self.call_count += 1
        logger.info(f"[ADAPTER] Processing perception #{self.call_count} for source={perception_output.source_id}")
        
        # Extract threat traits from perception output
        weapon = perception_output.traits.weapon_detected
        emotion = perception_output.traits.emotion
        tone = perception_output.traits.tone
        confidence_scores = perception_output.confidence_scores
        risk_hints = self._extract_risk_hints(perception_output)
        
        logger.info(f"[ADAPTER] Extracted traits: weapon={weapon}, emotion={emotion}, tone={tone}")
        
        # Get reasoning decision from cloud service
        reasoning_output = await self.service.reason(
            source_id=perception_output.source_id,
            weapon_detected=weapon,
            emotion=emotion,
            tone=tone,
            confidence_scores=confidence_scores,
            risk_hints=risk_hints,
            use_learning_agent=True,  # Enabled for scenario prediction and temporal analysis
            use_sop_context=True,  # Enabled for location-aware threat scoring
        )
        
        self.last_reasoning = reasoning_output
        
        logger.info(f"[ADAPTER] Reasoning complete: threat_level={reasoning_output.threat_level}, action={reasoning_output.recommended_action.action}")
        
        return reasoning_output
    
    def process_perception(self, perception_output):
        """
        Synchronous wrapper for async reasoning (for use in sync contexts).
        
        Args:
            perception_output: UnifiedPerceptionOutput from perception layer
        
        Returns:
            ReasoningOutput
        """
        if not self.is_ready:
            raise RuntimeError("Reasoning adapter not initialized")
        
        logger.info(f"[ADAPTER] process_perception called for source={perception_output.source_id}")
        
        # Check if we're already in an async context
        try:
            asyncio.get_running_loop()
            # We're in an async context - should not be calling sync wrapper
            logger.error("[ADAPTER] process_perception called from async context, use process_perception_async instead!")
            raise RuntimeError("Cannot call sync process_perception from async context. Use process_perception_async instead.")
        except RuntimeError as e:
            if "Cannot call sync" in str(e):
                raise
            # No running loop, we're in sync context - safe to proceed
            logger.debug("[ADAPTER] No running event loop, safe to use sync wrapper")
            pass
        
        try:
            # Get or create event loop for this thread
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                logger.info("[ADAPTER] Event loop was closed, creating new one")
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            # Create new event loop for this thread
            logger.info("[ADAPTER] No event loop, creating new one")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        try:
            logger.info("[ADAPTER] Starting cloud reasoning with 45s timeout")
            # Add timeout protection (45 seconds for cloud reasoning)
            result = loop.run_until_complete(
                asyncio.wait_for(
                    self.process_perception_async(perception_output),
                    timeout=45.0  # 45 second timeout for cloud reasoning
                )
            )
            logger.info(f"[ADAPTER] ✓ process_perception succeeded: threat_level={result.threat_level}, action={result.recommended_action.action}", extra={"flush": True})
            return result
        except asyncio.TimeoutError:
            logger.error(f"[ADAPTER] ✗ TIMEOUT: Cloud reasoning exceeded 45 seconds", extra={"flush": True})
            import traceback
            logger.error(f"[ADAPTER] Traceback: {traceback.format_exc()}")
            raise
        except Exception as e:
            logger.error(f"[ADAPTER] ✗ Error in process_perception: {str(e)}", extra={"flush": True})
            import traceback
            logger.error(f"[ADAPTER] Traceback: {traceback.format_exc()}")
            raise
    
    def _extract_risk_hints(self, perception_output):
        """Extract risk hints from perception output"""
        hints = []
        traits = perception_output.traits
        
        # Weapon hints
        if traits.weapon_detected != "unarmed":
            hints.append("visible_weapon")
        
        if traits.weapon_suppressed_due_to_uniform:
            hints.append("weapon_suppressed_uniformed_personnel")
        
        # Emotion hints
        if traits.emotion in ["angry", "fearful", "distressed", "panicked"]:
            hints.append("emotional_escalation")
        
        # Audio hints
        if traits.tone in ["panic", "threat", "abnormal", "distressed"]:
            hints.append("audio_escalation")
        
        # Speech flags
        if traits.speech_present and traits.keyword_flags:
            hints.append("speech_flags_present")
        
        # Acoustic events
        if traits.acoustic_events:
            if "gunshot" in str(traits.acoustic_events).lower():
                hints.append("gunshot_detected")
            if "scream" in str(traits.acoustic_events).lower():
                hints.append("scream_detected")
        
        logger.debug(f"[ADAPTER] Extracted risk hints: {hints}")
        return hints
    
    def get_state(self):
        """Get current reasoning adapter state"""
        state = {
            "is_ready": self.is_ready,
            "call_count": self.call_count,
            "reasoning_available": REASONING_AVAILABLE,
            "last_reasoning": None,
        }
        
        if self.last_reasoning:
            state["last_reasoning"] = {
                "timestamp": self.last_reasoning.timestamp.isoformat(),
                "source_id": self.last_reasoning.source_id,
                "threat_level": self.last_reasoning.threat_level,
                "recommended_action": self.last_reasoning.recommended_action.action,
                "confidence": self.last_reasoning.confidence,
            }
        
        return state
    
    def reset(self):
        """Reset reasoning adapter state"""
        self.call_count = 0
        self.last_reasoning = None
        logger.info("[ADAPTER] Reasoning adapter state reset")
        return {"status": "Reasoning adapter state reset"}


# Global instance
_reasoning_adapter = None


def get_reasoning_adapter(cloud_provider=None):
    """Get or create the global reasoning adapter instance
    
    Args:
        cloud_provider: Cloud provider to use. If None, uses CLOUD_REASONING_PROVIDER 
                       env var or defaults to 'ollama_free_api'
    """
    global _reasoning_adapter
    
    if _reasoning_adapter is None:
        if not REASONING_AVAILABLE:
            raise RuntimeError("Cloud Reasoning Service not available")
        
        # Determine provider from parameter or environment
        if cloud_provider is None:
            import os
            cloud_provider = os.getenv("CLOUD_REASONING_PROVIDER", "ollama_free_api")
        
        logger.info(f"[ADAPTER] Creating ReasoningAdapter with provider: {cloud_provider}")
        _reasoning_adapter = ReasoningAdapter(cloud_provider=cloud_provider)
    
    return _reasoning_adapter


def is_reasoning_available():
    """Check if reasoning layer is available"""
    return REASONING_AVAILABLE

