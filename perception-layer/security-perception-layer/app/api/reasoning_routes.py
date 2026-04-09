"""
Reasoning Router - REST API endpoints for reasoning layer

Handles integrated perception + reasoning workflows with Cloud Reasoning Agent
"""

import logging
from fastapi import APIRouter, HTTPException

from app.schemas import UnifiedPerceptionOutput
from app.reasoning_adapter import get_reasoning_adapter, is_reasoning_available


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/reasoning", tags=["reasoning"])


@router.get("/health")
async def health():
    """Check if reasoning layer is available"""
    try:
        if not is_reasoning_available():
            raise HTTPException(status_code=503, detail="Reasoning layer not available")
        
        adapter = get_reasoning_adapter()
        state = adapter.get_state()
        
        return {
            "status": "ok",
            "reasoning": "available",
            "service": "cloud",
            "agent_state": state
        }
    except Exception as e:
        logger.error(f"[ROUTES] Health check error: {e}")
        raise HTTPException(status_code=503, detail=f"Reasoning layer unavailable: {str(e)}")


@router.post("/assess")
async def assess_threat(perception: UnifiedPerceptionOutput):
    """
    Assess threat from unified perception output using Cloud Reasoning Agent.
    
    Takes perception data and returns comprehensive threat reasoning:
    - Threat classification (low/medium/high/critical)
    - Recommended actions (immediate_alert/escalate/monitor/etc)
    - Detailed reasoning explanation
    - Confidence scores for each modality
    - Threat metrics visualization
    
    Input: UnifiedPerceptionOutput from perception layer
    Output: ReasoningOutput with complete threat assessment
    """
    try:
        logger.info(f"[ROUTES] Assessing threat for source={perception.source_id}")
        
        adapter = get_reasoning_adapter()
        
        # Process perception through cloud reasoning agent (async)
        reasoning_output = await adapter.process_perception_async(perception)
        
        logger.info(f"[ROUTES] Assessment complete: threat_level={reasoning_output.threat_level}")
        
        # Return reasoning output directly (will be json serialized)
        return reasoning_output
        
    except Exception as e:
        logger.error(f"[ROUTES] Error during reasoning: {e}")
        raise HTTPException(status_code=500, detail=f"Error during reasoning: {str(e)}")


@router.post("/assess-full")
async def assess_threat_full(perception: UnifiedPerceptionOutput):
    """
    Full assessment with extended output for dashboard.
    
    Same as /assess but includes additional fields:
    - Full explanation with evidence
    - All threat metrics
    - Anomaly types
    - Reasoning version
    """
    try:
        logger.info(f"[ROUTES] Full assessment for source={perception.source_id}")
        
        adapter = get_reasoning_adapter()
        reasoning_output = await adapter.process_perception_async(perception)
        
        # Return with perception data for context
        return {
            "perception": perception.dict(),
            "reasoning": reasoning_output.dict(),
        }
        
    except Exception as e:
        logger.error(f"[ROUTES] Error during full reasoning: {e}")
        raise HTTPException(status_code=500, detail=f"Error during reasoning: {str(e)}")


@router.get("/state")
async def get_agent_state():
    """
    Get current reasoning agent state for debugging and monitoring.
    
    Returns:
    - Agent readiness status
    - Number of assessments processed
    - Last reasoning output summary
    - Connectivity status
    """
    try:
        adapter = get_reasoning_adapter()
        return adapter.get_state()
    except Exception as e:
        logger.error(f"[ROUTES] Error retrieving state: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving state: {str(e)}")


@router.post("/reset")
async def reset_agent():
    """
    Clear all reasoning agent state and restart.
    
    Useful for clearing history between test sessions.
    """
    try:
        logger.info("[ROUTES] Resetting reasoning agent")
        adapter = get_reasoning_adapter()
        return adapter.reset()
    except Exception as e:
        logger.error(f"[ROUTES] Error resetting agent: {e}")
        raise HTTPException(status_code=500, detail=f"Error resetting agent: {str(e)}")


@router.get("/info")
async def describe_reasoning():
    """
    Get information about the Cloud Reasoning Agent.
    
    Returns:
    - Reasoning capabilities
    - Threat levels supported
    - Recommended actions available
    - SOP awareness
    - Cloud/local routing info
    """
    try:
        return {
            "service_name": "Cloud Reasoning Agent",
            "reasoning_version": "cloud_v1.0",
            "threat_levels": ["low", "medium", "high", "critical"],
            "threat_thresholds": {
                "critical": 0.75,
                "high": 0.50,
                "medium": 0.25,
                "low": 0.0
            },
            "recommended_actions": [
                "immediate_alert",
                "escalate", 
                "elevated_monitoring",
                "monitor",
                "de_escalate",
                "all_clear"
            ],
            "threat_scoring": {
                "weapon_weight": 0.50,
                "emotion_weight": 0.25,
                "audio_weight": 0.20,
                "behavioral_weight": 0.05,
                "formula": "weapon*0.5 + emotion*0.25 + audio*0.20 + behavioral*0.05"
            },
            "capabilities": {
                "multi_modal_fusion": True,
                "threat_scoring": True,
                "cloud_local_routing": True,
                "sop_context_aware": True,
                "scenario_integration_ready": True,
                "fallback_to_local": True,
                "audit_logging": True,
                "detailed_explanations": True
            },
            "sop_context": {
                "location_types": ["school", "airport", "bank", "street", "office"],
                "security_levels": ["critical", "high", "medium", "low"],
                "multipliers": {
                    "school": 1.4,
                    "airport": 1.3,
                    "bank": 1.2,
                    "street": 1.0,
                    "office": 0.9
                }
            },
            "documentation": {
                "cloud_reasoning_readme": "/docs/CLOUD_REASONING_README.md",
                "integration_guide": "/docs/integration.py",
                "examples": "/docs/IMPLEMENTATION_SUMMARY.md"
            }
        }
    except Exception as e:
        logger.error(f"[ROUTES] Error getting info: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting info: {str(e)}")

