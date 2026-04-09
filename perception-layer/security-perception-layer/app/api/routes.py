from fastapi import APIRouter, HTTPException
import logging

from app.schemas import HealthResponse, PerceptionRequest, PerceptionResponse, UnifiedPerceptionOutput
from app.services.pipeline import perception_pipeline
from app.reasoning_adapter import get_reasoning_adapter, is_reasoning_available


logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get("/perception/models")
async def list_models() -> dict:
    return perception_pipeline.describe_models()


@router.post("/perception/infer", response_model=PerceptionResponse)
async def infer(request: PerceptionRequest) -> PerceptionResponse:
    return await perception_pipeline.run(request)


@router.post("/perception/unified", response_model=UnifiedPerceptionOutput)
async def infer_unified(request: PerceptionRequest) -> UnifiedPerceptionOutput:
    return await perception_pipeline.run_unified(request)


@router.post("/perception/unified-with-reasoning")
async def infer_unified_with_reasoning(request: PerceptionRequest):
    """
    Integrated endpoint: Perception + Cloud Reasoning in one call.
    
    This is the main production endpoint that:
    1. Runs multi-modal perception (weapon, emotion, audio, fusion)
    2. Passes unified output to Cloud Reasoning Agent
    3. Returns both perception and reasoning results
    
    Perfect for dashboard and end-to-end video processing.
    
    Returns:
    {
        "perception": UnifiedPerceptionOutput,
        "reasoning": ReasoningOutput,
        "integrated_timestamp": ISO timestamp
    }
    """
    try:
        logger.info(f"[ROUTES] Integrated request: source={request.source_id if hasattr(request, 'source_id') else 'unknown'}")
        
        # Step 1: Run perception layer
        perception_output = await perception_pipeline.run_unified(request)
        logger.info(f"[ROUTES] Perception complete: weapon={perception_output.traits.weapon_detected}, emotion={perception_output.traits.emotion}")
        
        # Step 2: Run reasoning layer
        if not is_reasoning_available():
            logger.warn("[ROUTES] Reasoning layer not available, returning perception only")
            return {
                "perception": perception_output,
                "reasoning": None,
                "warning": "Reasoning layer unavailable, returning perception only"
            }
        
        adapter = get_reasoning_adapter()
        reasoning_output = await adapter.process_perception_async(perception_output)
        logger.info(f"[ROUTES] Reasoning complete: threat_level={reasoning_output.threat_level}")
        
        # Step 3: Return integrated response
        return {
            "perception": perception_output.dict(),
            "reasoning": reasoning_output.dict(),
            "integrated_timestamp": reasoning_output.timestamp.isoformat(),
        }
        
    except Exception as e:
        logger.error(f"[ROUTES] Error in integrated endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Error in perception+reasoning: {str(e)}")
