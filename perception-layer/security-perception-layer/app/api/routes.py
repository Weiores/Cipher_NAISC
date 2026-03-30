from fastapi import APIRouter

from app.schemas import HealthResponse, PerceptionRequest, PerceptionResponse, UnifiedPerceptionOutput
from app.services.pipeline import perception_pipeline


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
