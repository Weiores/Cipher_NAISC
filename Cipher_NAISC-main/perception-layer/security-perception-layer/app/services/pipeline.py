from app.schemas import PerceptionRequest, PerceptionResponse, UnifiedPerceptionOutput
from app.services.fusion import build_fused_event, build_unified_output
from app.services.models.audio import AudioThreatDetectionAdapter
from app.services.models.emotion import EmotionDetectionAdapter
from app.services.models.uniform import UniformDetectionAdapter
from app.services.models.weapon import WeaponDetectionAdapter


class PerceptionPipeline:
    def __init__(self) -> None:
        self.weapon_model = WeaponDetectionAdapter()
        self.uniform_model = UniformDetectionAdapter()
        self.emotion_model = EmotionDetectionAdapter()
        self.audio_model = AudioThreatDetectionAdapter()

    def describe_models(self) -> dict:
        return {
            "models": [
                self.weapon_model.describe(),
                self.uniform_model.describe(),
                self.emotion_model.describe(),
                self.audio_model.describe(),
            ]
        }

    async def run(self, request: PerceptionRequest) -> PerceptionResponse:
        weapon = await self.weapon_model.infer(request)
        uniform = await self.uniform_model.infer(request)
        emotion = await self.emotion_model.infer(request)
        audio = await self.audio_model.infer(request)
        fused = build_fused_event(request, weapon, uniform, emotion, audio)
        unified = build_unified_output(request, weapon, uniform, emotion, audio, fused)

        return PerceptionResponse(
            weapon_model_output=weapon,
            uniform_model_output=uniform,
            emotion_model_output=emotion,
            audio_model_output=audio,
            fused_event=fused,
            unified_output=unified,
        )

    async def run_unified(self, request: PerceptionRequest) -> UnifiedPerceptionOutput:
        response = await self.run(request)
        return response.unified_output


perception_pipeline = PerceptionPipeline()
