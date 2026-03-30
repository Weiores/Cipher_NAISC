from fastapi import FastAPI

from app.api.routes import router


app = FastAPI(
    title="Security Perception Layer",
    version="0.1.0",
    description="Multimodal perception service for CCTV and bodycam streams.",
)
app.include_router(router)
