from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import logging

from app.api.routes import router as perception_router
from app.api.reasoning_routes import router as reasoning_router

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Security Perception & Reasoning Layer",
    version="0.2.0",
    description="Multimodal perception service with integrated threat reasoning for CCTV and bodycam streams.",
)

# Add CORS middleware to allow dashboard to call API endpoints
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include perception layer endpoints
app.include_router(perception_router)

# Include reasoning layer endpoints
app.include_router(reasoning_router)

# Serve static files (dashboard)
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Dashboard route
@app.get("/dashboard")
async def dashboard():
    """Serve the security dashboard"""
    dashboard_file = Path(__file__).parent / "static" / "dashboard.html"
    if dashboard_file.exists():
        return FileResponse(dashboard_file, media_type="text/html")
    return {"error": "Dashboard not found"}

@app.get("/")
async def root():
    """Redirect to dashboard"""
    return FileResponse(Path(__file__).parent / "static" / "dashboard.html", media_type="text/html")

@app.post("/api/videos/upload")
async def upload_video(file: UploadFile = File(...)):
    """Upload a video file to the videos directory for processing"""
    try:
        # Define videos directory (relative to security-perception-layer)
        videos_dir = Path(__file__).parent.parent / "videos"
        videos_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate file extension
        allowed_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.m4v'}
        file_ext = Path(file.filename).suffix.lower()
        
        if file_ext not in allowed_extensions:
            return {
                "error": f"File type not allowed. Allowed: {', '.join(allowed_extensions)}",
                "status": "error"
            }
        
        # Save file
        file_path = videos_dir / file.filename
        logger.info(f"[UPLOAD] Saving video to: {file_path}")
        
        # Write file in chunks to handle large files
        with open(file_path, 'wb') as f:
            content = await file.read()
            f.write(content)
        
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        logger.info(f"[UPLOAD] File saved: {file.filename} ({file_size_mb:.1f} MB)")
        
        return {
            "status": "success",
            "filename": file.filename,
            "size_mb": file_size_mb,
            "path": str(file_path),
            "message": f"Video '{file.filename}' uploaded successfully"
        }
        
    except Exception as e:
        logger.error(f"[UPLOAD] Error uploading file: {e}")
        return {
            "status": "error",
            "error": str(e),
            "message": f"Failed to upload video: {str(e)}"
        }
