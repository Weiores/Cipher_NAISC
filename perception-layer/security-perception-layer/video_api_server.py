"""
Lightweight HTTP server for video processing API
Avoids heavy dependencies by using built-in Python HTTP server
"""

import json
import sys
from pathlib import Path
from urllib.parse import urlparse, parse_qs
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
from datetime import datetime

print("[SERVER_INIT] video_api_server.py starting up...", flush=True)
print(f"[SERVER_INIT] Python file: {__file__}", flush=True)

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "reasoning-layer"))

from video_processor import processor, process_video_async

print("[SERVER_INIT] Imports completed", flush=True)

class VideoAPIHandler(BaseHTTPRequestHandler):
    """HTTP request handler for video processing API"""
    
    VIDEOS_DIR = Path(__file__).parent / "videos"
    
    def log_message(self, format, *args):
        """Override to add [HTTP] prefix to all requests"""
        print(f"[HTTP_BASE] {self.address_string()} - {format%args}", flush=True)
    
    def do_GET(self):
        """Handle GET requests"""
        path = self.path
        msg = f"[HTTP_GET] Received GET request to: {path}"
        print(msg, flush=True)
        print(msg, file=sys.stderr, flush=True)
        # Also log to file
        with open("http_requests.log", "a") as f:
            f.write(msg + "\n")
            f.flush()
        
        # List videos
        if path == "/api/videos" or path == "/api/videos/":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            
            videos = processor.list_videos(self.VIDEOS_DIR)
            response = {
                "status": "success",
                "videos": videos,
                "count": len(videos),
                "storage_path": str(self.VIDEOS_DIR)
            }
            self.wfile.write(json.dumps(response).encode())
        
        # Get processing results for a video
        elif path.startswith("/api/results/"):
            video_id = path.split("/")[-1]
            results = processor.get_results(video_id)
            
            if results:
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps(results).encode())
            else:
                self.send_response(404)
                self.send_header("Content-type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Results not found", "video_id": video_id}).encode())
        
        # Export frame data for learning agent (without frame base64)
        elif path.startswith("/api/export/"):
            video_id = path.split("/")[-1]
            results = processor.get_results(video_id)
            
            if results:
                # Export without base64 frame data for faster transfer
                export_data = {
                    'video_id': results['video_id'],
                    'total_frames': results['total_frames'],
                    'processed_frames': results['processed_frames'],
                    'fps': results['fps'],
                    'duration_sec': results['duration_sec'],
                    'summary': results.get('summary', {}),
                    'frames': [
                        {
                            'frame_number': d['frame_number'],
                            'timestamp_sec': d['timestamp_sec'],
                            'perception': d['perception'],
                            'reasoning': d['reasoning']
                        }
                        for d in results.get('detections', [])
                    ]
                }
                
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Content-Disposition", f"attachment; filename=analysis_{video_id}.json")
                self.end_headers()
                self.wfile.write(json.dumps(export_data).encode())
            else:
                self.send_response(404)
                self.send_header("Content-type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Results not found", "video_id": video_id}).encode())
        
        # Health check
        elif path == "/api/health":
            print(f"[HTTP_GET] Health check requested", flush=True)
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok", "service": "video-processor"}).encode())
        
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_OPTIONS(self):
        """Handle CORS preflight requests"""
        msg = f"[HTTP_OPTIONS] Received OPTIONS request to: {self.path}"
        print(msg, flush=True)
        print(msg, file=sys.stderr, flush=True)
        with open("http_requests.log", "a") as f:
            f.write(msg + "\n")
            f.flush()
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
    
    def do_POST(self):
        """Handle POST requests"""
        path = self.path
        msg = f"\n[HTTP_POST] Received POST request to: {path}"
        print(msg, flush=True)
        print(msg, file=sys.stderr, flush=True)
        with open("http_requests.log", "a") as f:
            f.write(msg + "\n")
            f.flush()
        
        # Log to file to debug
        with open("post_debug.log", "w") as f:
            f.write(f"do_POST called with path={path}\n")
            f.write(f"Checking if path == '/api/process': {path == '/api/process'}\n")
            f.flush()
        
        # Process video
        if path == "/api/process" or path == "/api/process/":
            with open("post_debug.log", "a") as f:
                f.write(f"Matched /api/process path\n")
                f.flush()
                
            print(f"[HTTP_POST] Processing /api/process request", flush=True)
            try:
                with open("post_debug.log", "a") as f:
                    f.write(f"Reading content_length\n")
                    f.flush()
                    
                content_length = int(self.headers.get("Content-Length", 0))
                print(f"[HTTP_POST] Content-Length: {content_length}", flush=True)
                with open("post_debug.log", "a") as f:
                    f.write(f"Content-Length: {content_length}\n")
                    f.flush()
                    
                body = self.rfile.read(content_length)
                data = json.loads(body.decode())
                print(f"[HTTP_POST] Parsed JSON: {data}", flush=True)
                
                video_name = data.get("video_name")
                video_id = data.get("video_id", video_name)
                fps_sample = data.get("fps_sample", 2.0)
                
                print(f"[HTTP_POST] video_name={video_name}, video_id={video_id}, fps_sample={fps_sample}", flush=True)
                
                if not video_name:
                    print(f"[HTTP_POST] ERROR: Missing video_name", flush=True)
                    self.send_error(400, "Missing video_name")
                    return
                
                video_path = self.VIDEOS_DIR / video_name
                print(f"[HTTP_POST] Checking video_path: {video_path}", flush=True)
                
                if not video_path.exists():
                    print(f"[HTTP_POST] ERROR: Video not found at {video_path}", flush=True)
                    self.send_error(404, f"Video not found: {video_name}")
                    return
                
                # Start processing in background
                print(f"[HTTP_POST] Calling process_video_async with video_path={video_path}, video_id={video_id}", flush=True)
                process_video_async(str(video_path), video_id)
                print(f"[HTTP_POST] process_video_async called successfully", flush=True)
                
                self.send_response(202)  # Accepted
                self.send_header("Content-type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                
                response = {
                    "status": "processing",
                    "video_id": video_id,
                    "video_name": video_name,
                    "message": "Video processing started. Check /api/results/{video_id} for results."
                }
                self.wfile.write(json.dumps(response).encode())
                print(f"[HTTP_POST] Response sent successfully", flush=True)
                
            except Exception as e:
                print(f"[HTTP_POST] ERROR: {str(e)}", flush=True)
                import traceback
                traceback.print_exc()
                self.send_error(400, str(e))
        
        else:
            print(f"[HTTP_POST] Unknown path: {path}", flush=True)
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        """Custom logging"""
        print(f"[API] {self.client_address[0]} - {format % args}")


def run_video_server(port=8001):
    """Run the video processing API server"""
    
    # Ensure videos directory exists
    VideoAPIHandler.VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    
    server_address = ("", port)
    httpd = HTTPServer(server_address, VideoAPIHandler)
    
    print(f"\n{'='*60}")
    print(f"✓ Video Processing API running at http://localhost:{port}")
    print(f"✓ Videos directory: {VideoAPIHandler.VIDEOS_DIR}")
    print(f"✓ Endpoints:")
    print(f"  GET  /api/videos       - List available videos")
    print(f"  POST /api/process      - Start processing a video")
    print(f"  GET  /api/results/{id} - Get processing results (with frames)")
    print(f"  GET  /api/export/{id}  - Export data for Learning Agent (no frames)")
    print(f"  GET  /api/health       - Health check")
    print(f"{'='*60}\n")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n✓ Video API server stopped")


if __name__ == "__main__":
    run_video_server()
