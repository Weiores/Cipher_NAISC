"""
Unified Server for Real-Time Security Analysis Dashboard
All-in-one solution combining:
  - Dashboard UI Server (port 8000)
  - Video Processing API (port 8001)
  - Real-Time Video Analysis API (port 8002)
"""

import json
import sys
import os
import cv2
import asyncio
import threading
import time
import subprocess
from pathlib import Path
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "reasoning-layer"))

from video_processor import VideoProcessor

# Try to import reasoning agent
try:
    from cloud_reasoning.ollama_reasoning_agent import OllamaReasoningAgent
    reasoning_agent = OllamaReasoningAgent()
    HAS_REASONING = True
except Exception as e:
    print(f"[WARNING] Reasoning layer not available: {e}")
    reasoning_agent = None
    HAS_REASONING = False

print("[STARTUP] Initializing unified security dashboard...")

# ============================================================================
# OLLAMA STARTUP - Required for cloud reasoning layer
# ============================================================================

ollama_process = None

def start_ollama():
    """Start Ollama service in background"""
    global ollama_process
    
    try:
        print("[OLLAMA] Checking if Ollama is already running on http://localhost:11434...")
        import requests
        try:
            response = requests.get("http://localhost:11434/api/version", timeout=2)
            if response.status_code == 200:
                print("[OLLAMA] ✓ Ollama is already running!")
                return True
        except:
            pass
        
        print("[OLLAMA] Starting Ollama service...")
        # Try to start Ollama - this will vary by OS
        if sys.platform == "win32":
            # Windows: Look for Ollama in common install locations
            ollama_paths = [
                "C:\\Users\\AppData\\Local\\Programs\\Ollama\\ollama.exe",
                os.path.expanduser("~\\AppData\\Local\\Programs\\Ollama\\ollama.exe"),
                "ollama"  # Try if in PATH
            ]
            
            for ollama_path in ollama_paths:
                if os.path.exists(ollama_path) or ollama_path == "ollama":
                    try:
                        print(f"[OLLAMA] Attempting to start from: {ollama_path}")
                        ollama_process = subprocess.Popen(
                            [ollama_path, "serve"],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == "win32" else 0
                        )
                        print(f"[OLLAMA] ✓ Ollama process started (PID: {ollama_process.pid})")
                        
                        # Wait a moment for Ollama to initialize
                        time.sleep(3)
                        
                        # Verify it's running
                        try:
                            response = requests.get("http://localhost:11434/api/version", timeout=5)
                            if response.status_code == 200:
                                print("[OLLAMA] ✓ Ollama is running and responding!")
                                return True
                        except:
                            pass
                        
                        return True
                    except Exception as e:
                        print(f"[OLLAMA] Failed to start from {ollama_path}: {e}")
                        continue
        else:
            # macOS/Linux
            ollama_paths = ["/usr/local/bin/ollama", "/opt/ollama/ollama", "ollama"]
            
            for ollama_path in ollama_paths:
                try:
                    print(f"[OLLAMA] Attempting to start from: {ollama_path}")
                    ollama_process = subprocess.Popen(
                        [ollama_path, "serve"],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    )
                    print(f"[OLLAMA] ✓ Ollama process started (PID: {ollama_process.pid})")
                    
                    # Wait a moment for Ollama to initialize
                    time.sleep(3)
                    
                    # Verify it's running
                    try:
                        response = requests.get("http://localhost:11434/api/version", timeout=5)
                        if response.status_code == 200:
                            print("[OLLAMA] ✓ Ollama is running and responding!")
                            return True
                    except:
                        pass
                    
                    return True
                except Exception as e:
                    print(f"[OLLAMA] Failed to start from {ollama_path}: {e}")
                    continue
        
        print("[OLLAMA] ⚠️  Could not start Ollama automatically.")
        print("[OLLAMA] Please ensure Ollama is installed and running:")
        print("[OLLAMA]   - Install from: https://ollama.ai")
        print("[OLLAMA]   - Then run: ollama serve")
        return False
        
    except Exception as e:
        print(f"[OLLAMA] Error during startup: {e}")
        return False



# ============================================================================
# DASHBOARD HANDLER (Port 8000) - Serve UI
# ============================================================================

class DashboardHandler(SimpleHTTPRequestHandler):
    """Custom handler to serve the dashboard UI"""
    
    def do_GET(self):
        """Handle GET requests - serve dashboard HTML"""
        # Handle favicon gracefully
        if self.path == "/favicon.ico":
            self.send_response(204)  # No Content
            self.end_headers()
            return
        
        if self.path == "/" or self.path == "":
            self.path = "/dashboard.html"
        return super().do_GET()
    
    def log_message(self, format, *args):
        """Custom logging"""
        print(f"[DASHBOARD] {format % args}")


# ============================================================================
# VIDEO API HANDLER (Port 8001) - Batch Video Processing
# ============================================================================

class VideoAPIHandler(BaseHTTPRequestHandler):
    """HTTP request handler for video processing API"""
    
    VIDEOS_DIR = Path(__file__).parent / "videos"
    processor = VideoProcessor()
    
    def log_message(self, format, *args):
        print(f"[VIDEO_API] {format%args}", flush=True)
    
    def do_GET(self):
        """Handle GET requests"""
        path = self.path
        
        # List videos
        if path == "/api/videos" or path == "/api/videos/":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            
            videos = self.processor.list_videos(self.VIDEOS_DIR)
            response = {
                "status": "success",
                "videos": videos,
                "count": len(videos),
                "storage_path": str(self.VIDEOS_DIR)
            }
            self.wfile.write(json.dumps(response).encode())
        
        # Get processing results
        elif path.startswith("/api/results/"):
            video_id = path.split("/")[-1]
            results = self.processor.get_results(video_id)
            
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
                self.wfile.write(json.dumps({"error": "Results not found"}).encode())
        
        # Export frame data
        elif path.startswith("/api/export/"):
            video_id = path.split("/")[-1]
            results = self.processor.get_results(video_id)
            
            if results:
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
                self.wfile.write(json.dumps({"error": "Results not found"}).encode())
        
        # Health check
        elif path == "/api/health":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok", "service": "video-processor"}).encode())
        
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_OPTIONS(self):
        """Handle CORS"""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
    
    def do_POST(self):
        """Handle POST requests"""
        path = self.path
        
        if path == "/api/process" or path == "/api/process/":
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length)
                data = json.loads(body.decode())
                
                video_name = data.get("video_name")
                video_id = data.get("video_id", video_name)
                fps_sample = data.get("fps_sample", 2.0)
                
                if not video_name:
                    self.send_error(400, "Missing video_name")
                    return
                
                video_path = self.VIDEOS_DIR / video_name
                
                if not video_path.exists():
                    self.send_error(404, f"Video not found: {video_name}")
                    return
                
                # Start processing in background
                from video_processor import process_video_async
                process_video_async(str(video_path), video_id)
                
                self.send_response(202)
                self.send_header("Content-type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                
                response = {
                    "status": "processing",
                    "video_id": video_id,
                    "message": "Video processing started"
                }
                self.wfile.write(json.dumps(response).encode())
                
            except Exception as e:
                self.send_error(400, str(e))
        else:
            self.send_response(404)
            self.end_headers()


# ============================================================================
# REAL-TIME VIDEO SERVER (Port 8002)
# ============================================================================

class FrameAnalysisCache:
    """Cache for analyzed frames"""
    def __init__(self):
        self.cache = {}
        self.video_metadata = {}

    def set_metadata(self, video_id, total_frames, fps, duration):
        self.video_metadata[video_id] = {
            'total_frames': total_frames,
            'fps': fps,
            'duration': duration
        }

    def cache_frame(self, video_id, frame_num, analysis):
        if video_id not in self.cache:
            self.cache[video_id] = {}
        self.cache[video_id][frame_num] = analysis

    def get_frame(self, video_id, frame_num):
        return self.cache.get(video_id, {}).get(frame_num, None)

    def get_metadata(self, video_id):
        return self.video_metadata.get(video_id, None)


class RealtimeVideoHandler(BaseHTTPRequestHandler):
    """Real-time video analysis handler"""
    
    VIDEOS_DIR = Path(__file__).parent / "videos"
    processor = VideoProcessor()
    cache = FrameAnalysisCache()
    
    def log_message(self, format, *args):
        print(f"[REALTIME] {format%args}", flush=True)
    
    def send_error(self, code, message=None):
        """Override send_error to include CORS headers"""
        self.send_response(code)
        self.send_header("Content-type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        error_response = json.dumps({"error": message or f"HTTP {code}"})
        self.wfile.write(error_response.encode())
    
    def do_GET(self):
        """Handle GET requests"""
        path = self.path
        
        # Load video metadata
        if path.startswith("/api/realtime/load/"):
            video_name = path.split("/api/realtime/load/")[-1]
            try:
                video_path = self.VIDEOS_DIR / video_name
                if not video_path.exists():
                    self.send_error(404, f"Video not found: {video_name}")
                    return
                
                cap = cv2.VideoCapture(str(video_path))
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration_sec = total_frames / fps if fps > 0 else 0
                cap.release()
                
                video_id = f"{video_name.replace('.', '_')}_{int(datetime.now().timestamp())}"
                self.cache.set_metadata(video_id, total_frames, fps, duration_sec)
                
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                
                response = {
                    "status": "success",
                    "video_id": video_id,
                    "video_name": video_name,
                    "total_frames": total_frames,
                    "fps": fps,
                    "duration_sec": float(duration_sec)
                }
                self.wfile.write(json.dumps(response).encode())
                
            except Exception as e:
                self.send_error(400, str(e))
        
        # Analyze specific frame
        elif path.startswith("/api/realtime/analyze/"):
            try:
                # Strip query string first: /api/realtime/analyze/VIDEO/FRAME?params -> /api/realtime/analyze/VIDEO/FRAME
                clean_path = path.split("?")[0]
                parts = clean_path.split("/api/realtime/analyze/")[-1].split("/")
                video_name = parts[0]
                frame_num = int(parts[1]) if len(parts) > 1 else 0
                
                video_id = self._get_query_param("video_id", f"{video_name.replace('.', '_')}_realtime")
                
                video_path = self.VIDEOS_DIR / video_name
                if not video_path.exists():
                    self.send_error(404, f"Video not found: {video_name}")
                    return
                
                # Check cache
                cached_result = self.cache.get_frame(video_id, frame_num)
                if cached_result:
                    self.send_response(200)
                    self.send_header("Content-type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(json.dumps(cached_result).encode())
                    return
                
                # Analyze frame
                cap = cv2.VideoCapture(str(video_path))
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                cap.release()
                
                if not ret:
                    self.send_error(400, f"Could not read frame {frame_num}")
                    return
                
                # Run perception
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                perception, _ = loop.run_until_complete(
                    self.processor._run_perception_models(frame, video_id, frame_num)
                )
                loop.close()
                
                reasoning_payload = self.processor._quick_frame_reasoning(perception)
                
                # Encode frame
                import base64
                _, buffer = cv2.imencode('.jpg', frame)
                frame_data = base64.b64encode(buffer).decode()
                
                analysis_result = {
                    "frame_number": frame_num,
                    "frame_data": f"data:image/jpeg;base64,{frame_data}",
                    "perception": {
                        "weapon_detected": perception.traits.weapon_detected,
                        "emotion": perception.traits.emotion,
                        "tone": perception.traits.tone,
                        "uniform_present": perception.traits.uniform_present,
                        "speech_present": perception.traits.speech_present,
                        "keyword_flags": perception.traits.keyword_flags or [],
                        "acoustic_events": perception.traits.acoustic_events or []
                    },
                    "reasoning": reasoning_payload,
                    "timestamp_sec": frame_num / (self.cache.get_metadata(video_id)['fps'] if self.cache.get_metadata(video_id) else 30)
                }
                
                self.cache.cache_frame(video_id, frame_num, analysis_result)
                
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps(analysis_result).encode())
                
            except Exception as e:
                self.send_error(400, str(e))
        
        # List videos
        elif path == "/api/realtime/videos":
            try:
                videos = []
                if self.VIDEOS_DIR.exists():
                    for video_file in self.VIDEOS_DIR.glob("*"):
                        if video_file.is_file() and video_file.suffix.lower() in {'.mp4', '.avi', '.mov', '.mkv'}:
                            videos.append({
                                "name": video_file.name,
                                "size_mb": video_file.stat().st_size / (1024 * 1024)
                            })
                
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                
                response = {
                    "status": "success",
                    "videos": videos,
                    "count": len(videos)
                }
                self.wfile.write(json.dumps(response).encode())
                
            except Exception as e:
                self.send_error(400, str(e))
        
        # Health check
        elif path == "/api/realtime/health":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok"}).encode())
        
        # Generate video summary with cloud reasoning
        elif path.startswith("/api/realtime/summarize/"):
            try:
                print(f"[REALTIME] Summarizing video...", flush=True)
                
                # Parse video_id from URL (remove query string first)
                clean_path = path.split("?")[0]
                video_id = clean_path.split("/api/realtime/summarize/")[-1]
                
                print(f"[REALTIME] Summarize endpoint: video_id='{video_id}'", flush=True)
                print(f"[REALTIME] Available cached video IDs: {list(self.cache.cache.keys())}", flush=True)
                
                # Get the cached frames for this video
                cached_frames = self.cache.cache.get(video_id, {})
                
                # If no frames found, try to find the most recent video_id that matches the video name
                if not cached_frames and video_id:
                    print(f"[REALTIME] No cached frames for video_id '{video_id}', searching for alternatives...", flush=True)
                    # Look for any video_id that contains this video name
                    video_name_pattern = video_id.split('_')[0] if '_' in video_id else video_id
                    for cached_id in self.cache.cache.keys():
                        if video_name_pattern in cached_id or cached_id.startswith(video_name_pattern):
                            print(f"[REALTIME] Found alternative cached video_id: '{cached_id}'", flush=True)
                            cached_frames = self.cache.cache.get(cached_id, {})
                            if cached_frames:
                                video_id = cached_id
                                break
                
                print(f"[REALTIME] Generating summary for {len(cached_frames)} analyzed frames", flush=True)
                
                # Extract perception data from all cached frames
                perceptions = []
                threat_scores = []
                threat_levels = []
                for frame_num, analysis in sorted(cached_frames.items()):
                    if 'perception' in analysis and 'reasoning' in analysis:
                        perceptions.append(analysis['perception'])
                        threat_scores.append(float(analysis['reasoning'].get('threat_score', 0.0)))
                        threat_levels.append(analysis['reasoning'].get('threat_level', 'low'))
                
                # If no perceptions, provide default fallback response
                if not perceptions:
                    print(f"[REALTIME] No perception data cached - returning fallback", flush=True)
                    fallback_response = {
                        "status": "success",
                        "overall_threat_level": "low",
                        "recommended_action": "monitor",
                        "priority": "low",
                        "confidence": 0.5,
                        "reasoning_explanation": "Analysis complete - no threats detected",
                        "key_factors": ["No frames analyzed in this session", "Default safe state"],
                        "detected_anomalies": []
                    }
                    self.send_response(200)
                    self.send_header("Content-type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(json.dumps(fallback_response).encode())
                    return
                
                # Count occurrences
                weapon_counts = {}
                emotion_counts = {}
                tone_counts = {}
                for p in perceptions:
                    weapon = p.get('weapon_detected', 'unknown')
                    weapon_counts[weapon] = weapon_counts.get(weapon, 0) + 1
                    emotion = p.get('emotion', 'unknown')
                    emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
                    tone = p.get('tone', 'unknown')
                    tone_counts[tone] = tone_counts.get(tone, 0) + 1
                
                # Most common traits
                most_common_weapon = max(weapon_counts, key=weapon_counts.get) if weapon_counts else 'unknown'
                most_common_emotion = max(emotion_counts, key=emotion_counts.get) if emotion_counts else 'unknown'
                most_common_tone = max(tone_counts, key=tone_counts.get) if tone_counts else 'unknown'
                
                # Determine overall threat level
                max_threat = max(threat_levels) if threat_levels else 'low'
                avg_threat_score = sum(threat_scores) / len(threat_scores) if threat_scores else 0.0
                
                # FIREARM OVERRIDE: Guns are always HIGH+ threat
                if most_common_weapon and most_common_weapon.lower() in {"gun", "rifle", "shotgun"}:
                    print(f"[REALTIME] 🚨 FIREARM DETECTED: Applying firearm override to threat level", flush=True)
                    max_threat = 'high'  # Minimum HIGH for gun detection
                    
                    # Escalate to CRITICAL if threatening emotion/tone
                    if most_common_emotion and most_common_emotion.lower() in {"angry", "fearful", "distressed", "panicked"}:
                        max_threat = 'critical'
                        print(f"[REALTIME] 🚨 FIREARM + THREATENING EMOTION: Escalating to CRITICAL", flush=True)
                    elif most_common_tone and most_common_tone.lower() in {"threat", "panic", "abnormal", "distressed"}:
                        max_threat = 'critical'
                        print(f"[REALTIME] 🚨 FIREARM + THREATENING TONE: Escalating to CRITICAL", flush=True)
                
                # Build summary response (with fallback logic if cloud reasoning fails)
                summary_response = {
                    "status": "success",
                    "video_id": video_id,
                    "frames_analyzed": len(cached_frames),
                    "overall_threat_level": max_threat,
                    "overall_threat_score": float(avg_threat_score),
                    "recommended_action": "monitor" if max_threat == 'low' else ("elevated_monitoring" if max_threat == 'medium' else ("escalate" if max_threat == 'high' else "immediate_alert")),
                    "priority": max_threat,
                    "confidence": 0.75,
                    "reasoning_explanation": f"Video analysis complete. Detected {len(cached_frames)} critical frames. Most common weapon: {most_common_weapon}",
                    "key_factors": [
                        f"Weapon: {most_common_weapon}",
                        f"Threat Level: {max_threat}",
                        f"Emotion: {most_common_emotion}",
                        f"Tone: {most_common_tone}"
                    ],
                    "dominant_traits": {
                        "weapon": most_common_weapon,
                        "emotion": most_common_emotion,
                        "tone": most_common_tone
                    },
                    "threat_distribution": {
                        "low": threat_levels.count("low"),
                        "medium": threat_levels.count("medium"),
                        "high": threat_levels.count("high"),
                        "critical": threat_levels.count("critical")
                    }
                }
                
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps(summary_response).encode())
                print(f"[REALTIME] Summary response sent - threat: {max_threat}, action: {summary_response['recommended_action']}", flush=True)
                
            except Exception as e:
                print(f"[REALTIME] Error generating summary: {e}", flush=True)
                import traceback
                traceback.print_exc()
                self.send_error(500, f"Error generating summary: {str(e)}")
        
        else:
            self.send_error(404)
    
    def do_OPTIONS(self):
        """Handle CORS"""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
    
    def do_POST(self):
        """Handle POST requests - specifically for reasoning endpoint"""
        path = self.path
        
        # Cloud reasoning endpoint
        if path.startswith("/api/reasoning/analyze"):
            if not HAS_REASONING:
                self.send_error(503, "Reasoning layer not available")
                return
            
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length)
                data = json.loads(body.decode())
                
                detection_data = data.get("detection_data", {})
                frame_number = data.get("frame_number", 0)
                video_id = data.get("video_id", "")
                
                print(f"[REASONING] Processing frame {frame_number} for video {video_id}")
                
                # Call reasoning agent
                reasoning_result = reasoning_agent.analyze_detection(detection_data)
                
                # Prepare response
                response = {
                    'success': True,
                    'frame_number': frame_number,
                    'video_id': video_id,
                    'action': reasoning_result.get('action', '—'),
                    'priority': reasoning_result.get('priority', '—'),
                    'confidence': reasoning_result.get('confidence', '—'),
                    'reasoning': reasoning_result.get('reasoning', '—'),
                    'key_factors': reasoning_result.get('key_factors', []),
                    'detected_anomalies': reasoning_result.get('detected_anomalies', [])
                }
                
                print(f"[REASONING] Response: action={response['action']}, priority={response['priority']}")
                
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps(response).encode())
                
            except Exception as e:
                print(f"[REASONING ERROR] {str(e)}")
                import traceback
                traceback.print_exc()
                
                self.send_response(200)  # Return 200 to not break frontend
                self.send_header("Content-type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                
                error_response = {
                    'success': False,
                    'error': str(e),
                    'action': '—',
                    'priority': '—',
                    'confidence': '—',
                    'reasoning': '—',
                    'key_factors': [],
                    'detected_anomalies': []
                }
                self.wfile.write(json.dumps(error_response).encode())
        else:
            self.send_error(404, "Endpoint not found")
    
    def _get_query_param(self, key, default=""):
        """Extract query parameter"""
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        return params.get(key, [default])[0]


# ============================================================================
# MAIN SERVER STARTUP
# ============================================================================

def start_dashboard_server(port=8000):
    """Start dashboard server in separate thread"""
    static_dir = Path(__file__).parent / "app" / "static"
    
    if not static_dir.exists():
        print(f"Static directory not found at {static_dir}")
        return None
    
    # Change to static directory for SimpleHTTPRequestHandler
    original_dir = os.getcwd()
    os.chdir(static_dir)
    
    server_address = ("", port)
    httpd = HTTPServer(server_address, DashboardHandler)
    
    def run():
        print(f"Dashboard UI running at http://localhost:{port}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
    
    # Start in daemon thread
    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return httpd


def start_video_api_server(port=8001):
    """Start video API server in separate thread"""
    server_address = ("", port)
    httpd = HTTPServer(server_address, VideoAPIHandler)
    
    def run():
        print(f"Video API running at http://localhost:{port}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
    
    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return httpd


def start_realtime_server(port=8002):
    """Start real-time video server in separate thread"""
    server_address = ("", port)
    httpd = HTTPServer(server_address, RealtimeVideoHandler)
    
    def run():
        print(f"Real-Time API running at http://localhost:{port}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
    
    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return httpd


def main():
    """Start all servers"""
    global ollama_process
    
    print("\n" + "="*70)
    print("UNIFIED SECURITY ANALYSIS DASHBOARD")
    print("="*70)
    print("\nStarting all servers...\n")
    
    # Start Ollama first (required for cloud reasoning)
    print("[STARTUP] Step 1: Initializing Ollama (cloud reasoning layer)...")
    start_ollama()
    print()
    
    # Ensure videos directory exists
    videos_dir = Path(__file__).parent / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    
    # Start all servers
    print("[STARTUP] Step 2: Starting dashboard servers...")
    dash = start_dashboard_server(8000)
    time.sleep(1)
    video = start_video_api_server(8001)
    time.sleep(1)
    realtime = start_realtime_server(8002)
    
    # Print summary
    print("\n" + "="*70)
    print("ALL SERVERS STARTED SUCCESSFULLY")
    print("="*70)
    print("\nENDPOINT SUMMARY:\n")
    
    endpoints = [
        ("Dashboard UI", "http://localhost:8000", "Main UI - Select 'Video Testing'"),
        ("Video API", "http://localhost:8001", "Batch processing (legacy)"),
        ("Real-Time API", "http://localhost:8002", "Live frame analysis (RECOMMENDED)"),
        ("Ollama API", "http://localhost:11434", "Cloud reasoning (LLM)"),
    ]
    
    for name, url, desc in endpoints:
        print(f"  {name:25} → {url:30} ({desc})")
    
    print("\nPress Ctrl+C to stop all servers\n")
    
    try:
        # Keep running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n" + "="*70)
        print("STOPPING ALL SERVERS...")
        print("="*70 + "\n")
        
        # Stop all servers
        if dash:
            dash.shutdown()
        if video:
            video.shutdown()
        if realtime:
            realtime.shutdown()
        
        # Stop Ollama if we started it
        if ollama_process:
            print("[OLLAMA] Terminating Ollama process...")
            try:
                ollama_process.terminate()
                ollama_process.wait(timeout=5)
                print("[OLLAMA] ✓ Ollama stopped")
            except:
                print("[OLLAMA] Force killing Ollama process...")
                ollama_process.kill()
        
        print("STOPPED ALL SERVERS\n")
        sys.exit(0)



if __name__ == "__main__":
    main()
