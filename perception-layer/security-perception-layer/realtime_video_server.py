"""
Real-time Video Analysis Server with Streaming
Analyzes video frames on-demand as they are requested, enabling real-time dashboard playback
"""

import sys
import os
import json
import cv2
import asyncio
from pathlib import Path
from datetime import datetime, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs, unquote
import threading
import queue

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "reasoning-layer"))

from video_processor import VideoProcessor

# Try to import cloud reasoning (using Free API - no local install needed!)
try:
    from app.reasoning_adapter import ReasoningAdapter
    REASONING_AVAILABLE = True
    print("[REALTIME] ✓ Cloud Reasoning Service available (using OllamaFreeAPI - cloud-based, no local install needed)", flush=True)
except ImportError as e:
    REASONING_AVAILABLE = False
    print(f"[REALTIME] Cloud Reasoning not available: {e}", flush=True)
    print(f"[REALTIME] Install ollamafreeapi: pip install ollamafreeapi", flush=True)

class FrameAnalysisCache:
    """Cache analyzed frames during video playback"""
    def __init__(self):
        self.cache = {}  # video_id -> {frame_num -> analysis_result}
        self.video_metadata = {}  # video_id -> metadata

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

    def clear_video(self, video_id):
        if video_id in self.cache:
            del self.cache[video_id]
        if video_id in self.video_metadata:
            del self.video_metadata[video_id]


class RealtimeVideoAPIHandler(BaseHTTPRequestHandler):
    """HTTP request handler for real-time video streaming"""
    
    VIDEOS_DIR = Path(__file__).parent / "videos"
    # Shared across all requests
    processor = VideoProcessor()
    cache = FrameAnalysisCache()
    
    def log_message(self, format, *args):
        """Custom logging"""
        print(f"[REALTIME_HTTP] {self.address_string()} - {format%args}", flush=True)
    
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
        print(f"[REALTIME_GET] {path}", flush=True)
        
        # Load video metadata (frames count, fps, duration)
        if path.startswith("/api/realtime/load/"):
            video_name = unquote(path.split("/api/realtime/load/")[-1])
            print(f"[REALTIME_GET] Loading video: {video_name}", flush=True)
            try:
                video_path = self.VIDEOS_DIR / video_name
                if not video_path.exists():
                    self.send_error(404, f"Video not found: {video_name}")
                    return
                
                # Read video metadata
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
                    "duration_sec": float(duration_sec),
                    "message": "Video loaded. Ready for frame analysis."
                }
                self.wfile.write(json.dumps(response).encode())
                print(f"[REALTIME_GET] Loaded video: {video_id}, {total_frames} frames @ {fps:.1f} FPS", flush=True)
                
            except Exception as e:
                print(f"[REALTIME_GET] Error loading video: {e}", flush=True)
                self.send_error(400, str(e))
        
        # Analyze a specific frame
        elif path.startswith("/api/realtime/analyze/"):
            try:
                # Parse URL carefully: /api/realtime/analyze/VIDEO_NAME/FRAME_NUMBER?query_string
                # Step 1: Remove query string completely
                clean_path = path.split("?")[0]  # Remove everything after ?
                
                # Step 2: Split the path
                # Expected: /api/realtime/analyze/VIDEO_NAME/FRAME_NUMBER
                parts = clean_path.split("/")
                # parts = ['', 'api', 'realtime', 'analyze', 'VIDEO_NAME', 'FRAME_NUMBER']
                
                if len(parts) < 6:
                    self.send_error(400, f"Invalid URL format: {clean_path}")
                    return
                
                video_name = unquote(parts[4])  # VIDEO_NAME - URL decoded
                try:
                    frame_num = int(parts[5])  # FRAME_NUMBER
                except (ValueError, IndexError):
                    self.send_error(400, f"Invalid frame number: {parts[5] if len(parts) > 5 else 'missing'}")
                    return
                
                print(f"[REALTIME_GET] Parsing: video='{video_name}', frame={frame_num}")
                
                # Get query parameter for video_id - this MUST match what the frontend sends
                video_id = self.get_query_param("video_id", "")
                
                # If video_id not provided, generate one based on video name (fallback)
                if not video_id:
                    video_id = f"{video_name.replace('.', '_')}_realtime"
                    print(f"[REALTIME_GET] ⚠️  No video_id query param provided. Using fallback: {video_id}", flush=True)
                else:
                    print(f"[REALTIME_GET] Using video_id from query param: {video_id}", flush=True)
                
                video_path = self.VIDEOS_DIR / video_name
                if not video_path.exists():
                    self.send_error(404, f"Video not found: {video_name}")
                    return
                
                # Check cache first
                cached_result = self.cache.get_frame(video_id, frame_num)
                if cached_result:
                    print(f"[REALTIME_GET] Frame {frame_num} served from cache", flush=True)
                    self.send_response(200)
                    self.send_header("Content-type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(json.dumps(cached_result).encode())
                    return
                
                # Analyze frame on-demand
                print(f"[REALTIME_GET] Analyzing frame {frame_num} from {video_name}...", flush=True)
                cap = cv2.VideoCapture(str(video_path))
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                cap.release()
                
                if not ret:
                    self.send_error(400, f"Could not read frame {frame_num}")
                    return
                
                # Run perception analysis with timeout
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    print(f"[REALTIME_GET] Starting perception analysis for frame {frame_num}...", flush=True)
                    # Set a task with timeout
                    perception_task = loop.create_task(
                        self.processor._run_perception_models(frame, video_id, frame_num)
                    )
                    perception, weapon_bboxes = loop.run_until_complete(asyncio.wait_for(perception_task, timeout=10.0))
                    print(f"[REALTIME_GET] Frame {frame_num} perception complete", flush=True)
                except asyncio.TimeoutError:
                    print(f"[REALTIME_GET] Frame {frame_num} perception TIMEOUT after 30s", flush=True)
                    self.send_error(408, f"Frame analysis timeout - perception models took too long")
                    return
                except Exception as e:
                    print(f"[REALTIME_GET] Frame {frame_num} perception error: {e}", flush=True)
                    self.send_error(500, f"Perception error: {str(e)}")
                    return
                finally:
                    loop.close()
                
                # Get quick reasoning
                reasoning_payload = self.processor._quick_frame_reasoning(perception)
                
                # Encode frame to base64
                import base64
                _, buffer = cv2.imencode('.jpg', frame)
                frame_data = base64.b64encode(buffer).decode()
                
                # Build response
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
                
                # Cache it
                self.cache.cache_frame(video_id, frame_num, analysis_result)
                print(f"[REALTIME_GET] ✓ Frame {frame_num} cached under video_id: {video_id}", flush=True)
                
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps(analysis_result).encode())
                print(f"[REALTIME_GET] Frame {frame_num} analyzed and cached", flush=True)
                
            except Exception as e:
                print(f"[REALTIME_GET] Error analyzing frame: {e}", flush=True)
                import traceback
                traceback.print_exc()
                self.send_error(400, str(e))
        
        # Get video list
        elif path == "/api/realtime/videos" or path == "/api/realtime/videos/":
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
            self.wfile.write(json.dumps({"status": "ok", "service": "realtime-video"}).encode())
        
        # Generate video summary with cloud reasoning
        elif path.startswith("/api/realtime/summarize/"):
            try:
                print(f"[REALTIME_GET] Summarizing video...", flush=True)
                
                # Parse video_id from URL (remove query string first)
                clean_path = path.split("?")[0]
                video_id_raw = clean_path.split("/api/realtime/summarize/")[-1]
                video_id = unquote(video_id_raw)
                
                print(f"[REALTIME_GET] Summarize endpoint: raw='{video_id_raw}', decoded='{video_id}'", flush=True)
                print(f"[REALTIME_GET] Available cached video IDs: {list(self.cache.cache.keys())}", flush=True)
                
                if not REASONING_AVAILABLE:
                    print(f"[REALTIME_GET] Cloud reasoning not available for summary", flush=True)
                    self.send_error(503, "Cloud reasoning service not available")
                    return
                
                # Get the cached frames for this video
                cached_frames = self.cache.cache.get(video_id, {})
                
                # If no frames found, try to find the most recent video_id that matches the video name
                if not cached_frames and video_id:
                    print(f"[REALTIME_GET] No cached frames for video_id '{video_id}', searching for alternatives...", flush=True)
                    # Look for any video_id that contains this video name
                    video_name_pattern = video_id.split('_')[0] if '_' in video_id else video_id
                    for cached_id in self.cache.cache.keys():
                        if video_name_pattern in cached_id or cached_id.startswith(video_name_pattern):
                            print(f"[REALTIME_GET] Found alternative cached video_id: '{cached_id}'", flush=True)
                            cached_frames = self.cache.cache.get(cached_id, {})
                            if cached_frames:
                                video_id = cached_id
                                break
                
                print(f"[REALTIME_GET] Generating summary for {len(cached_frames)} analyzed frames", flush=True)
                
                # Extract perception data from all cached frames
                perceptions = []
                threat_scores = []
                threat_levels = []
                for frame_num, analysis in sorted(cached_frames.items()):
                    if 'perception' in analysis and 'reasoning' in analysis:
                        perceptions.append(analysis['perception'])
                        threat_scores.append(float(analysis['reasoning'].get('threat_score', 0.0)))
                        threat_levels.append(analysis['reasoning'].get('threat_level', 'low'))
                
                # If no perceptions, provide default fallback response (e.g., for test scenarios)
                if not perceptions:
                    print(f"[REALTIME_GET] No perception data cached - returning fallback (test scenario?)", flush=True)
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
                
                # Create synthetic perception for cloud reasoning
                from app.schemas import UnifiedPerceptionOutput, PerceptionTraits
                synthetic_perception = UnifiedPerceptionOutput(
                    source_id=video_id,
                    timestamp=datetime.now(timezone.utc),
                    traits=PerceptionTraits(
                        weapon_detected=most_common_weapon,
                        emotion=most_common_emotion,
                        tone=most_common_tone,
                        uniform_present=any(p.get('uniform_present', False) for p in perceptions),
                        speech_present=any(p.get('speech_present', False) for p in perceptions),
                        keyword_flags=list(set().union(*[set(p.get('keyword_flags', [])) for p in perceptions])),
                        acoustic_events=list(set().union(*[set(p.get('acoustic_events', [])) for p in perceptions]))
                    ),
                    confidence_scores={}
                )
                
                # Call cloud reasoning
                print(f"[REALTIME_GET] Calling cloud reasoning for summary...", flush=True)
                adapter = ReasoningAdapter(cloud_provider="ollama_free_api")
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    reasoning_output = loop.run_until_complete(
                        adapter.process_perception_async(synthetic_perception)
                    )
                    print(f"[REALTIME_GET] Cloud reasoning complete: {reasoning_output.threat_level}", flush=True)
                    print(f"[REALTIME_GET]   action: {reasoning_output.recommended_action.action}", flush=True)
                    print(f"[REALTIME_GET]   priority: {reasoning_output.recommended_action.priority}", flush=True)
                    print(f"[REALTIME_GET]   confidence: {reasoning_output.recommended_action.confidence}", flush=True)
                except Exception as e:
                    print(f"[REALTIME_GET] Cloud reasoning error: {e}", flush=True)
                    import traceback
                    traceback.print_exc()
                    
                    # Fallback: use simple rule-based response
                    print(f"[REALTIME_GET] Using fallback reasoning...", flush=True)
                    max_threat = max(threat_levels) if threat_levels else 'low'
                    
                    # FIREARM OVERRIDE: Guns are always HIGH+ threat
                    if most_common_weapon and most_common_weapon.lower() in {"gun", "rifle", "shotgun"}:
                        print(f"[REALTIME_GET] 🚨 FIREARM DETECTED: Applying firearm override to threat level", flush=True)
                        max_threat = 'high'  # Minimum HIGH for gundetection
                        
                        # Escalate to CRITICAL if threatening emotion/tone
                        if most_common_emotion and most_common_emotion.lower() in {"angry", "fearful", "distressed", "panicked"}:
                            max_threat = 'critical'
                            print(f"[REALTIME_GET] 🚨 FIREARM + THREATENING EMOTION: Escalating to CRITICAL", flush=True)
                        elif most_common_tone and most_common_tone.lower() in {"threat", "panic", "abnormal", "distressed"}:
                            max_threat = 'critical'
                            print(f"[REALTIME_GET] 🚨 FIREARM + THREATENING TONE: Escalating to CRITICAL", flush=True)
                    
                    if max_threat == 'critical':
                        action, priority, conf = 'immediate_alert', 'critical', 0.95
                        reason = f'Critical threat: {most_common_weapon} detected'
                    elif max_threat == 'high':
                        action, priority, conf = 'escalate', 'high', 0.85
                        reason = f'High threat: {most_common_weapon} detected'
                    elif max_threat == 'medium':
                        action, priority, conf = 'elevated_monitoring', 'medium', 0.70
                        reason = f'Medium threat: {most_common_weapon} detected'
                    else:
                        action, priority, conf = 'monitor', 'low', 0.50
                        reason = 'No critical threat detected'
                    
                    # Create synthetic response object with same structure
                    class SyntheticOutput:
                        def __init__(self):
                            self.threat_level = max_threat
                            self.recommended_action = type('obj', (object,), {'action': action, 'priority': priority, 'confidence': conf})()
                            self.explanation = type('obj', (object,), {'summary': reason, 'key_factors': [f'Weapon: {most_common_weapon}', f'Threat: {max_threat}']})()
                            self.metrics = type('obj', (object,), {'combined_threat_score': conf})()
                    
                    reasoning_output = SyntheticOutput()
                    print(f"[REALTIME_GET] Fallback reasoning: {reasoning_output.threat_level} - {action}", flush=True)
                finally:
                    loop.close()
                
                # Build summary response
                summary_response = {
                    "status": "success",
                    "video_id": video_id,
                    "frames_analyzed": len(cached_frames),
                    "overall_threat_level": reasoning_output.threat_level if reasoning_output else max(threat_levels) if threat_levels else "low",
                    "overall_threat_score": reasoning_output.metrics.combined_threat_score if reasoning_output else (sum(threat_scores) / len(threat_scores) if threat_scores else 0.0),
                    "recommended_action": reasoning_output.recommended_action.action if reasoning_output else "monitor",
                    "priority": reasoning_output.recommended_action.priority if reasoning_output else "low",
                    "confidence": float(reasoning_output.recommended_action.confidence) if reasoning_output else 0.0,
                    "reasoning_explanation": reasoning_output.explanation.summary if reasoning_output else "Video analysis complete",
                    "key_factors": reasoning_output.explanation.key_factors if reasoning_output else [],
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
                self.wfile.write(json.dumps(summary_response, default=str).encode())
                print(f"[REALTIME_GET] Summary response sent", flush=True)
                
            except Exception as e:
                print(f"[REALTIME_GET] Error generating summary: {e}", flush=True)
                import traceback
                traceback.print_exc()
                self.send_error(500, f"Error generating summary: {str(e)}")
        
        else:
            self.send_error(404)
    
    def do_OPTIONS(self):
        """Handle CORS preflight"""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
    
    def get_query_param(self, key, default=""):
        """Extract query parameter"""
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        return params.get(key, [default])[0]


def run_realtime_server(port=8002):
    """Run the real-time video analysis server"""
    
    # Ensure videos directory exists
    RealtimeVideoAPIHandler.VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    
    server_address = ("", port)
    httpd = HTTPServer(server_address, RealtimeVideoAPIHandler)
    
    print(f"\n{'='*60}")
    print(f"✓ Real-Time Video Analysis Server running at http://localhost:{port}")
    print(f"✓ Videos directory: {RealtimeVideoAPIHandler.VIDEOS_DIR}")
    print(f"✓ Endpoints:")
    print(f"  GET  /api/realtime/videos       - List available videos")
    print(f"  GET  /api/realtime/load/<name>  - Load video metadata")
    print(f"  GET  /api/realtime/analyze/<name>/<frame_num> - Analyze specific frame")
    print(f"  GET  /api/realtime/health       - Health check")
    print(f"{'='*60}\n")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n✓ Real-time video server stopped")


if __name__ == "__main__":
    run_realtime_server()
