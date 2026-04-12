"""
Video processing server for testing detection and reasoning
Handles video uploads and processes frames through perception and reasoning layers
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import json
import cv2
import base64
from threading import Thread
import queue
import asyncio
import tempfile

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "reasoning-layer"))

# Force unbuffered output
sys.stdout = buffer = sys.stdout
os.environ['PYTHONUNBUFFERED'] = '1'

from app.reasoning_adapter import get_reasoning_adapter
from app.schemas import UnifiedPerceptionOutput, UnifiedTraits, UnifiedDecision, PerceptionRequest, VideoInput, AudioInput, PerceptionResponse, AudioDetection
from app.services.pipeline import perception_pipeline


# Create a single event loop for the entire video processor
def _get_event_loop():
    """Get or create an event loop for async operations"""
    try:
        loop = asyncio.get_running_loop()
        return loop
    except RuntimeError:
        pass
    
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError("Event loop is closed")
        return loop
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop

class VideoProcessor:
    """Process video frames and generate detection results"""
    
    def __init__(self):
        # Default to rule-based reasoning for video playback responsiveness.
        # This can be overridden with VIDEO_REASONING_PROVIDER.
        provider = os.getenv("VIDEO_REASONING_PROVIDER", "cloud_agent")
        self.reasoning_adapter = get_reasoning_adapter(cloud_provider=provider)
        self.processing_queue = queue.Queue()
        self.results_cache = {}
        self.weapon_bboxes = {}  # Store bounding boxes per frame

    async def _run_audio_analysis_once(self, video_path: str, video_id: str) -> AudioDetection:
        """
        Run audio analysis once for the entire video before processing frames.
        This avoids re-analyzing the same audio for every frame.
        
        Args:
            video_path: Path to video file
            video_id: Unique identifier for this processing session
        
        Returns:
            AudioDetection object with tone, confidence, and other audio features
        """
        try:
            print(f"[AUDIO_ONCE] Starting single audio analysis for video: {video_path}", flush=True)
            
            # Create perception request with audio source (points to original video file)
            request = PerceptionRequest(
                source_id=f"video-audio-{video_id}",
                audio=AudioInput(
                    source_type="stream_audio",
                    uri=str(video_path)
                )
            )
            
            # Run only the audio model - direct call instead of full pipeline
            from app.services.models.audio import AudioThreatDetectionAdapter
            audio_model = AudioThreatDetectionAdapter()
            audio_result = await audio_model.infer(request)
            
            print(f"[AUDIO_ONCE] SUCCESS: Audio analysis complete: tone={audio_result.tone}, "
                  f"confidence={audio_result.confidence}, speech_present={audio_result.speech_present}", flush=True)
            
            return audio_result
            
        except Exception as e:
            print(f"[AUDIO_ONCE] ERROR: Error during audio analysis: {str(e)}", flush=True)
            import traceback
            traceback.print_exc()
            # Return safe default if audio analysis fails
            return AudioDetection(
                tone="unknown",
                confidence=0.0,
                speech_present=False,
            )

    def _build_reasoning_payload(self, reasoning, perception):
        """Normalize reasoning output to a stable JSON-safe payload."""
        metrics = getattr(reasoning, 'metrics', None)
        action = getattr(reasoning, 'recommended_action', None)
        explanation = getattr(reasoning, 'explanation', None)

        threat_level = getattr(reasoning, 'threat_level', None) or 'medium'
        threat_score = float(getattr(metrics, 'combined_threat_score', 0.0) or 0.0)

        action_name = getattr(action, 'action', None) or 'monitor'
        priority = getattr(action, 'priority', None) or 'medium'
        confidence = float(getattr(action, 'confidence', threat_score) or threat_score)
        reason = getattr(action, 'reason', None) or 'Reasoning fallback applied due to unavailable details.'

        summary = getattr(explanation, 'summary', None) or 'Reasoning details unavailable for this frame.'
        key_factors = getattr(explanation, 'key_factors', None) or [
            f"Weapon: {perception.traits.weapon_detected}",
            f"Emotion: {perception.traits.emotion}",
            f"Tone: {perception.traits.tone}"
        ]
        anomalies = getattr(explanation, 'anomalies_detected', None) or []

        return {
            'threat_level': threat_level,
            'threat_score': threat_score,
            'action': action_name,
            'priority': priority,
            'confidence': confidence,
            'reason': reason,
            'summary': summary,
            'key_factors': key_factors,
            'anomalies': anomalies,
            'trend': getattr(metrics, 'trend', None) or 'stable',
            'metrics': {
                'weapon_score': float(getattr(metrics, 'weapon_threat_score', 0.0) or 0.0),
                'emotion_score': float(getattr(metrics, 'emotion_threat_score', 0.0) or 0.0),
                'audio_score': float(getattr(metrics, 'audio_threat_score', 0.0) or 0.0),
                'behavioral_score': float(getattr(metrics, 'behavioral_anomaly_score', 0.0) or 0.0),
            }
        }

    def _pick_most_common(self, values, fallback="unknown"):
        """
        Return most frequent non-empty value from a list.
        
        SPECIAL CASE for weapons (security priority):
        - If ANY critical weapon (gun/rifle/shotgun) detected, return that
        - Else if ANY high-threat weapon (knife/blade) detected, return that
        - Else return most common (emotion/tone use cases)
        - Else return fallback
        """
        counts = {}
        for value in values:
            if value is None:
                continue
            key = str(value).strip().lower()
            if not key or key == "—":
                continue
            counts[key] = counts.get(key, 0) + 1

        if not counts:
            return fallback
        
        # For weapon detection: PRIORITY MODE (not frequency mode)
        # This is critical for security - ANY gun detected means weapon status is "gun"
        # NOT "unarmed just because 79% of frames were peaceful"
        critical_weapons = {"gun", "rifle", "shotgun"}
        high_threat_weapons = {"knife", "blade"}
        
        # Check for critical weapons first
        for weapon in critical_weapons:
            if weapon in counts:
                return weapon
        
        # Then check for high-threat weapons
        for weapon in high_threat_weapons:
            if weapon in counts:
                return weapon
        
        # For non-weapon cases (emotion, tone), use frequency
        return max(counts.items(), key=lambda item: item[1])[0]

    def _quick_frame_reasoning(self, perception):
        """Fast heuristic frame reasoning used before video-level cloud reasoning."""
        weapon = str(perception.traits.weapon_detected or "unknown").lower()
        emotion = str(perception.traits.emotion or "unknown").lower()
        tone = str(perception.traits.tone or "unknown").lower()

        score = 0.0
        if weapon in {"gun", "rifle", "shotgun"}:
            score += 0.75
        elif weapon in {"knife", "blade"}:
            score += 0.55
        elif weapon not in {"unarmed", "unknown", "none"}:
            score += 0.35

        if emotion in {"angry", "fearful", "distressed", "panicked"}:
            score += 0.2
        if tone in {"threat", "panic", "abnormal", "distressed"}:
            score += 0.2
        if perception.traits.uniform_present:
            score -= 0.15

        score = max(0.0, min(1.0, score))
        if score >= 0.75:
            level = "critical"
            action = "immediate_alert"
            priority = "critical"
        elif score >= 0.5:
            level = "high"
            action = "escalate"
            priority = "high"
        elif score >= 0.25:
            level = "medium"
            action = "elevated_monitoring"
            priority = "medium"
        else:
            level = "low"
            action = "monitor"
            priority = "low"

        anomalies = []
        if weapon not in {"unarmed", "unknown", "none"}:
            anomalies.append("armed_individual")
        if emotion in {"angry", "fearful", "distressed", "panicked"}:
            anomalies.append("emotional_distress")
        if tone in {"threat", "panic", "abnormal", "distressed"}:
            anomalies.append("audio_escalation")

        return {
            'threat_level': level,
            'threat_score': score,
            'action': action,
            'priority': priority,
            'confidence': min(1.0, 0.5 + score / 2.0),
            'reason': 'Fast per-frame heuristic used; final recommendation is computed at video summary stage.',
            'summary': f'Frame-level threat {level} from weapon={weapon}, emotion={emotion}, tone={tone}.',
            'key_factors': [
                f'weapon={weapon}',
                f'emotion={emotion}',
                f'tone={tone}'
            ],
            'anomalies': anomalies,
            'trend': 'frame-local',
            'metrics': {
                'weapon_score': min(1.0, 1.0 if weapon in {'gun', 'rifle', 'shotgun'} else 0.7 if weapon in {'knife', 'blade'} else 0.3 if weapon not in {'unarmed', 'unknown', 'none'} else 0.0),
                'emotion_score': 0.7 if emotion in {'angry', 'fearful', 'distressed', 'panicked'} else 0.0,
                'audio_score': 0.7 if tone in {'threat', 'panic', 'abnormal', 'distressed'} else 0.0,
                'behavioral_score': score,
            }
        }

    def _build_summary_perception(self, summary, video_id):
        """Create a synthetic unified perception object for one cloud reasoning call."""
        threat_level = summary.get('overall_threat_level', 'medium')
        threat_score = float(summary.get('overall_threat_score', 0.0) or 0.0)
        dominant = summary.get('dominant_traits', {})

        weapon = dominant.get('weapon', 'unknown')
        emotion = dominant.get('emotion', 'unknown')
        tone = dominant.get('tone', 'unknown')

        risk_hints = []
        if weapon not in {'unarmed', 'unknown', 'none'}:
            risk_hints.append('visible_weapon')
        if emotion in {'angry', 'fearful', 'distressed', 'panicked'}:
            risk_hints.append('emotional_escalation')
        if tone in {'threat', 'panic', 'abnormal', 'distressed'}:
            risk_hints.append('audio_escalation')
        if summary.get('peak_threat_score', 0.0) and float(summary.get('peak_threat_score', 0.0)) >= 0.8:
            risk_hints.append('peak_frame_critical')

        # CRITICAL FIX for weapon confidence:
        # If weapon detected in ANY frame, we are 100% confident it exists.
        # weapon_frame_ratio tells us HOW OFTEN it appeared (21.7%), not confidence.
        weapon_confidence = (
            1.0 if weapon not in {'unarmed', 'unknown', 'none'} 
            else 0.0
        )
        weapon_frame_ratio = float(summary.get('weapon_frame_ratio', 0.0) or 0.0)
        
        # For logging/diagnostics, include both pieces of info
        if weapon_confidence == 1.0 and weapon_frame_ratio > 0:
            risk_hints.append(f'weapon_present_in_{int(weapon_frame_ratio*100)}pct_of_frames')

        confidence_scores = {
            'weapon': weapon_confidence,  # 1.0 if weapon detected, 0.0 if unarmed
            'emotion': threat_score,
            'tone': min(1.0, float(summary.get('peak_threat_score', 0.0) or 0.0)),
            'uniform': float(summary.get('uniform_frame_ratio', 0.0) or 0.0),
        }

        return UnifiedPerceptionOutput(
            source_id=f"video-summary-{video_id}",
            timestamp=datetime.now(),
            model_backends={'video_summary': 'aggregated_detection_frames'},
            confidence_scores=confidence_scores,
            traits=UnifiedTraits(
                weapon_detected=weapon,
                raw_weapon_detected=weapon,
                weapon_class_evidence=[],
                visual_secondary_evidence=[],
                uniform_present=(float(summary.get('uniform_frame_ratio', 0.0) or 0.0) >= 0.5),
                uniform_confidence=float(summary.get('uniform_frame_ratio', 0.0) or 0.0),
                uniform_evidence=[],
                weapon_suppressed_due_to_uniform=False,
                emotion=emotion,
                tone=tone,
                speech_present=False,
                acoustic_events=[],
                keyword_flags=[],
                transcript=None
            ),
            risk_hints=risk_hints,
            decision=UnifiedDecision(
                threat_level=threat_level,
                anomaly_type=[a.get('name') for a in summary.get('top_anomalies', [])],
                recommended_response=[summary.get('recommended_action', 'monitor')],
                confidence=float(summary.get('confidence', 0.0) or 0.0),
                rationale_summary=summary.get('summary_text', 'Video summary generated from detection frames')
            )
        )

    def _apply_cloud_reasoning_to_summary(self, summary, video_id):
        """Run one cloud reasoning pass on aggregated summary and merge recommendation."""
        try:
            aggregated_perception = self._build_summary_perception(summary, video_id)
            reasoning = self.reasoning_adapter.process_perception(aggregated_perception)
            cloud_payload = self._build_reasoning_payload(reasoning, aggregated_perception)

            summary['recommended_action'] = cloud_payload.get('action', summary.get('recommended_action', 'monitor'))
            summary['priority'] = cloud_payload.get('priority', summary.get('priority', 'medium'))
            summary['confidence'] = float(cloud_payload.get('confidence', summary.get('confidence', 0.0)) or 0.0)
            summary['cloud_reasoning'] = {
                'threat_level': cloud_payload.get('threat_level', summary.get('overall_threat_level', 'medium')),
                'threat_score': float(cloud_payload.get('threat_score', summary.get('overall_threat_score', 0.0)) or 0.0),
                'action': cloud_payload.get('action', summary.get('recommended_action', 'monitor')),
                'priority': cloud_payload.get('priority', summary.get('priority', 'medium')),
                'confidence': float(cloud_payload.get('confidence', summary.get('confidence', 0.0)) or 0.0),
                'reason': cloud_payload.get('reason', ''),
                'summary': cloud_payload.get('summary', ''),
                'provider': 'single_pass_video_summary'
            }
            summary['summary_text'] = cloud_payload.get('summary', summary.get('summary_text', ''))
            return summary
        except Exception as e:
            summary['cloud_reasoning'] = {
                'provider': 'single_pass_video_summary',
                'status': 'failed',
                'error': str(e)
            }
            return summary

    def _build_video_summary(self, detections):
        """Build one aggregated summary from all processed detection frames."""
        total = len(detections)
        if total == 0:
            return {
                'total_detection_frames': 0,
                'overall_threat_level': 'low',
                'overall_threat_score': 0.0,
                'peak_threat_score': 0.0,
                'recommended_action': 'monitor',
                'priority': 'low',
                'confidence': 0.0,
                'threat_distribution': {'low': 0, 'medium': 0, 'high': 0, 'critical': 0},
                'dominant_traits': {
                    'weapon': 'unknown',
                    'emotion': 'unknown',
                    'tone': 'unknown'
                },
                'weapon_frame_ratio': 0.0,
                'uniform_frame_ratio': 0.0,
                'top_anomalies': [],
                'summary_text': 'No detection frames were processed.'
            }

        threat_distribution = {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}
        threat_weights = {'low': 1.0, 'medium': 2.0, 'high': 3.0, 'critical': 4.0}
        weighted_sum = 0.0
        weight_total = 0.0
        threat_scores = []
        confidences = []

        weapons = []
        emotions = []
        tones = []
        weapon_frames = 0
        uniform_frames = 0
        anomalies_counter = {}

        for detection in detections:
            perception = detection.get('perception', {})
            reasoning = detection.get('reasoning', {})
            level = str(reasoning.get('threat_level', 'medium')).lower()
            if level not in threat_distribution:
                level = 'medium'

            threat_distribution[level] += 1
            confidence = float(reasoning.get('confidence', 0.0) or 0.0)
            weighted_sum += threat_weights[level] * max(confidence, 0.1)
            weight_total += max(confidence, 0.1)

            score = float(reasoning.get('threat_score', 0.0) or 0.0)
            threat_scores.append(score)
            confidences.append(confidence)

            weapon = perception.get('weapon_detected', 'unknown')
            emotion = perception.get('emotion', 'unknown')
            tone = perception.get('tone', 'unknown')
            weapons.append(weapon)
            emotions.append(emotion)
            tones.append(tone)

            if str(weapon).lower() not in {'unarmed', 'unknown', 'none'}:
                weapon_frames += 1
            if bool(perception.get('uniform_present', False)):
                uniform_frames += 1

            for anomaly in reasoning.get('anomalies', []) or []:
                key = str(anomaly).strip().lower()
                if not key:
                    continue
                anomalies_counter[key] = anomalies_counter.get(key, 0) + 1

        normalized_threat = (weighted_sum / weight_total) if weight_total else 1.0
        if normalized_threat >= 3.5:
            overall_level = 'critical'
        elif normalized_threat >= 2.5:
            overall_level = 'high'
        elif normalized_threat >= 1.5:
            overall_level = 'medium'
        else:
            overall_level = 'low'

        critical_count = threat_distribution['critical']
        high_count = threat_distribution['high']
        medium_count = threat_distribution['medium']
        critical_ratio = critical_count / total
        high_ratio = high_count / total
        medium_ratio = medium_count / total

        if critical_count > 0 or critical_ratio >= 0.1:
            action = 'immediate_alert'
            priority = 'critical'
        elif high_ratio >= 0.25:
            action = 'escalate'
            priority = 'high'
        elif medium_ratio >= 0.4:
            action = 'elevated_monitoring'
            priority = 'medium'
        else:
            action = 'monitor'
            priority = 'low'

        avg_threat_score = sum(threat_scores) / total
        peak_threat_score = max(threat_scores) if threat_scores else 0.0
        avg_confidence = sum(confidences) / total if confidences else 0.0

        dominant_weapon = self._pick_most_common(weapons)
        dominant_emotion = self._pick_most_common(emotions)
        dominant_tone = self._pick_most_common(tones)

        top_anomalies = [
            {'name': name, 'count': count}
            for name, count in sorted(anomalies_counter.items(), key=lambda item: item[1], reverse=True)[:5]
        ]

        summary_text = (
            f"Processed {total} detection frames. "
            f"Overall threat: {overall_level.upper()} (avg score {avg_threat_score:.2f}, peak {peak_threat_score:.2f}). "
            f"Dominant traits: weapon={dominant_weapon}, emotion={dominant_emotion}, tone={dominant_tone}."
        )

        return {
            'total_detection_frames': total,
            'overall_threat_level': overall_level,
            'overall_threat_score': round(avg_threat_score, 4),
            'peak_threat_score': round(peak_threat_score, 4),
            'recommended_action': action,
            'priority': priority,
            'confidence': round(avg_confidence, 4),
            'threat_distribution': threat_distribution,
            'dominant_traits': {
                'weapon': dominant_weapon,
                'emotion': dominant_emotion,
                'tone': dominant_tone
            },
            'weapon_frame_ratio': round(weapon_frames / total, 4),
            'uniform_frame_ratio': round(uniform_frames / total, 4),
            'top_anomalies': top_anomalies,
            'summary_text': summary_text
        }
        
    def process_video(self, video_path, video_id, fps_sample=2.0, force_reprocess=True):
        """
        Process video file frame by frame
        
        Args:
            video_path: Path to video file
            video_id: Unique identifier for this processing session
            fps_sample: Sample every Nth frame (e.g., 2.0 = every other frame)
            force_reprocess: If True, always reprocess even if cached results exist
        
        Returns:
            {
                'video_id': str,
                'total_frames': int,
                'processed_frames': int,
                'detections': list of detection results,
                'status': 'completed' | 'error'
            }
        """
        
        print(f"\n\n{'='*80}", flush=True)
        print(f"[VIDEO] STARTING VIDEO PROCESSING", flush=True)
        print(f"[VIDEO] video_path={video_path}", flush=True)
        print(f"[VIDEO] video_id={video_id}", flush=True)
        print(f"[VIDEO] fps_sample={fps_sample}", flush=True)
        print(f"[VIDEO] force_reprocess={force_reprocess}", flush=True)
        print(f"{'='*80}\n", flush=True)
        
        # Check cache UNLESS force_reprocess is True
        if not force_reprocess and video_id in self.results_cache:
            print(f"[VIDEO] Using cached results for {video_id}", flush=True)
            return self.results_cache[video_id]
        
        print(f"[VIDEO] Cache bypass: Processing fresh", flush=True)

        self.results_cache[video_id] = {
            'video_id': video_id,
            'status': 'processing',
            'processed_frames': 0,
            'total_frames': 0,
            'progress_pct': 0.0,
            'detections': []
        }
        
        try:
            # ===== STEP 1: RUN AUDIO ANALYSIS ONCE FOR ENTIRE VIDEO =====
            loop = _get_event_loop()
            print(f"[VIDEO] Step 1/2: Running audio analysis...", flush=True)
            audio_result = loop.run_until_complete(self._run_audio_analysis_once(video_path, video_id))
            
            # ===== STEP 2: PROCESS VIDEO FRAMES WITH PRECOMPUTED AUDIO =====
            print(f"[VIDEO] Step 2/2: Processing frames with vision models...", flush=True)
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            frame_count = 0
            processed_count = 0
            detections = []
            frame_sample_interval = max(1, int(fps / fps_sample))
            
            # Get event loop for async operations
            loop = _get_event_loop()
            
            print(f"[VIDEO] Processing {video_id}: {total_frames} frames @ {fps:.1f} FPS")
            print(f"[VIDEO] Sampling every {frame_sample_interval} frames (Sample rate: {fps_sample} FPS)")
            self.results_cache[video_id]['total_frames'] = total_frames
            
            while frame_count < total_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Sample frames
                if frame_count % frame_sample_interval == 0:
                    # Run perception models on actual frame - returns unified output + weapon bboxes
                    # Use asyncio.run_until_complete with fallback for event loop issues
                    try:
                        perception, weapon_bboxes = loop.run_until_complete(
                            self._run_perception_models(frame, video_id, frame_count, video_path, audio_result)
                        )
                    except RuntimeError as e:
                        print(f"[VIDEO] Event loop error: {e}. Creating new loop...", flush=True)
                        # Create a fresh event loop for this thread
                        new_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(new_loop)
                        perception, weapon_bboxes = new_loop.run_until_complete(
                            self._run_perception_models(frame, video_id, frame_count, video_path, audio_result)
                        )
                        loop = new_loop
                    
                    # Use fast frame-level heuristic; one cloud pass is done after aggregation.
                    reasoning_payload = self._quick_frame_reasoning(perception)
                    
                    # Draw bounding boxes on frame for visualization
                    display_frame = frame.copy()
                    self._draw_detections(display_frame, perception, weapon_bboxes)
                    
                    # Encode frame for transmission
                    _, buffer = cv2.imencode('.jpg', display_frame)
                    frame_b64 = base64.b64encode(buffer).decode('utf-8')
                    detection_result = {
                        'frame_number': frame_count,
                        'timestamp_sec': frame_count / fps,
                        'frame_data': f"data:image/jpeg;base64,{frame_b64}",
                        'perception': {
                            'weapon_detected': perception.traits.weapon_detected,
                            'emotion': perception.traits.emotion,
                            'tone': perception.traits.tone,
                            'uniform_present': perception.traits.uniform_present,
                        },
                        'reasoning': reasoning_payload
                    }
                    
                    detections.append(detection_result)
                    processed_count += 1

                    self.results_cache[video_id]['processed_frames'] = processed_count
                    self.results_cache[video_id]['progress_pct'] = round((frame_count / max(1, total_frames)) * 100.0, 2)
                    self.results_cache[video_id]['detections'] = detections
                    
                    # Print progress
                    if processed_count % 10 == 0:
                        print(f"[VIDEO] Processed {processed_count} frames...")
                
                frame_count += 1
            
            cap.release()
            video_summary = self._build_video_summary(detections)
            video_summary = self._apply_cloud_reasoning_to_summary(video_summary, video_id)
            
            result = {
                'video_id': video_id,
                'total_frames': total_frames,
                'processed_frames': processed_count,
                'fps': fps,
                'duration_sec': total_frames / fps,
                'progress_pct': 100.0,
                'detections': detections,
                'summary': video_summary,
                'status': 'completed'
            }
            
            self.results_cache[video_id] = result
            print(f"\n[VIDEO] Completed {video_id}: {processed_count} frames processed out of {total_frames}", flush=True)
            print(f"[VIDEO] Detections in result: {len(detections)} frames with frame_data", flush=True)
            if detections:
                print(f"[VIDEO] First detection keys: {list(detections[0].keys())}", flush=True)
                print(f"[VIDEO] First detection has frame_data: {'frame_data' in detections[0]}", flush=True)
            return result
            
        except Exception as e:
            print(f"[VIDEO] Error processing {video_id}: {str(e)}")
            error_result = {
                'video_id': video_id,
                'status': 'error',
                'error': str(e)
            }
            self.results_cache[video_id] = error_result
            return error_result
    
    async def _run_perception_models(self, frame, video_id, frame_number, video_path, precomputed_audio: AudioDetection):
        """
        Run vision-based perception models on video frame.
        
        Audio analysis is now run separately once for the entire video and passed in,
        avoiding redundant re-analysis of the same audio for every frame.
        
        Args:
            frame: Individual video frame
            video_id: Unique ID for the video
            frame_number: Frame sequence number
            video_path: Path to original video file
            precomputed_audio: AudioDetection result from single video-level analysis
        
        Returns:
            Tuple of (UnifiedPerceptionOutput, weapon_bounding_boxes)
        """
        temp_frame_path = None
        try:
            # Save frame to temporary file so models can process it
            # WeaponDetectionAdapter requires a file URI, not raw bytes
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                temp_frame_path = tmp.name
                cv2.imwrite(temp_frame_path, frame)
            
            # Create perception request with frame file for vision models only
            # NOTE: We do NOT include audio here - audio was already processed separately
            request = PerceptionRequest(
                source_id=f"video-{video_id}",
                video=VideoInput(
                    stream_type="uploaded_video",
                    uri=temp_frame_path,
                    frame_sample_fps=2.0,
                    camera_id=video_id
                ),
                # audio=None  # Skip audio - use precomputed result instead
            )
            
            # Run through perception pipeline - bypasses audio model
            # The pipeline will handle vision models only since audio URI is not provided
            print(f"[PERCEPTION] Running VISION models on frame {frame_number}...", flush=True)
            response = await perception_pipeline.run(request)
            unified_output = response.unified_output
            weapon_model = response.weapon_model_output
            
            # INJECT PRECOMPUTED AUDIO DATA into unified output
            # This ensures all frames have consistent audio tone from the video-level analysis
            unified_output.traits.tone = precomputed_audio.tone
            unified_output.traits.speech_present = precomputed_audio.speech_present
            unified_output.traits.acoustic_events = precomputed_audio.acoustic_events
            unified_output.traits.keyword_flags = precomputed_audio.keyword_flags
            unified_output.traits.transcript = precomputed_audio.transcript
            
            print(f"[PERCEPTION] Frame {frame_number}: "
                  f"weapon={unified_output.traits.weapon_detected} (confidence={weapon_model.confidence}), "
                  f"bounding_boxes={weapon_model.bounding_boxes}, "
                  f"emotion={unified_output.traits.emotion}, "
                  f"tone={unified_output.traits.tone} (precomputed), "
                  f"uniform={unified_output.traits.uniform_present}", flush=True)
            
            # Store weapon bounding boxes for visualization
            self.weapon_bboxes[frame_number] = weapon_model.bounding_boxes
            
            return unified_output, weapon_model.bounding_boxes
            
        except Exception as e:
            print(f"[PERCEPTION] Error processing frame {frame_number}: {str(e)}")
            import traceback
            traceback.print_exc()
            # Fallback to safe defaults if model fails
            return UnifiedPerceptionOutput(
                source_id=f"video-{video_id}",
                timestamp=datetime.now(),
                model_backends={},
                confidence_scores={},
                traits=UnifiedTraits(
                    weapon_detected="unknown",
                    raw_weapon_detected="unknown",
                    weapon_class_evidence=[],
                    visual_secondary_evidence=[],
                    uniform_present=False,
                    uniform_confidence=0.0,
                    uniform_evidence=[],
                    weapon_suppressed_due_to_uniform=False,
                    emotion="unknown",
                    tone="unknown",
                    speech_present=False,
                    acoustic_events=[],
                    keyword_flags=[],
                    transcript=None
                ),
                risk_hints=[],
                decision=UnifiedDecision(
                    threat_level="unknown",
                    anomaly_type=[],
                    recommended_response=[],
                    confidence=0.0,
                    rationale_summary="Model inference failed"
                )
            ), []
        finally:
            # Clean up temporary file
            if temp_frame_path and os.path.exists(temp_frame_path):
                try:
                    os.unlink(temp_frame_path)
                except Exception as e:
                    print(f"[WARNING] Failed to delete temp file {temp_frame_path}: {e}")
    
    def _draw_detections(self, frame, perception, weapon_bboxes):
        """
        Draw detection bounding boxes and labels on frame
        
        Args:
            frame: OpenCV frame to draw on
            perception: UnifiedPerceptionOutput with detection results
            weapon_bboxes: List of actual YOLO bounding boxes [[x1, y1, x2, y2], ...]
        """
        height, width = frame.shape[:2]
        
        print(f"\n[DRAW] ===== DETECTION DRAWING START =====", flush=True)
        print(f"[DRAW] Frame size: {width}x{height}", flush=True)
        print(f"[DRAW] weapon_detected: {perception.traits.weapon_detected}", flush=True)
        print(f"[DRAW] weapon_bboxes type: {type(weapon_bboxes)}", flush=True)
        print(f"[DRAW] weapon_bboxes: {weapon_bboxes}", flush=True)
        print(f"[DRAW] num_bboxes: {len(weapon_bboxes) if weapon_bboxes else 0}", flush=True)
        
        # Draw weapon detection bounding boxes from YOLO
        if weapon_bboxes and len(weapon_bboxes) > 0:
            print(f"[DRAW] SUCCESS: Processing {len(weapon_bboxes)} bounding boxes", flush=True)
            # weapon_bboxes format: [[x1, y1, x2, y2], ...] with normalized or absolute coordinates
            for idx, bbox in enumerate(weapon_bboxes):
                print(f"[DRAW]   Box #{idx}: {bbox} (type: {type(bbox)})", flush=True)
                if bbox and len(bbox) >= 4:
                    try:
                        x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
                        print(f"[DRAW]   => Raw coords: ({x1}, {y1}) to ({x2}, {y2})", flush=True)
                        
                        # Check if coordinates are normalized (0-1) or absolute
                        if x1 <= 1.0 and y1 <= 1.0 and x2 <= 1.0 and y2 <= 1.0:
                            # Normalized coordinates - scale to frame size
                            x1_px = int(x1 * width)
                            y1_px = int(y1 * height)
                            x2_px = int(x2 * width)
                            y2_px = int(y2 * height)
                            print(f"[DRAW]   => Normalized: scaled to ({x1_px}, {y1_px}) to ({x2_px}, {y2_px})", flush=True)
                        else:
                            # Already absolute coordinates
                            x1_px, y1_px, x2_px, y2_px = int(x1), int(y1), int(x2), int(y2)
                            print(f"[DRAW]   => Absolute: pixel coords ({x1_px}, {y1_px}) to ({x2_px}, {y2_px})", flush=True)
                        
                        # Validate coordinates are within frame
                        if x1_px < 0 or y1_px < 0 or x2_px > width or y2_px > height:
                            print(f"[DRAW]   WARNING: Coords out of bounds! Frame is {width}x{height}", flush=True)
                        
                        # Draw red bounding box for weapon
                        print(f"[DRAW]   => Drawing rectangle from ({x1_px}, {y1_px}) to ({x2_px}, {y2_px})", flush=True)
                        cv2.rectangle(frame, (x1_px, y1_px), (x2_px, y2_px), (0, 0, 255), 3)
                        
                        # Draw weapon label with confidence
                        label = f"{perception.traits.weapon_detected.upper()}"
                        conf = perception.confidence_scores.get('weapon', 0.0) if perception.confidence_scores else 0.0
                        text = f"{label} ({conf:.2f})"
                        
                        # Draw label background
                        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                        cv2.rectangle(frame, (x1_px, y1_px - text_size[1] - 5), 
                                     (x1_px + text_size[0], y1_px), (0, 0, 255), -1)
                        cv2.putText(frame, text, (x1_px, y1_px - 5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        print(f"[DRAW]   SUCCESS: Drawn successfully", flush=True)
                    except Exception as e:
                        print(f"[DRAW]   ERROR: Error drawing bbox: {e}", flush=True)
                else:
                    print(f"[DRAW]   ERROR: Invalid bbox format: {bbox}", flush=True)
        else:
            print(f"[DRAW] ERROR: No bounding boxes to draw", flush=True)
        
        # Draw uniform detection if present
        if perception.traits.uniform_present:
            # Draw green border for uniform
            cv2.rectangle(frame, (5, 5), (width - 5, height - 5), (0, 255, 0), 2)
            cv2.putText(frame, "UNIFORM DETECTED", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Add emotion and tone information at bottom
        info_y = height - 60
        cv2.putText(frame, f"Emotion: {perception.traits.emotion}", (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Tone: {perception.traits.tone}", (10, info_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        print(f"[DRAW] ===== DETECTION DRAWING END =====\n", flush=True)
    
    def get_results(self, video_id):
        """Retrieve cached results for a video"""
        return self.results_cache.get(video_id)
    
    def list_videos(self, video_dir):
        """List available video files"""
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
        video_dir = Path(video_dir)
        
        if not video_dir.exists():
            return []
        
        videos = []
        for file in sorted(video_dir.glob('*')):
            if file.suffix.lower() in video_extensions:
                stat = file.stat()
                videos.append({
                    'name': file.name,
                    'path': str(file),
                    'size_mb': stat.st_size / (1024*1024),
                    'modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
                })
        
        return videos


# Global processor instance
processor = VideoProcessor()
"""
def process_video_async(video_path, video_id):
    #Process video in background thread - always reprocess fresh
    print(f"\n[ASYNC] process_video_async called with video_id={video_id}", flush=True)
    print(f"[ASYNC] video_path={video_path}", flush=True)
    print(f"[ASYNC] path exists: {Path(video_path).exists()}", flush=True)
    
    log_file = Path(__file__).parent / f"processing_{video_id}.log"
    

    def _safe_process():
        # Wrapper with error handling for background thread
        try:
            with open(log_file, 'w') as f:
                f.write(f"[ASYNC_WORKER] Starting processing for {video_id}\n")
                f.write(f"[ASYNC_WORKER] video_path={video_path}\n")
                f.write(f"[ASYNC_WORKER] path exists: {Path(video_path).exists()}\n")
                f.flush()
            
            print(f"[ASYNC_WORKER] Starting processing for {video_id}", flush=True)
            result = processor.process_video(video_path, video_id, 2.0, True)  # force_reprocess=True
            
            with open(log_file, 'a') as f:
                f.write(f"[ASYNC_WORKER] Processing returned: {result['status']}\n")
                f.write(f"[ASYNC_WORKER] Processed {result.get('processed_frames', 0)} frames\n")
                f.write(f"[ASYNC_WORKER] Detections: {len(result.get('detections', []))} frames\n")
                f.flush()
            print(f"[ASYNC_WORKER] Result: status={result['status']}, frames={result.get('processed_frames', 0)}", flush=True)
            
            with open(log_file, 'a') as f:
                f.write(f"[ASYNC_WORKER] Completed processing for {video_id}\n")
                f.flush()
            print(f"[ASYNC_WORKER] Completed processing for {video_id}", flush=True)
        except Exception as e:
            error_msg = f"[ASYNC_WORKER] ERROR processing {video_id}: {str(e)}\n"
            with open(log_file, 'a') as f:
                f.write(error_msg)
                import traceback
                tb = traceback.format_exc()
                f.write(tb + "\n")
                f.flush()
            
            print(error_msg, flush=True)
            import traceback
            traceback.print_exc()
            # Store error result in cache
            processor.results_cache[video_id] = {
                'video_id': video_id,
                'status': 'error',
                'error': str(e)
            }
    
    thread = Thread(target=_safe_process)
    thread.daemon = False  # Changed to non-daemon so it doesn't get killed
    thread.start()
    print(f"[ASYNC] Background thread started for {video_id}", flush=True)
    """


if __name__ == "__main__":
    # Test
    video_dir = Path(__file__).parent / "videos"
    videos = processor.list_videos(video_dir)
    print(f"Found {len(videos)} videos:")
    for v in videos:
        print(f"  - {v['name']} ({v['size_mb']:.1f}MB)")
