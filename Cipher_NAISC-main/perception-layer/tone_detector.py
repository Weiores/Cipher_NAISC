"""
Tone detector for Cipher_NAISC — Whisper (video) / emotion proxy (webcam).

Webcam source  → tone derived from emotion label; no audio capture needed.
Video file     → ffmpeg extracts WAV → Whisper tiny transcribes → keyword
                 classification (aggressive / tense / neutral).
                 Falls back to librosa heuristic if Whisper is unavailable.
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ToneDetectionResult:
    """Audio / tone analysis result — compatible with the rest of the pipeline."""

    tone: str            # calm | neutral | tense | aggressive | panic | threat | unknown
    confidence: float
    speech_present: bool = False
    skipped: bool = False
    acoustic_events: list[str] = field(default_factory=list)
    transcript: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ------------------------------------------------------------------
# Emotion → tone proxy table (used when audio is unavailable)
# ------------------------------------------------------------------

_EMOTION_TONE: dict[str, tuple[str, float]] = {
    "angry":      ("aggressive", 0.75),
    "fearful":    ("tense",      0.65),
    "distressed": ("tense",      0.65),
    "neutral":    ("neutral",    0.80),
    "calm":       ("calm",       0.85),
    "unknown":    ("unknown",    0.00),
}

_AGGRESSIVE_WORDS = frozenset({
    "kill", "die", "fight", "attack", "shoot", "weapon", "gun", "knife",
    "threat", "hurt", "destroy", "murder", "bomb", "hate", "blood", "stab",
})
_TENSE_WORDS = frozenset({
    "help", "please", "stop", "no", "leave", "scared", "afraid",
    "danger", "emergency", "run", "escape", "call", "police",
})


def _is_webcam(source: Any) -> bool:
    """Return True when *source* is a webcam index (integer or digit string)."""
    if isinstance(source, int):
        return True
    return str(source).strip().lstrip("-").isdigit()


class ToneDetector:
    """Tone analysis: Whisper for video files, emotion proxy for webcam."""

    def __init__(self, video_source: Any = None) -> None:
        src = video_source if video_source is not None else os.getenv("VIDEO_SOURCE", "0")
        self._is_webcam: bool = _is_webcam(src)
        self._whisper = None
        self._librosa_available = False
        self._ffmpeg_ok = self._check_ffmpeg()

        if self._is_webcam:
            logger.info("[TONE] Webcam source — using emotion proxy for tone")
            return

        # Video-file path: try Whisper first, then librosa
        try:
            import whisper as _w
            self._whisper = _w.load_model("tiny")
            logger.info("[TONE] Whisper tiny model loaded")
        except Exception as exc:
            logger.warning(f"[TONE] Whisper unavailable: {exc}")

        if self._whisper is None:
            try:
                import librosa  # noqa: F401
                self._librosa_available = True
                logger.info("[TONE] librosa fallback available")
            except ImportError:
                logger.warning("[TONE] Neither Whisper nor librosa available")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self,
        frame: np.ndarray | None = None,
        emotion_hint: str | None = None,
        audio_segment: str | None = None,
    ) -> ToneDetectionResult:
        """Detect tone for the current frame.

        Webcam / no Whisper  → emotion proxy.
        Video + audio_segment path provided  → Whisper transcription.
        Otherwise  → emotion proxy.
        """
        if self._is_webcam or self._whisper is None:
            return self._from_emotion(emotion_hint)

        if audio_segment is not None:
            return self._whisper_analyse(audio_segment)

        return self._from_emotion(emotion_hint)

    def detect_placeholder(self) -> ToneDetectionResult:
        """Safe neutral default (webcam / audio unavailable)."""
        return ToneDetectionResult(tone="neutral", confidence=0.0, skipped=True)

    def detect_from_file(self, video_path: str | Path) -> ToneDetectionResult:
        """Analyse tone from a full video file (used for pre-computation).

        Extracts audio via ffmpeg, then runs Whisper or librosa.
        """
        if _is_webcam(video_path):
            return self.detect_placeholder()

        vp = Path(video_path)
        if not vp.exists():
            logger.warning(f"[TONE] Video not found: {vp}")
            return self.detect_placeholder()

        wav_path = self._extract_audio(vp)
        if wav_path is None:
            return self.detect_placeholder()

        try:
            if self._whisper is not None:
                return self._whisper_analyse(wav_path)
            if self._librosa_available:
                return self._librosa_analyse(wav_path)
        finally:
            try:
                Path(wav_path).unlink(missing_ok=True)
            except Exception:
                pass

        return self.detect_placeholder()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _from_emotion(self, emotion_hint: str | None) -> ToneDetectionResult:
        label, conf = _EMOTION_TONE.get(emotion_hint or "unknown", ("unknown", 0.0))
        logger.info(f"[TONE] Emotion proxy: {emotion_hint} → {label} ({conf:.0%})")
        return ToneDetectionResult(tone=label, confidence=conf)

    def _whisper_analyse(self, audio_path: str) -> ToneDetectionResult:
        try:
            result = self._whisper.transcribe(audio_path, fp16=False)
            transcript = result.get("text", "").lower().strip()
            logger.info(f"[TONE] Whisper transcript: '{transcript[:120]}'")

            words = set(transcript.split())
            if words & _AGGRESSIVE_WORDS:
                return ToneDetectionResult(
                    tone="aggressive",
                    confidence=0.85,
                    speech_present=True,
                    acoustic_events=["aggressive_language"],
                    transcript=transcript,
                )
            if words & _TENSE_WORDS:
                return ToneDetectionResult(
                    tone="tense",
                    confidence=0.70,
                    speech_present=True,
                    acoustic_events=["distress_language"],
                    transcript=transcript,
                )
            return ToneDetectionResult(
                tone="neutral",
                confidence=0.60,
                speech_present=bool(transcript),
                transcript=transcript,
            )
        except Exception as exc:
            logger.warning(f"[TONE] Whisper transcription failed: {exc}")
            return ToneDetectionResult(tone="unknown", confidence=0.0)

    def _librosa_analyse(self, wav_path: str) -> ToneDetectionResult:
        """librosa heuristic — fallback when Whisper is not installed."""
        try:
            import librosa

            y, sr = librosa.load(wav_path, sr=None, mono=True, duration=60)
            if len(y) == 0:
                return ToneDetectionResult(tone="unknown", confidence=0.0)

            rms = float(np.sqrt(np.mean(y ** 2)))
            zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
            sc = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
            speech_present = rms > 0.01
            events: list[str] = []

            if rms > 0.15 and zcr > 0.15:
                tone, conf = "panic", 0.75
                events.append("high_energy_speech")
            elif sc > 3000 and rms > 0.08:
                tone, conf = "threat", 0.70
                events.append("elevated_frequency")
            elif rms > 0.06:
                tone, conf = "abnormal", 0.65
                events.append("moderate_energy")
            elif rms < 0.005:
                tone, conf = "calm", 0.80
            else:
                tone, conf = "calm", 0.65

            logger.debug(f"[TONE] librosa: rms={rms:.4f} zcr={zcr:.4f} sc={sc:.0f} → {tone} ({conf:.2f})")
            return ToneDetectionResult(
                tone=tone,
                confidence=round(conf, 4),
                speech_present=speech_present,
                acoustic_events=events,
            )
        except Exception as exc:
            logger.warning(f"[TONE] librosa analysis failed: {exc}")
            return ToneDetectionResult(tone="unknown", confidence=0.0)

    def _check_ffmpeg(self) -> bool:
        try:
            r = subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=5)
            return r.returncode == 0
        except Exception:
            return False

    def _extract_audio(self, video_path: Path) -> str | None:
        """Extract 16 kHz mono WAV from video via ffmpeg; return temp path or None."""
        try:
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.close()
            cmd = [
                "ffmpeg", "-y", "-i", str(video_path),
                "-vn", "-acodec", "pcm_s16le",
                "-ar", "16000", "-ac", "1",
                tmp.name,
            ]
            result = subprocess.run(cmd, capture_output=True, timeout=60)
            if result.returncode == 0 and Path(tmp.name).stat().st_size > 0:
                return tmp.name
            Path(tmp.name).unlink(missing_ok=True)
            return None
        except Exception as exc:
            logger.debug(f"[TONE] Audio extraction failed: {exc}")
            return None


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    source = sys.argv[1] if len(sys.argv) > 1 else "0"
    detector = ToneDetector(video_source=source)
    if len(sys.argv) > 1 and not source.isdigit():
        print(detector.detect_from_file(source))
    else:
        print(detector.detect(emotion_hint="angry"))
