import os
import shutil
import subprocess
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from scipy.io import wavfile

from app.schemas import AudioDetection, PerceptionRequest
from app.services.models.base import ModelAdapter

try:
    import whisper
except ImportError:  # pragma: no cover - optional runtime dependency
    whisper = None


class AudioThreatDetectionAdapter(ModelAdapter):
    name = "audio_threat_detection"
    intended_backend = "VAD plus speech emotion plus transcription plus acoustic event classifier for bodycam audio"
    frame_ms = 30
    hop_ms = 15

    def __init__(self) -> None:
        self.temp_dir = Path(__file__).resolve().parents[3] / ".tmp_audio"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.whisper_model_name = os.getenv("WHISPER_MODEL_NAME", "tiny.en")
        self.whisper_model = None

    def describe(self) -> dict[str, str]:
        details = super().describe()
        details["active_backend"] = "waveform_heuristics_plus_whisper" if whisper is not None else "waveform_heuristics"
        details["whisper_model_name"] = self.whisper_model_name if whisper is not None else ""
        return details

    async def infer(self, request: PerceptionRequest) -> AudioDetection:
        audio_source = self._resolve_audio_source(request)
        if audio_source is None:
            return AudioDetection(
                tone="unknown",
                confidence=0.0,
                speech_present=False,
            )

        analysis = self._analyze_audio(audio_source)
        if analysis is not None:
            return analysis

        return AudioDetection(
            tone="unknown",
            confidence=0.0,
            speech_present=False,
        )

    def _resolve_audio_source(self, request: PerceptionRequest) -> Path | None:
        if request.audio and request.audio.uri:
            source = Path(request.audio.uri)
            if source.exists():
                return source

        if request.video and request.video.uri:
            source = Path(request.video.uri)
            if source.exists():
                return source

        return None

    def _analyze_audio(self, source: Path) -> AudioDetection | None:
        wav_path = self._extract_audio_to_wav(source)
        if wav_path is None:
            return None

        transcript = self._transcribe_audio(wav_path)

        try:
            sample_rate, audio = wavfile.read(str(wav_path))
        except Exception:
            wav_path.unlink(missing_ok=True)
            return None

        wav_path.unlink(missing_ok=True)

        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32)
        max_val = float(np.max(np.abs(audio))) if audio.size else 0.0
        if max_val > 0:
            audio = audio / max_val
        if audio.size == 0:
            return AudioDetection(tone="unknown", confidence=0.0, speech_present=False)

        rms, zcr = self._frame_features(audio, sample_rate)
        if rms.size == 0:
            return AudioDetection(tone="unknown", confidence=0.0, speech_present=False)

        median_rms = float(np.median(rms))
        p95_rms = float(np.percentile(rms, 95))
        mean_zcr = float(np.mean(zcr))
        peak = float(np.max(np.abs(audio)))
        speech_ratio = float(np.mean(rms > max(0.08, median_rms * 2.5)))
        burst_ratio = float(np.mean(rms > max(0.18, median_rms * 4.0)))

        tone, confidence = self._classify_tone(
            speech_ratio=speech_ratio,
            burst_ratio=burst_ratio,
            p95_rms=p95_rms,
            mean_zcr=mean_zcr,
            peak=peak,
        )
        acoustic_events = self._detect_events(
            speech_ratio=speech_ratio,
            burst_ratio=burst_ratio,
            p95_rms=p95_rms,
            peak=peak,
        )
        keyword_flags = self._extract_keyword_flags(transcript)
        if "threat_language" in keyword_flags and tone == "calm":
            tone, confidence = "threat", 0.8
        elif "distress_language" in keyword_flags and tone in {"calm", "unknown"}:
            tone, confidence = "panic", 0.8

        return AudioDetection(
            tone=tone,
            confidence=confidence,
            speech_present=speech_ratio > 0.05,
            acoustic_events=acoustic_events,
            transcript=transcript,
            keyword_flags=keyword_flags,
        )

    def _extract_audio_to_wav(self, source: Path) -> Path | None:
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        output = self.temp_dir / f"{source.stem}_decoded.wav"
        command = [
            ffmpeg_exe,
            "-y",
            "-i",
            str(source),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            str(output),
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0 or not output.exists():
            output.unlink(missing_ok=True)
            return None
        return output

    def _transcribe_audio(self, wav_path: Path) -> str | None:
        model = self._get_whisper_model()
        if model is None:
            return None

        ffmpeg_dir = str(self._ensure_ffmpeg_alias_dir())
        original_path = os.environ.get("PATH", "")
        if ffmpeg_dir not in original_path:
            os.environ["PATH"] = ffmpeg_dir + os.pathsep + original_path

        try:
            result = model.transcribe(
                str(wav_path),
                language="en",
                fp16=False,
                verbose=False,
            )
        except Exception:
            return None

        text = str(result.get("text", "")).strip()
        return text or None

    def _ensure_ffmpeg_alias_dir(self) -> Path:
        source_exe = Path(imageio_ffmpeg.get_ffmpeg_exe())
        alias_dir = self.temp_dir / "ffmpeg"
        alias_dir.mkdir(parents=True, exist_ok=True)
        alias_exe = alias_dir / "ffmpeg.exe"
        if not alias_exe.exists():
            shutil.copyfile(source_exe, alias_exe)
        return alias_dir

    def _get_whisper_model(self):
        if whisper is None:
            return None
        if self.whisper_model is not None:
            return self.whisper_model

        download_root = self.temp_dir / "whisper_models"
        download_root.mkdir(parents=True, exist_ok=True)
        try:
            self.whisper_model = whisper.load_model(self.whisper_model_name, download_root=str(download_root))
        except Exception:
            return None
        return self.whisper_model

    def _frame_features(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
        frame_len = max(1, int(sample_rate * self.frame_ms / 1000))
        hop_len = max(1, int(sample_rate * self.hop_ms / 1000))
        if audio.size < frame_len:
            padded = np.pad(audio, (0, frame_len - audio.size))
            frames = padded.reshape(1, -1)
        else:
            frame_count = 1 + (audio.size - frame_len) // hop_len
            indices = np.arange(frame_len)[None, :] + hop_len * np.arange(frame_count)[:, None]
            frames = audio[indices]

        rms = np.sqrt(np.mean(np.square(frames), axis=1))
        signs = np.sign(frames)
        signs[signs == 0] = 1
        zcr = np.mean(signs[:, 1:] != signs[:, :-1], axis=1)
        return rms, zcr

    def _classify_tone(
        self,
        *,
        speech_ratio: float,
        burst_ratio: float,
        p95_rms: float,
        mean_zcr: float,
        peak: float,
    ) -> tuple[str, float]:
        if speech_ratio < 0.02 and peak < 0.12:
            return "unknown", 0.0
        if peak > 0.95 and burst_ratio > 0.12:
            return "panic", 0.84
        if speech_ratio > 0.18 and p95_rms > 0.22 and mean_zcr > 0.12:
            return "threat", 0.79
        if burst_ratio > 0.08 or peak > 0.8:
            return "abnormal", 0.68
        return "calm", 0.62

    def _detect_events(
        self,
        *,
        speech_ratio: float,
        burst_ratio: float,
        p95_rms: float,
        peak: float,
    ) -> list[str]:
        events: list[str] = []
        if speech_ratio > 0.05:
            events.append("speech_like_activity")
        if p95_rms > 0.2:
            events.append("high_intensity_audio")
        if peak > 0.95:
            events.append("sharp_impulse")
        if burst_ratio > 0.1:
            events.append("repeated_bursts")
        return events

    def _extract_keyword_flags(self, transcript: str | None) -> list[str]:
        if not transcript:
            return []

        text = transcript.lower()
        flags: list[str] = []
        threat_terms = ["drop it", "shoot", "gun", "knife", "weapon", "get down", "don't move"]
        distress_terms = ["help", "please", "stop", "no", "call", "officer"]

        if any(term in text for term in threat_terms):
            flags.append("threat_language")
        if any(term in text for term in distress_terms):
            flags.append("distress_language")
        return flags
