"""
Video processor for Cipher_NAISC.

Accepts a video file path or an RTSP/webcam stream URL, decodes frames, and
passes them to the perception layer.  Returns structured detection results
per frame: { timestamp, frame_id, detections }.

CLI usage:
  python video_processor.py --source <path_or_url> [--fps 2] [--max-frames 0]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level streaming control — camera hardware is NOT opened until
# enable_streaming() is explicitly called.
# ---------------------------------------------------------------------------

_frame_lock = threading.Lock()
_latest_frame_jpeg: bytes | None = None

# Set to True the first time _process_frame() saves an annotated JPEG.
# The background capture loop only writes raw frames while this is False
# (i.e. before perception has started), so annotated frames are never overwritten.
_has_annotated_frame: bool = False

# The live camera capture object.  None means the camera is closed.
_cap: cv2.VideoCapture | None = None

# Guards _cap so enable/disable calls from different threads are safe.
_cap_lock = threading.Lock()

# Serialises frame reads — prevents the background capture thread and the
# VideoProcessor.stream() loop from calling cap.read() concurrently.
# Also held by disable_streaming/switch_source before releasing the cap, so
# an in-progress read always completes before the capture object is destroyed.
_cap_read_lock = threading.Lock()

# Streaming gate — the stream() loop reads this before touching _cap.
_streaming_enabled: bool = False

# True when VIDEO_SOURCE is a file path (not a webcam index or RTSP URL).
# File sources loop automatically and skip the enable_streaming() privacy gate.
_source_is_file: bool = False

logger.info("[PROCESSOR] System ready - camera standby")


# ---------------------------------------------------------------------------
# Background frame-capture daemon
# ---------------------------------------------------------------------------

def _frame_capture_loop() -> None:
    """Daemon: continuously read frames from _cap and store the latest as JPEG.

    This is what makes /stream/frame serve live video even when the full
    VideoProcessor perception pipeline (from main.py) is not running.
    Handles file looping automatically.  Holds _cap_read_lock only for the
    duration of a single cap.read() call — never while sleeping.
    """
    while True:
        if not _streaming_enabled or _cap is None:
            time.sleep(0.04)
            continue

        ret, frame, cap_local = False, None, None

        with _cap_read_lock:
            cap_local = _cap
            if cap_local is not None:
                ret, frame = cap_local.read()
                if not ret and _source_is_file:
                    cap_local.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    logger.debug("[CAPTURE] Video file looped")

        if ret and frame is not None:
            # Only write raw frames before perception has saved its first annotated
            # frame.  Once _has_annotated_frame is True this branch never runs again,
            # so the capture loop can never overwrite a frame that has bounding boxes.
            if not _has_annotated_frame:
                _store_frame_jpeg(frame)
        else:
            time.sleep(0.04)


_capture_thread = threading.Thread(
    target=_frame_capture_loop, daemon=True, name="frame-capture"
)
_capture_thread.start()
logger.info("[PROCESSOR] Background frame capture thread started")


def enable_streaming(source: str | int | None = None) -> None:
    """Open the physical camera (or video file) and start accepting frame reads.

    For webcam sources this is gated behind user consent (the dashboard
    "Activate" button).  For file sources main.py calls this automatically
    at startup — no privacy concern for pre-recorded footage.

    Safe to call from any thread.
    """
    global _cap, _streaming_enabled, _source_is_file

    if source is None:
        source = os.getenv("VIDEO_SOURCE", "0")

    # Determine whether this is a file or a live camera
    src_str = str(source).strip()
    is_file = (
        src_str != ""
        and not src_str.isdigit()
        and not src_str.lower().startswith("rtsp")
    )
    _source_is_file = is_file

    if isinstance(source, str) and source.isdigit():
        source = int(source)

    with _cap_lock:
        if _cap is not None:
            _streaming_enabled = True
            logger.info("[PROCESSOR] Streaming re-enabled on existing capture")
            return

        logger.info(f"[PROCESSOR] Opening source: {source} (file={is_file})")

        # CAP_DSHOW is needed on Windows for webcams; files open fine without it
        if sys.platform == "win32" and not is_file:
            cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            logger.error(f"[PROCESSOR] Failed to open source: {source}")
            return

        if not is_file:
            # Allow the camera sensor to stabilise before the first read
            logger.info("[PROCESSOR] Camera warming up (2 s)…")
            time.sleep(2)

        _cap = cap
        _streaming_enabled = True
        if is_file:
            logger.info("[PROCESSOR] Video file opened - streaming enabled (demo mode)")
        else:
            logger.info("[PROCESSOR] Camera opened - streaming enabled")


def disable_streaming() -> None:
    """Release the physical camera and clear the frame buffer.

    After this call the webcam light goes off.  Safe to call from any thread.
    """
    global _cap, _streaming_enabled, _latest_frame_jpeg

    _streaming_enabled = False

    # Acquire _cap_read_lock first so we wait for any in-progress cap.read()
    # to finish before we release the capture object.
    with _cap_read_lock:
        with _cap_lock:
            if _cap is not None:
                _cap.release()
                _cap = None

    with _frame_lock:
        _latest_frame_jpeg = None

    logger.info("[PROCESSOR] Camera released - streaming disabled")


def switch_source(new_source: str | int | None) -> None:
    """Hot-swap the video source without restarting the system.

    Releases the current capture, then opens the new source if provided.
    Passing None stops the feed and returns to standby.
    Safe to call from any thread.
    """
    global _cap, _streaming_enabled, _source_is_file, _latest_frame_jpeg

    # Stop and release whatever is currently open; wait for any in-progress read.
    _streaming_enabled = False
    with _cap_read_lock:
        with _cap_lock:
            if _cap is not None:
                _cap.release()
                _cap = None
    with _frame_lock:
        _latest_frame_jpeg = None

    if new_source is None:
        _source_is_file = False
        logger.info("[PROCESSOR] Feed stopped — returning to standby")
        return

    src_str = str(new_source).strip()
    is_file = src_str != "" and not src_str.isdigit() and not src_str.lower().startswith("rtsp")
    _source_is_file = is_file

    src: str | int = int(src_str) if src_str.isdigit() else src_str

    logger.info(f"[PROCESSOR] Switching to source: {src} (file={is_file})")

    if sys.platform == "win32" and not is_file:
        cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(src)

    if not cap.isOpened():
        logger.error(f"[PROCESSOR] switch_source: failed to open {src}")
        return

    if not is_file:
        logger.info("[PROCESSOR] Camera warming up (2 s)…")
        time.sleep(2)

    with _cap_lock:
        _cap = cap
    _streaming_enabled = True
    logger.info(f"[PROCESSOR] Switched to source: {new_source}")


def get_latest_frame_jpeg() -> bytes | None:
    """Return the most recent annotated frame as JPEG bytes, or None."""
    with _frame_lock:
        return _latest_frame_jpeg


def _store_frame_jpeg(frame_bgr: np.ndarray) -> None:
    global _latest_frame_jpeg
    ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if ok:
        with _frame_lock:
            _latest_frame_jpeg = buf.tobytes()
        logger.info(f"[DEBUG] JPEG saved, size={len(_latest_frame_jpeg)} bytes")


def _annotate_frame(frame: np.ndarray, detection: dict[str, Any]) -> np.ndarray:
    """Draw bounding boxes and labels on a copy of the frame."""
    out = frame.copy()
    h, w = out.shape[:2]

    weapon = detection.get("weapon", {})
    bbox = weapon.get("bbox") or []
    label = weapon.get("label", "")
    conf = weapon.get("confidence", 0.0)

    is_weapon = label not in {"", "unarmed", "unknown", "unknown_object"} and conf > 0.1

    if bbox and len(bbox) >= 4:
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        color = (0, 0, 255) if is_weapon else (0, 255, 0)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        txt = f"{label} {conf:.0%}"
        cv2.rectangle(out, (x1, max(y1 - 16, 0)), (x1 + len(txt) * 8, y1), color, -1)
        cv2.putText(out, txt, (x1, max(y1 - 3, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    emotion = detection.get("emotion", {})
    e_label = emotion.get("label", "")
    if e_label and e_label not in {"neutral", "unknown"}:
        color = (0, 165, 255)  # orange for threatening emotion
        cv2.putText(out, f"Emotion: {e_label}", (6, h - 38), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    tone = detection.get("tone", {})
    t_label = tone.get("label", "")
    if t_label:
        cv2.putText(out, f"Tone: {t_label}", (6, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    # Timestamp overlay
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
    cv2.putText(out, ts, (w - 90, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)

    return out

# Add project paths so perception-layer modules are importable
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "perception-layer"))

# ---------------------------------------------------------------------------
# .env loader
# ---------------------------------------------------------------------------

def _load_env() -> None:
    env_path = _ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_env()


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

def _make_frame_result(
    frame_id: int,
    timestamp: str,
    detections: list[dict[str, Any]],
    is_danger: bool = False,
    danger_reasons: list[str] | None = None,
) -> dict[str, Any]:
    """Build a standardised per-frame result dict."""
    return {
        "frame_id": frame_id,
        "timestamp": timestamp,
        "is_danger": is_danger,
        "danger_reasons": danger_reasons or [],
        "detections": detections,
    }


# ---------------------------------------------------------------------------
# Main processor
# ---------------------------------------------------------------------------

class VideoProcessor:
    """Decodes video frames and runs the perception pipeline.

    Args:
        source:          Path to a video file, integer webcam index, or RTSP URL.
        fps_sample:      Target frames-per-second to sample (< source FPS to skip frames).
        on_frame_result: Optional callback invoked for every processed frame result.
        on_danger:       Optional callback invoked only when danger is detected.
    """

    def __init__(
        self,
        source: str | int,
        fps_sample: float = 2.0,
        on_frame_result: Callable[[dict[str, Any]], None] | None = None,
        on_danger: Callable[[dict[str, Any], np.ndarray], None] | None = None,
    ) -> None:
        self.source = source
        self.fps_sample = fps_sample
        self.on_frame_result = on_frame_result
        self.on_danger = on_danger
        self._stop_requested = False

        # Lazy-load perception layer (may be unavailable in minimal installs)
        self._perception: Any = None
        self._load_perception()

    def _load_perception(self) -> None:
        try:
            from perception_layer import PerceptionLayer  # type: ignore[import]
            # Only pass video_path for real files — not for webcam indices ("0", "1", …)
            is_file = isinstance(self.source, str) and not self.source.isdigit()
            self._perception = PerceptionLayer(
                video_path=self.source if is_file else None
            )
            logger.info("[PROCESSOR] PerceptionLayer loaded")
        except ImportError as exc:
            logger.warning(f"[PROCESSOR] PerceptionLayer unavailable ({exc}); using placeholder")

    # ------------------------------------------------------------------

    def process(self, max_frames: int = 0) -> list[dict[str, Any]]:
        """Process video source synchronously and return all frame results.

        Args:
            max_frames: Stop after this many *sampled* frames. 0 = no limit.

        Returns:
            List of frame result dicts.
        """
        cap = self._open_capture()
        if cap is None:
            return []

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        sample_interval = max(1, int(fps / self.fps_sample))

        logger.info(
            f"[PROCESSOR] Source FPS={fps:.1f} sample_every={sample_interval} total_frames={total_frames}"
        )

        results: list[dict[str, Any]] = []
        frame_index = 0
        sampled_count = 0

        try:
            while not self._stop_requested:
                # Respect the global streaming flag — pause capture when disabled
                if not _streaming_enabled:
                    time.sleep(0.1)
                    continue

                ret, frame = cap.read()
                if not ret:
                    for _retry in range(5):
                        time.sleep(0.5)
                        ret, frame = cap.read()
                        if ret:
                            break
                if not ret:
                    logger.info(f"[PROCESSOR] End of stream at frame {frame_index}")
                    break

                if frame_index % sample_interval == 0:
                    result = self._process_frame(frame, sampled_count)
                    results.append(result)
                    sampled_count += 1

                    if self.on_frame_result:
                        self.on_frame_result(result)

                    if result["is_danger"] and self.on_danger:
                        self.on_danger(result, frame)

                    if max_frames and sampled_count >= max_frames:
                        logger.info(f"[PROCESSOR] Reached max_frames={max_frames}")
                        break

                frame_index += 1
        finally:
            cap.release()

        logger.info(f"[PROCESSOR] Done: {frame_index} source frames, {sampled_count} sampled")
        return results

    def stream(self, max_frames: int = 0):
        """Generator variant — yields frame results one by one.

        The camera is NOT opened here.  This method waits until
        enable_streaming() has been called (and _cap is set), then reads
        frames from the shared module-level capture object.  The camera
        is closed by disable_streaming(), not by this method.
        """
        frame_index = 0
        sampled_count = 0
        sample_interval = 1  # process every frame — no skipping
        streaming_logged = False

        logger.info("[PROCESSOR] Stream loop started — waiting for streaming to be enabled")

        while not self._stop_requested:
            # Block until the camera has been opened by enable_streaming()
            if not _streaming_enabled or _cap is None:
                time.sleep(0.1)
                continue

            logger.info(
                f"[PROCESSOR] Loop tick: streaming={_streaming_enabled} frame_count={frame_index}"
            )

            if not streaming_logged:
                actual_fps = _cap.get(cv2.CAP_PROP_FPS) or 30.0
                logger.info(f"[PROCESSOR] Streaming active: FPS={actual_fps:.1f} sample_every={sample_interval} frames")
                streaming_logged = True

            ret, frame, cap_local = False, None, None
            with _cap_read_lock:
                cap_local = _cap
                if cap_local is not None:
                    ret, frame = cap_local.read()
                    logger.info(f"[DEBUG] Frame read: ret={ret} shape={frame.shape if ret else 'None'}")
                    if not ret and _source_is_file:
                        cap_local.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        frame_index = 0
                        logger.info("[PROCESSOR] Video looped back to start")

            if not ret:
                if not (_source_is_file and cap_local is not None):
                    # Live camera or cap not open — wait before retrying
                    time.sleep(0.05)
                    streaming_logged = False  # re-log FPS after reconnect
                continue

            if frame_index % sample_interval == 0:
                logger.info(f"[PROCESSOR] Frame captured {sampled_count}")
                result = self._process_frame(frame, sampled_count)
                sampled_count += 1

                if self.on_frame_result:
                    self.on_frame_result(result)
                if result["is_danger"] and self.on_danger:
                    self.on_danger(result, frame)

                yield result

                if max_frames and sampled_count >= max_frames:
                    break

            frame_index += 1

    def stop(self) -> None:
        """Signal the processing loop to stop after the current frame."""
        self._stop_requested = True

    # ------------------------------------------------------------------

    def _open_capture(self) -> cv2.VideoCapture | None:
        source = self.source
        # Convert numeric string to int for webcam
        if isinstance(source, str) and source.isdigit():
            source = int(source)

        if sys.platform == "win32":
            cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            logger.error(f"[PROCESSOR] Cannot open source: {source}")
            return None

        # Windows cameras need time to initialise before the first frame is valid
        logger.debug("[PROCESSOR] Warming up camera for 2s…")
        time.sleep(2)
        return cap

    def _process_frame(self, frame: np.ndarray, frame_id: int) -> dict[str, Any]:
        global _has_annotated_frame
        ts = datetime.now(timezone.utc).isoformat()

        if self._perception is not None:
            try:
                logger.info(f"[PROCESSOR] Sending frame {frame_id} to perception layer")
                logger.info(f"[DEBUG] Sending frame to perception, id={frame_id} shape={frame.shape}")
                perception_result = self._perception.process_frame(frame, frame_id=frame_id)
                logger.info(f"[DEBUG] process_frame done, saving JPEG now")
                logger.info(
                    "[PROCESSOR] Perception result: danger=%s reasons=%s",
                    perception_result.is_danger,
                    perception_result.danger_reasons,
                )
                detections = [perception_result.to_dict()]
                # Save JPEG after perception has drawn weapon boxes on the frame.
                # weapon_detector._annotate_frame() modifies `frame` in-place during
                # process_frame(), so by here `frame` already carries the boxes.
                try:
                    annotated = _annotate_frame(frame, detections[0])
                    _store_frame_jpeg(annotated)
                except Exception as ann_exc:
                    logger.debug(f"[PROCESSOR] Frame annotation failed: {ann_exc}")
                    _store_frame_jpeg(frame)
                _has_annotated_frame = True
                logger.info(f"[DEBUG] JPEG saved, size={len(_latest_frame_jpeg) if _latest_frame_jpeg else 0} bytes")
                return _make_frame_result(
                    frame_id=frame_id,
                    timestamp=ts,
                    detections=detections,
                    is_danger=perception_result.is_danger,
                    danger_reasons=perception_result.danger_reasons,
                )
            except Exception as exc:
                logger.warning(f"[PROCESSOR] Perception error on frame {frame_id}: {exc}")

        # Placeholder result when perception layer is unavailable
        logger.warning(f"[PROCESSOR] Frame {frame_id}: perception layer unavailable, returning placeholder")
        _store_frame_jpeg(frame)
        return _make_frame_result(
            frame_id=frame_id,
            timestamp=ts,
            detections=[{"label": "unknown", "confidence": 0.0}],
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cipher_NAISC video processor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source", default=os.getenv("VIDEO_SOURCE", "0"),
                        help="Video file path, webcam index (0), or RTSP URL")
    parser.add_argument("--fps", type=float, default=2.0,
                        help="Sample rate in frames per second")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Stop after N sampled frames (0 = unlimited)")
    parser.add_argument("--output", default="",
                        help="Optional path to write results JSON")
    parser.add_argument("--log-level", default=os.getenv("LOG_LEVEL", "INFO"))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    def _on_danger(result: dict[str, Any], frame: np.ndarray) -> None:
        logger.warning(
            "[DANGER] frame_id=%d reasons=%s",
            result["frame_id"], result["danger_reasons"],
        )

    processor = VideoProcessor(
        source=args.source,
        fps_sample=args.fps,
        on_danger=_on_danger,
    )

    results = processor.process(max_frames=args.max_frames)
    logger.info(f"[CLI] Processed {len(results)} frames")

    if args.output:
        Path(args.output).write_text(
            json.dumps(results, indent=2), encoding="utf-8"
        )
        logger.info(f"[CLI] Results written to {args.output}")
    else:
        # Print summary to stdout
        danger_frames = [r for r in results if r["is_danger"]]
        print(f"\n=== Processing complete ===")
        print(f"Total sampled frames : {len(results)}")
        print(f"Danger frames        : {len(danger_frames)}")
        for r in danger_frames:
            print(f"  Frame {r['frame_id']:4d} @ {r['timestamp']} – {r['danger_reasons']}")


if __name__ == "__main__":
    main()
