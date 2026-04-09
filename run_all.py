from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
UI_DIR = ROOT / "ui-layer"
PERCEPTION_DIR = ROOT / "perception-layer" / "security-perception-layer"
VIDEOS_DIR = PERCEPTION_DIR / "videos"
UI_PYTHON = UI_DIR / ".venv" / "Scripts" / "python.exe"
PERCEPTION_PYTHON = PERCEPTION_DIR / ".venv" / "Scripts" / "python.exe"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run NAISC UI + perception + embedded reasoning from one Python file."
    )
    parser.add_argument("video", help="Absolute or relative path to the input video file.")
    parser.add_argument(
        "--ui-port",
        type=int,
        default=8010,
        help="Port for the UI alert service. Default: 8010",
    )
    parser.add_argument(
        "--keep-ui-running",
        action="store_true",
        help="Leave the UI service running after the script completes.",
    )
    return parser.parse_args()


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def wait_for_http(url: str, timeout_seconds: float = 20.0) -> dict[str, Any]:
    deadline = time.time() + timeout_seconds
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:  # pragma: no cover - environment-dependent
            last_error = exc
            time.sleep(1)
    raise RuntimeError(f"Timed out waiting for {url}: {last_error}")


def post_json(url: str, payload: dict[str, Any], timeout: int = 60) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def start_ui_service(port: int) -> subprocess.Popen[str]:
    require_file(UI_PYTHON, "UI Python executable")
    process = subprocess.Popen(
        [str(UI_PYTHON), "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(port)],
        cwd=UI_DIR,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    wait_for_http(f"http://127.0.0.1:{port}/telegram/health")
    return process


def copy_video(source: Path) -> Path:
    require_file(source, "Input video")
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    destination = VIDEOS_DIR / source.name
    shutil.copy2(source, destination)
    return destination


def build_video_id(video_path: Path) -> str:
    safe_chars = []
    for char in video_path.stem.lower():
        safe_chars.append(char if char.isalnum() else "_")
    return "".join(safe_chars).strip("_") or "processed_video"


def run_video_processing(video_path: Path) -> Path:
    require_file(PERCEPTION_PYTHON, "Perception Python executable")
    env = dict(os.environ)
    env["VIDEO_REASONING_PROVIDER"] = "cloud_agent"
    video_id = build_video_id(video_path)
    output_path = PERCEPTION_DIR / f"result_{video_id}.json"
    inline_code = """
import json
from pathlib import Path
from video_processor import processor

video_path = Path(sys.argv[1]).resolve()
video_id = sys.argv[2]
output_path = Path(sys.argv[3]).resolve()
result = processor.process_video(str(video_path), video_id, fps_sample=2.0, force_reprocess=True)
output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
"""
    completed = subprocess.run(
        [str(PERCEPTION_PYTHON), "-c", f"import sys\n{inline_code}", str(video_path), video_id, str(output_path)],
        cwd=PERCEPTION_DIR,
        env=env,
        text=True,
    )
    if completed.returncode not in (0, 2):
        raise RuntimeError(f"Video processing failed with exit code {completed.returncode}")
    return output_path


def load_result(result_path: Path) -> dict[str, Any]:
    require_file(result_path, "Processing result")
    return json.loads(result_path.read_text(encoding="utf-8"))


def choose_best_detection(result: dict[str, Any]) -> dict[str, Any]:
    detections = result.get("detections", [])
    if not detections:
        raise RuntimeError("No detections found in result file.")

    priority_order = {"critical": 3, "high": 2, "medium": 1, "low": 0}
    best = max(
        detections,
        key=lambda item: (
            priority_order.get(item.get("reasoning", {}).get("threat_level", "low"), -1),
            float(item.get("reasoning", {}).get("threat_score", 0.0) or 0.0),
            float(item.get("reasoning", {}).get("confidence", 0.0) or 0.0),
        ),
    )
    return best


def build_alert_payload(result: dict[str, Any], detection: dict[str, Any], video_path: Path) -> dict[str, Any]:
    reasoning = detection["reasoning"]
    perception = detection["perception"]
    anomalies = list(reasoning.get("anomalies", []))
    return {
        "source_id": result.get("video_id", "processed-video"),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "threat_level": reasoning["threat_level"],
        "confidence": float(reasoning.get("confidence", 0.0) or 0.0),
        "recommended_action": {
            "action": reasoning["action"],
            "priority": reasoning["priority"],
            "reason": reasoning["reason"],
            "confidence": float(reasoning.get("confidence", 0.0) or 0.0),
        },
        "explanation": {
            "summary": reasoning["summary"],
            "key_factors": list(reasoning.get("key_factors", [])),
            "evidence": {
                "weapon_evidence": [f"weapon={perception.get('weapon_detected', 'unknown')}"],
                "emotion_evidence": [f"emotion={perception.get('emotion', 'unknown')}"],
                "audio_evidence": [f"tone={perception.get('tone', 'unknown')}"],
                "risk_indicators": anomalies,
            },
            "anomalies_detected": anomalies,
            "temporal_analysis": "Selected from processed video frames.",
            "confidence_reasoning": "Payload derived from the highest-priority processed frame.",
        },
        "anomaly_types": anomalies,
        "metrics": {
            "weapon_threat_score": float(reasoning["metrics"].get("weapon_score", 0.0) or 0.0),
            "emotion_threat_score": float(reasoning["metrics"].get("emotion_score", 0.0) or 0.0),
            "audio_threat_score": float(reasoning["metrics"].get("audio_score", 0.0) or 0.0),
            "behavioral_anomaly_score": float(reasoning["metrics"].get("behavioral_score", 0.0) or 0.0),
            "combined_threat_score": float(reasoning.get("threat_score", 0.0) or 0.0),
            "trend": "stable",
            "frames_in_history": 1,
            "context_anomaly_flag": False,
        },
        "reasoning_version": "video_frame_heuristic_v1",
        "location": f"Processed video: {video_path.name}",
        "anomaly_type": "weapon_detected",
        "top_scenario": "Armed incident detected in processed video",
        "source": "run_all.py",
        "video_path": str(video_path.relative_to(ROOT)).replace("\\", "/"),
        "video_caption": f"NAISC processed alert for {video_path.name}",
    }


def main() -> int:
    args = parse_args()
    source_video = Path(args.video).expanduser().resolve()

    require_file(UI_PYTHON, "UI Python executable")
    require_file(PERCEPTION_PYTHON, "Perception Python executable")

    ui_process: subprocess.Popen[str] | None = None
    try:
        print(f"[1/4] Starting UI layer on port {args.ui_port}...")
        ui_process = start_ui_service(args.ui_port)
        print("[2/4] Copying input video into perception-layer/videos...")
        copied_video = copy_video(source_video)
        print("[3/4] Running perception + embedded reasoning on the video...")
        result_path = run_video_processing(copied_video)
        print("[4/4] Loading result and sending best incident to the UI layer...")
        result = load_result(result_path)
        best_detection = choose_best_detection(result)
        payload = build_alert_payload(result, best_detection, copied_video)
        response = post_json(f"http://127.0.0.1:{args.ui_port}/telegram/alert", payload)

        print("")
        print("Run complete")
        print(f"Video: {copied_video}")
        print(f"Processed frames: {result.get('processed_frames')}/{result.get('total_frames')}")
        print(f"Best frame: {best_detection.get('frame_number')} @ {best_detection.get('timestamp_sec')}s")
        print(f"Threat: {best_detection['reasoning'].get('threat_level')}")
        print(f"Action: {best_detection['reasoning'].get('action')}")
        print(f"Telegram status: {response.get('status')}")
        print(f"Telegram detail: {response.get('detail')}")
        return 0
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        print(f"HTTP error: {exc.code} {body}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        if ui_process is not None and not args.keep_ui_running:
            ui_process.terminate()
            try:
                ui_process.wait(timeout=10)
            except subprocess.TimeoutExpired:  # pragma: no cover - environment-dependent
                ui_process.kill()


if __name__ == "__main__":
    raise SystemExit(main())
