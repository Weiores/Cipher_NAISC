
import json
import urllib.request
from pathlib import Path
import sys
import datetime

def main():
    current_dir = Path(__file__).parent
    results_path = current_dir / "perception-layer" / "security-perception-layer" / "result_florida_man_knife.json"
    
    if not results_path.exists():
        print(f"Error: Results file not found at {results_path}")
        return

    with open(results_path, "r") as f:
        data = json.load(f)

    # Find a critical detection
    critical_detection = next((d for d in data.get("detections", []) if d["reasoning"]["threat_level"] == "critical"), None)
    
    if not critical_detection:
        print("No critical detection found in the results.")
        # Fallback to the overall summary if no frame-level critical
        reasoning = data.get("summary", {}).get("cloud_reasoning")
        if not reasoning:
            print("No reasoning data found at all.")
            return
        
        reasoning_payload = {
            "source_id": data["video_id"],
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "threat_level": reasoning["threat_level"],
            "confidence": reasoning["confidence"],
            "recommended_action": {
                "action": reasoning["action"],
                "priority": reasoning["priority"],
                "reason": reasoning["reason"],
                "confidence": reasoning["confidence"]
            },
            "explanation": {
                "summary": reasoning["summary"],
                "key_factors": ["Video Summary Pass"],
                "evidence": {},
                "anomalies_detected": [],
                "confidence_reasoning": "Based on aggregate video analysis"
            },
            "metrics": {
                "weapon_threat_score": data["summary"]["weapon_frame_ratio"],
                "emotion_threat_score": 0.0,
                "audio_threat_score": 0.0,
                "behavioral_score": 0.0,
                "combined_threat_score": data["summary"]["overall_threat_score"],
                "trend": "stable",
                "frames_in_history": data["summary"]["total_detection_frames"]
            }
        }
    else:
        # Use the real ReasoningOutput from the critical frame
        reasoning_payload = critical_detection["reasoning"]
        # Ensure source_id is set
        if "source_id" not in reasoning_payload:
            reasoning_payload["source_id"] = f"video-{data['video_id']}"
        # Ensure timestamp is set
        reasoning_payload["timestamp"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        # Rename behavioral_score to behavioral_anomaly_score if needed by schema
        if "metrics" in reasoning_payload and "behavioral_score" in reasoning_payload["metrics"]:
             reasoning_payload["metrics"]["behavioral_anomaly_score"] = reasoning_payload["metrics"].pop("behavioral_score")

    # Send the ReasoningOutput directly as the payload (from_payload handles this)
    payload = reasoning_payload
    payload["location"] = "Pine Lakes Pkwy (Road Rage)"
    payload["anomaly_type"] = "weapon_detected"
    payload["top_scenario"] = "Armed Road Rage - Knife and Gun Brandished"
    payload["source"] = "Uploaded Video Analysis"
    payload["video_path"] = str(current_dir / "perception-layer" / "security-perception-layer" / "videos" / "YTDown.com_YouTube_Florida-man-attacks-driver-with-knife-du_Media_VsO6F2BBu-E_001_720p.mp4")
    payload["video_caption"] = "Florida man attacks driver with knife; driver pulls gun."

    url = "http://localhost:8010/telegram/alert"
    print(f"Sending alert to {url}...")
    
    try:
        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as response:
            res_data = response.read().decode("utf-8")
            if response.status == 200:
                print("Alert successfully sent to UI layer!")
                print(f"Response: {res_data}")
            else:
                print(f"Failed to send alert. Status code: {response.status}")
                print(f"Detail: {res_data}")
    except urllib.error.URLError as e:
        print("\nError: Could not connect to the UI layer server.")
        print("Please ensure the UI server is running by executing:")
        print("  cd NAISC/ui-layer")
        print("  .venv\\Scripts\\activate")
        print("  uvicorn app.main:app --reload --port 8010")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
