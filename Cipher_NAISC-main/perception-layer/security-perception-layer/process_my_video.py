
import sys
import os
from pathlib import Path
import json
import asyncio

# Add paths for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir.parent.parent / "reasoning-layer"))

# Set environment variable to use rule-based reasoning (fastest, no setup)
os.environ["VIDEO_REASONING_PROVIDER"] = "cloud_agent"

from video_processor import processor

def main():
    video_name = "YTDown.com_YouTube_Florida-man-attacks-driver-with-knife-du_Media_VsO6F2BBu-E_001_720p.mp4"
    video_path = current_dir / "videos" / video_name
    video_id = "florida_man_knife"
    
    print(f"Processing video: {video_path}")
    if not video_path.exists():
        print(f"Error: Video file not found at {video_path}")
        return

    # Process the video (sample every 2 seconds)
    print("Starting processing... this may take a minute.")
    # processor.process_video is sync but calls async reasoning internally via a loop
    result = processor.process_video(str(video_path), video_id, fps_sample=2.0, force_reprocess=True)
    
    # Save results
    output_path = current_dir / f"result_{video_id}.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    
    if result['status'] == 'error':
        print(f"Error during processing: {result.get('error')}")
        return

    print(f"Processing complete! Status: {result['status']}")
    print(f"Processed {result['processed_frames']} frames.")
    print(f"Results saved to {output_path}")
    
    # Print a summary of detections
    detections = result.get("detections", [])
    print(f"\nSummary of detections ({len(detections)} frames):")
    for d in detections:
        frame_num = d["frame_number"]
        timestamp = d["timestamp_sec"]
        threat = d["reasoning"]["threat_level"]
        action = d["reasoning"]["action"]
        perception_data = d.get("perception", {})
        weapon = perception_data.get("weapon_detected", "unknown")
        print(f"  Frame {frame_num:4} ({timestamp:4.1f}s): Threat={threat:8} Action={action:15} Weapon={weapon}")

if __name__ == "__main__":
    main()
