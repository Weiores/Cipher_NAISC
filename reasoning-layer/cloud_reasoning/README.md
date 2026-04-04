# Cloud Reasoning Agent

**Deployment Target:** Cloud Servers (AWS, Azure, On-Premise)

## Overview

Full-featured reasoning agent that receives perception data from officer devices and returns comprehensive threat assessments with detailed explanations.

**Best For:**
- Centralized decision-making
- Detailed analysis and explanations
- Integration with learning agent and scenario predictions
- Offline replay and investigation
- High accuracy requirements

## Key Characteristics

| Metric | Value |
|--------|-------|
| **Latency** | ~100-500ms (acceptable for cloud) |
| **Accuracy** | Highest (4-factor analysis) |
| **Resources** | High (full Python stack) |
| **Dependencies** | Learning agent, SOP database, scenario predictor |
| **Deployment** | Server/Container (AWS Lambda, Docker, VM) |
| **Offline** | ❌ Requires cloud server |

## Threat Scoring (4-Factor Weighted Analysis)

```
Combined Threat = (Weapon × 0.50)
                + (Emotion × 0.25)
                + (Audio × 0.20)
                + (Behavioral × 0.05)

Threat Levels:
  0.0 - 0.24   → LOW
  0.25 - 0.49  → MEDIUM
  0.50 - 0.74  → HIGH
  0.75 - 1.0   → CRITICAL
```

### Factor Scoring

**Weapon Threat (50% weight)**
- Gun/Rifle/Shotgun: 0.95
- Knife/Blade: 0.85
- Bat: 0.60
- Stick: 0.40
- Unarmed: 0.0

**Emotion Threat (25% weight)**
- Panicked/Distressed: 0.80
- Angry: 0.75
- Fearful: 0.60
- Neutral/Calm/Happy: 0.0

**Audio Threat (20% weight)**
- Panic: 0.85
- Threat: 0.80
- Distressed: 0.70
- Abnormal: 0.60
- Neutral/Calm: 0.0

**Behavioral Anomalies (5% weight)**
- Visible weapon: 0.40
- Emotional escalation: 0.30
- Audio escalation: 0.30
- Speech flags: 0.20
- Weapon + uniformed personnel: -0.20 (reduces threat)

## SOP Context Multipliers

### Location Multipliers
```python
{
    "school": 1.4,      # Highest sensitivity
    "airport": 1.3,
    "bank": 1.2,
    "street": 1.0,      # Baseline
    "office": 0.9
}
```

### Security Level Multipliers
```python
{
    "critical": 1.5,
    "high": 1.3,
    "medium": 1.0,
    "low": 0.8
}
```

## Learning Agent Integration

Optionally integrates top 3 scenario predictions:
- Scenario probability adds up to 10% to threat score
- If escalation predicted within 60s: additional 15% boost
- Sets trend to "escalating"

## Output Format

```python
ReasoningOutput {
    threat_level: "critical" | "high" | "medium" | "low",
    
    recommended_action: {
        action: "immediate_alert" | "escalate" | "elevated_monitoring" | "monitor",
        priority: "critical" | "high" | "medium" | "low",
        reason: "Human-readable explanation",
        confidence: 0.0 - 1.0
    },
    
    explanation: {
        summary: "Overall threat assessment",
        key_factors: ["factor1", "factor2", ...],
        evidence: {weapon, emotion, tone, ...},
        anomalies_detected: [...],
        temporal_analysis: "Trend analysis"
    },
    
    metrics: {
        weapon_threat_score: 0.0 - 1.0,
        emotion_threat_score: 0.0 - 1.0,
        audio_threat_score: 0.0 - 1.0,
        behavioral_anomaly_score: 0.0 - 1.0,
        combined_threat_score: 0.0 - 1.0
    }
}
```

## API Usage

```python
from cloud_reasoning.cloud_reasoning_agent import CloudReasoningAgent, SOPContext

# Initialize
agent = CloudReasoningAgent()

# Prepare SOP context (optional)
sop = SOPContext(
    location_type="school",
    time_of_day="peak_hours",
    security_level="high",
    active_alerts=[],
    personnel_count=500,
    restricted_weapons=["gun", "explosive"]
)

# Get threat assessment
output = agent.reason(
    source_id="camera_1",
    weapon_detected="gun",
    emotion="angry",
    tone="threat",
    confidence_scores={"weapon": 0.95, "emotion": 0.8, "tone": 0.85},
    risk_hints=["visible_weapon", "emotional_escalation"],
    scenario_predictions=None,  # Optional learning agent predictions
    sop_context=sop
)

# Use output
print(f"Threat: {output.threat_level}")
print(f"Action: {output.recommended_action.action}")
print(f"Confidence: {output.confidence:.2%}")
```

## Deployment Options

### Option 1: AWS Lambda
```python
# Serverless cloud function
# Scales automatically
# Pay per invocation
```

### Option 2: Docker Container
```bash
docker build -t cloud-reasoning .
docker run -p 8000:8000 cloud-reasoning
```

### Option 3: Dedicated Server
```bash
python -m uvicorn cloud_api:app --host 0.0.0.0 --port 8000
```

## Architecture Diagram

```
Officer's Device (Phone/Bodycam)
  ├─ Run lightweight perception models
  └─ Send perception results → CLOUD SERVER
                                  ↓
                    CLOUD REASONING AGENT
                    (Full 4-factor analysis)
                    ├─ Threat scoring
                    ├─ SOP context
                    ├─ Learning integration
                    └─ Detailed explanation
                                  ↓
                    Return threat_level + action
                                  ↓
                    Display on officer's device
```

## Example Scenarios

### Scenario 1: Gun at School
```
Input:
  weapon_detected="gun"
  emotion="angry"
  tone="threat"
  location="school"
  security_level="high"

Calculation:
  weapon_threat = 0.95 × 0.95 = 0.9025
  emotion_threat = 0.75 × 0.8 = 0.60
  audio_threat = 0.80 × 0.85 = 0.68
  behavioral = 0.40 + 0.30 = 0.70
  
  combined = (0.9025 × 0.5) + (0.60 × 0.25) + (0.68 × 0.2) + (0.70 × 0.05)
           = 0.451 + 0.15 + 0.136 + 0.035
           = 0.772
  
  With SOP: 0.772 × 1.4 (school) × 1.3 (high) = 1.40 → capped at 1.0
  
Output:
  threat_level = "CRITICAL" ⚠️
  action = "immediate_alert"
```

### Scenario 2: Unarmed Person
```
Input:
  weapon_detected="unarmed"
  emotion="calm"
  tone="neutral"

Calculation:
  weapon_threat = 0.0 × 0.9 = 0.0
  emotion_threat = 0.0 × 0.9 = 0.0
  audio_threat = 0.0 × 0.9 = 0.0
  behavioral = 0.0
  
  combined = 0.0

Output:
  threat_level = "LOW" ✓
  action = "monitor"
```

## File Structure

```
cloud_reasoning/
├── cloud_reasoning_agent.py  (600+ lines)
├── README.md                 (this file)
├── example_usage.py
└── requirements.txt
```

## Next Steps

1. Deploy to cloud server
2. Set up API endpoint: `POST /reason`
3. Connect officer devices to cloud API
4. Monitor response times and accuracy
5. Integrate learning agent (optional)

---

**For local/edge alternative, see:** `../local_reasoning/README.md`
