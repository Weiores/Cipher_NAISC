# NAISC UI Layer: Telegram Alerting

This service accepts `ReasoningOutput` JSON from the existing reasoning pipeline and turns important incidents into Telegram channel alerts with demo-safe fallback logging.

## Best Integration Point

The cleanest integration point is the shared reasoning schema in `../reasoning-layer/schemas.py`.

- `POST /telegram/alert` accepts the raw `ReasoningOutput` returned by the reasoning layer.
- If you have extra operator context such as `location`, `top_scenario`, or a custom `anomaly_type`, send an envelope:

```json
{
  "reasoning": { "... existing ReasoningOutput ..." },
  "location": "Gate A",
  "anomaly_type": "weapon_detected",
  "top_scenario": "Armed intruder approaching entrance"
}
```

This keeps the Telegram integration separate from the perception and reasoning services while reusing the existing model contract.

## Features

- Configurable alert routing policy
- Forced routing for `weapon_detected` or `critical` threats
- Duplicate suppression by `location + anomaly_type + threat_level`
- Escalation when an alert is not acknowledged in time
- Channel-only alert posting
- Short incident IDs in every alert, for example `INC-91C801`
- Optional `Open Console` URL button for external dashboards, enriched with incident context in query params
- Reply-based escalation and incident update posts that stay grouped under the original channel alert
- Optional incident clip upload as a regular Telegram video reply
- In-memory feedback capture ready to swap for a database later
- Demo mode when Telegram credentials are missing

## Local Setup

```bash
cd NAISC/ui-layer
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
uvicorn app.main:app --reload --port 8010
```

## Environment Variables

Set these in `.env`:

- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`
- `OPERATOR_CONSOLE_URL`
- `TELEGRAM_MAX_VIDEO_BYTES`
- `ALERT_MIN_THREAT_LEVEL`
- `ALWAYS_SEND_THREAT_LEVELS`
- `ALWAYS_SEND_ANOMALIES`
- `ALERT_DEDUPE_WINDOW_SECONDS`
- `ALERT_ESCALATION_TIMEOUT_SECONDS`
- `ALERT_ESCALATION_ENABLED`

If `TELEGRAM_BOT_TOKEN` or `TELEGRAM_CHAT_ID` is missing, the service stays up and logs the formatted alert locally for demos.

## Endpoints

- `GET /telegram/health`
- `POST /telegram/alert`

## Sample Payload

See [`samples/reasoning_alert.json`](C:\Users\aaore\Downloads\Coding\NAISC\ui-layer\samples\reasoning_alert.json).

## Trigger an Alert

```bash
curl -X POST "http://127.0.0.1:8010/telegram/alert" ^
  -H "Content-Type: application/json" ^
  --data-binary "@samples/reasoning_alert.json"
```

There is no webhook mode in this version. Alerts are broadcast to the configured Telegram channel only.

If `OPERATOR_CONSOLE_URL` is set, the alert includes an `Open Console` button. The service appends:

- `incident_id`
- `alert_id`
- `source_id`
- `location`
- `anomaly_type`

If `video_path` is included in the alert payload and points to a local file inside the repo, the bot sends it as a regular Telegram video reply under the alert message.

Oversized clips are skipped before upload. The default max size is `45,000,000` bytes and can be changed with `TELEGRAM_MAX_VIDEO_BYTES`.

## Behavior Notes

- Escalation sends a follow-up channel message if the alert remains unresolved after the configured timeout.
- If a later alert arrives for the same `location + anomaly_type` and the threat level changes or the action becomes `de_escalate` / `all_clear`, the bot posts a reply update under the original alert instead of creating a separate top-level thread.
- Because this is channel-only mode, Telegram button callbacks and operator feedback capture are not used.
