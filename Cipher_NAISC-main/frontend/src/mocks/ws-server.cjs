/**
 * CIPHER — Mock WebSocket Server
 *
 * Fires real-shaped CipherMessage payloads on a timer.
 * Run with: node src/mocks/ws-server.cjs
 *
 * Simulates all three modes: cloud, degraded, incident
 * Switch mode by sending a message: { "setMode": "degraded" }
 */

const { WebSocketServer } = require('ws')
const { randomUUID } = require('crypto')

const PORT = 8000
const wss = new WebSocketServer({ port: PORT })

let currentMode = 'cloud'
let alertCounter = 0

console.log(`[MOCK WS] Server running on ws://localhost:${PORT}/ws`)

// ── Helpers ──────────────────────────────────

function ts() {
  return new Date().toISOString()
}

function randFloat(min, max) {
  return Math.round((Math.random() * (max - min) + min) * 100) / 100
}

function randInt(min, max) {
  return Math.floor(Math.random() * (max - min + 1)) + min
}

// ── Mock data builders ───────────────────────

function buildAlerts(mode) {
  const baseAlerts = {
    cloud: [
      { type: 'Weapon detected', detail: 'Knife · CAM-01 Entrance · vision 91%', level: 'critical', cameraId: 'CAM-01' },
      { type: 'Emotion: fearful', detail: '3 subjects · CAM-01 · emotion 84%', level: 'high', cameraId: 'CAM-01' },
      { type: 'Voice stress: panic', detail: 'Audio zone A · tone 78%', level: 'high' },
      { type: 'Behaviour anomaly', detail: 'Rapid movement · CAM-01 · 67%', level: 'medium', cameraId: 'CAM-01' },
    ],
    degraded: [
      { type: 'Weapon detected', detail: 'SOP rule match · CAM-01 · degraded mode', level: 'critical', cameraId: 'CAM-01' },
      { type: 'Motion threshold', detail: 'SOP zone A trigger · no ML enrichment', level: 'high' },
    ],
    incident: [
      { type: 'Weapon: firearm', detail: 'CAM-01 · vision 93% · PRIORITY 1', level: 'critical', cameraId: 'CAM-01' },
      { type: 'Weapon: knife', detail: 'CAM-02 · vision 87% · secondary', level: 'critical', cameraId: 'CAM-02' },
      { type: 'Emotion: terror', detail: 'Multiple subjects · CAM-01 CAM-02 · 91%', level: 'critical' },
      { type: 'Voice: screaming', detail: 'Audio zones A B C · tone 95%', level: 'critical' },
      { type: 'Mass movement', detail: 'Crowd dispersal · CAM-02 CAM-03 · 88%', level: 'high', cameraId: 'CAM-02' },
      { type: 'Entry breach', detail: 'Emergency exit forced · zone D · sensor', level: 'high' },
    ],
  }

  return (baseAlerts[mode] || baseAlerts.cloud).map((a) => ({
    alertId: randomUUID(),
    type: a.type,
    detail: a.detail,
    level: a.level,
    cameraId: a.cameraId,
    timestamp: ts(),
    acknowledged: false,
    isDegraded: mode === 'degraded',
  }))
}

function buildZones(mode) {
  const configs = {
    cloud: [
      { zoneId: 'z-carpark', zoneName: 'Car park', floor: 1, threatLevel: 'clear', activeAlerts: 0 },
      { zoneId: 'z-reception', zoneName: 'Reception', floor: 1, threatLevel: 'high', activeAlerts: 2 },
      { zoneId: 'z-server', zoneName: 'Server room', floor: 1, threatLevel: 'clear', activeAlerts: 0 },
      { zoneId: 'z-lobby', zoneName: 'Lobby', floor: 1, threatLevel: 'medium', activeAlerts: 1 },
      { zoneId: 'z-entrance', zoneName: 'Entrance', floor: 1, threatLevel: 'critical', activeAlerts: 4 },
      { zoneId: 'z-stairwell', zoneName: 'Stairwell', floor: 1, threatLevel: 'clear', activeAlerts: 0 },
      { zoneId: 'z-canteen', zoneName: 'Canteen', floor: 1, threatLevel: 'clear', activeAlerts: 0 },
      { zoneId: 'z-corridor', zoneName: 'Corridor A', floor: 1, threatLevel: 'low', activeAlerts: 1 },
      { zoneId: 'z-exit', zoneName: 'Exit B', floor: 1, threatLevel: 'clear', activeAlerts: 0 },
    ],
    degraded: [
      { zoneId: 'z-carpark', zoneName: 'Car park', floor: 1, threatLevel: 'clear', activeAlerts: 0 },
      { zoneId: 'z-reception', zoneName: 'Reception', floor: 1, threatLevel: 'medium', activeAlerts: 1 },
      { zoneId: 'z-server', zoneName: 'Server room', floor: 1, threatLevel: 'clear', activeAlerts: 0 },
      { zoneId: 'z-lobby', zoneName: 'Lobby', floor: 1, threatLevel: 'medium', activeAlerts: 1 },
      { zoneId: 'z-entrance', zoneName: 'Entrance', floor: 1, threatLevel: 'high', activeAlerts: 2 },
      { zoneId: 'z-stairwell', zoneName: 'Stairwell', floor: 1, threatLevel: 'clear', activeAlerts: 0 },
      { zoneId: 'z-canteen', zoneName: 'Canteen', floor: 1, threatLevel: 'clear', activeAlerts: 0 },
      { zoneId: 'z-corridor', zoneName: 'Corridor A', floor: 1, threatLevel: 'clear', activeAlerts: 0 },
      { zoneId: 'z-exit', zoneName: 'Exit B', floor: 1, threatLevel: 'clear', activeAlerts: 0 },
    ],
    incident: [
      { zoneId: 'z-carpark', zoneName: 'Car park', floor: 1, threatLevel: 'high', activeAlerts: 2 },
      { zoneId: 'z-reception', zoneName: 'Reception', floor: 1, threatLevel: 'critical', activeAlerts: 5 },
      { zoneId: 'z-server', zoneName: 'Server room', floor: 1, threatLevel: 'medium', activeAlerts: 1 },
      { zoneId: 'z-lobby', zoneName: 'Lobby', floor: 1, threatLevel: 'critical', activeAlerts: 4 },
      { zoneId: 'z-entrance', zoneName: 'Entrance', floor: 1, threatLevel: 'critical', activeAlerts: 6 },
      { zoneId: 'z-stairwell', zoneName: 'Stairwell', floor: 1, threatLevel: 'high', activeAlerts: 2 },
      { zoneId: 'z-canteen', zoneName: 'Canteen', floor: 1, threatLevel: 'high', activeAlerts: 3 },
      { zoneId: 'z-corridor', zoneName: 'Corridor A', floor: 1, threatLevel: 'critical', activeAlerts: 4 },
      { zoneId: 'z-exit', zoneName: 'Exit B', floor: 1, threatLevel: 'medium', activeAlerts: 1 },
    ],
  }

  return (configs[mode] || configs.cloud).map((z) => ({
    ...z,
    lastUpdated: ts(),
  }))
}

function buildDecision(mode) {
  const decisions = {
    cloud: {
      action: 'dispatch',
      threatLevel: 'high',
      confidence: 0.91,
      rationale: 'Weapon at entrance 91% conf. Emotion: fearful 84%. Voice: panic 78%. Threshold exceeded.',
      isDegraded: false,
      generatedAt: ts(),
    },
    degraded: {
      action: 'escalate',
      threatLevel: 'high',
      confidence: 0.45,
      rationale: 'Cloud unavailable. SOP match: armed intruder. No ML enrichment. Manual verification required.',
      isDegraded: true,
      generatedAt: ts(),
    },
    incident: {
      action: 'lockdown',
      threatLevel: 'critical',
      confidence: 0.93,
      rationale: 'ACTIVE SHOOTER PROTOCOL. Firearm 93% + knife 87%. Mass panic confirmed. Immediate lockdown required.',
      isDegraded: false,
      generatedAt: ts(),
    },
  }
  return decisions[mode] || decisions.cloud
}

function buildScenarios(mode) {
  if (mode === 'degraded') {
    return [
      { rank: 1, name: 'SOP: armed intruder protocol', probability: 0, isAvailable: false },
      { rank: 2, name: 'SOP: evacuation trigger check', probability: 0, isAvailable: false },
      { rank: 3, name: 'N/A — learning agent offline', probability: 0, isAvailable: false },
    ]
  }
  if (mode === 'incident') {
    return [
      { rank: 1, name: 'Active shooter — immediate threat', probability: 0.89, isAvailable: true },
      { rank: 2, name: 'Multiple armed assailants', probability: 0.08, isAvailable: true },
      { rank: 3, name: 'Hostage situation developing', probability: 0.03, isAvailable: true },
    ]
  }
  return [
    { rank: 1, name: 'Armed confrontation escalation', probability: 0.72, isAvailable: true },
    { rank: 2, name: 'Robbery attempt in progress', probability: 0.19, isAvailable: true },
    { rank: 3, name: 'Domestic disturbance, no weapon', probability: 0.09, isAvailable: true },
  ]
}

function buildMessage(mode) {
  const isOnline = mode !== 'degraded'
  const isIncident = mode === 'incident'

  return {
    messageId: randomUUID(),
    timestamp: ts(),
    mode,
    fusedEvent: {
      eventId: randomUUID(),
      siteId: 'site-001',
      timestamp: ts(),
      durationSeconds: randInt(15, 120),
      weapon: {
        detected: isOnline ? (isIncident ? 'gun' : 'knife') : 'none',
        confidence: isOnline ? randFloat(0.88, 0.95) : 0,
        cameraId: 'CAM-01',
        boundingBox: isOnline ? { x: 0.12, y: 0.08, width: 0.42, height: 0.78, label: isIncident ? 'FIREARM' : 'WEAPON' } : null,
      },
      emotion: {
        detected: isOnline ? (isIncident ? 'fearful' : 'fearful') : 'none',
        confidence: isOnline ? randFloat(0.80, 0.92) : 0,
        subjectCount: isOnline ? randInt(2, 6) : 0,
        cameraId: 'CAM-01',
      },
      voice: {
        detected: isOnline ? 'panic' : 'none',
        confidence: isOnline ? randFloat(0.74, 0.88) : 0,
        audioZone: 'zone-A',
      },
      behaviour: {
        detected: 'anomalous',
        confidence: randFloat(0.60, 0.75),
        patternType: isIncident ? 'mass_movement' : 'rapid_movement',
        cameraId: 'CAM-01',
      },
      sourceIds: isIncident ? ['CAM-01', 'CAM-02', 'AUDIO-A', 'AUDIO-B'] : ['CAM-01', 'AUDIO-A'],
      mode,
    },
    decision: buildDecision(mode),
    scenarios: buildScenarios(mode),
    alerts: buildAlerts(mode),
    zones: buildZones(mode),
    cameras: [
      { cameraId: 'CAM-01', label: 'Entrance', location: 'Front entrance', isOnline: true, threatLevel: isOnline ? 'critical' : 'high', activeBoundingBoxes: [], streamUrl: null },
      { cameraId: 'CAM-02', label: 'Lobby', location: 'Main lobby', isOnline: true, threatLevel: isIncident ? 'critical' : 'clear', activeBoundingBoxes: [], streamUrl: null },
      { cameraId: 'CAM-03', label: 'Corridor', location: 'Corridor A', isOnline: true, threatLevel: 'clear', activeBoundingBoxes: [], streamUrl: null },
      { cameraId: 'CAM-04', label: 'Stairwell', location: 'Stairwell B', isOnline: true, threatLevel: 'clear', activeBoundingBoxes: [], streamUrl: null },
    ],
    models: {
      weapon: { name: 'Weapon detection', status: isOnline ? 'online' : 'offline', confidence: isOnline ? randFloat(0.88, 0.95) : 0, lastInference: ts(), inferenceMs: randInt(12, 35) },
      emotion: { name: 'Emotion detection', status: isOnline ? 'online' : 'offline', confidence: isOnline ? randFloat(0.80, 0.92) : 0, lastInference: ts(), inferenceMs: randInt(18, 45) },
      tone: { name: 'Tone / voice stress', status: isOnline ? 'online' : 'offline', confidence: isOnline ? randFloat(0.74, 0.88) : 0, lastInference: ts(), inferenceMs: randInt(8, 22) },
      overallStatus: isOnline ? 'online' : 'offline',
      fusionStatus: isOnline ? 'online' : 'offline',
    },
    learningAgent: {
      status: isOnline ? 'online' : 'offline',
      modelVersion: 'v3.2.1',
      lastTrained: new Date(Date.now() - 6 * 60 * 60 * 1000).toISOString(),
      predictionCount: randInt(130, 160),
      accuracySevenDay: isOnline ? randFloat(0.91, 0.96) : 0,
      sopVersion: '2.4.1',
      isAvailable: isOnline,
    },
  }
}

// ── WebSocket server ─────────────────────────

wss.on('connection', (ws, req) => {
  console.log(`[MOCK WS] Client connected from ${req.socket.remoteAddress}`)

  // Send initial state immediately on connect
  ws.send(JSON.stringify(buildMessage(currentMode)))

  // Handle incoming messages (e.g. operator actions or mode switch)
  ws.on('message', (data) => {
    try {
      const msg = JSON.parse(data.toString())
      if (msg.setMode && ['cloud', 'degraded', 'incident'].includes(msg.setMode)) {
        currentMode = msg.setMode
        console.log(`[MOCK WS] Mode switched to: ${currentMode}`)
        broadcast(buildMessage(currentMode))
      }
    } catch {
      // Ignore malformed messages
    }
  })

  ws.on('close', () => {
    console.log('[MOCK WS] Client disconnected')
  })
})

// ── Broadcast to all connected clients ───────

function broadcast(payload) {
  const msg = JSON.stringify(payload)
  wss.clients.forEach((client) => {
    if (client.readyState === 1) client.send(msg)
  })
}

// ── Periodic updates ─────────────────────────

// Broadcast a new event every 4 seconds
setInterval(() => {
  broadcast(buildMessage(currentMode))
}, 4000)

// Inject a single new alert every 7 seconds
setInterval(() => {
  const alert = buildAlerts(currentMode)[0]
  alert.alertId = randomUUID()
  alert.timestamp = ts()
  broadcast({ type: 'alert_inject', alert })
}, 7000)

console.log('[MOCK WS] Broadcasting every 4s. Connect at ws://localhost:8000')
console.log('[MOCK WS] Send { "setMode": "degraded" } to switch modes')
