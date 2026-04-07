// ─────────────────────────────────────────────
// SENTINEL OPS — Core Type Definitions
// This file is the contract between frontend and backend.
// Any backend schema change must be reflected here first.
// ─────────────────────────────────────────────

// ── Enums / Union types ──────────────────────

export type SystemMode = 'cloud' | 'degraded' | 'incident'

export type ThreatLevel = 'critical' | 'high' | 'medium' | 'low' | 'clear'

export type WeaponType = 'knife' | 'gun' | 'unarmed' | 'none'

export type EmotionType = 'fearful' | 'angry' | 'neutral' | 'none'

export type VoiceStressType = 'panic' | 'elevated' | 'calm' | 'none'

export type BehaviourType = 'anomalous' | 'normal'

export type DecisionAction = 'dispatch' | 'escalate' | 'lockdown' | 'monitor' | 'standby'

export type ModelStatus = 'online' | 'offline' | 'degraded'

export type ConnectionStatus = 'connected' | 'disconnected' | 'reconnecting'

export type UserRole = 'operator' | 'analyst' | 'admin'

// ── Operator / Auth ──────────────────────────

export interface Operator {
  id: string
  name: string
  displayName: string
  role: UserRole
  siteId: string
}

// ── Model health ─────────────────────────────

export interface ModelHealth {
  name: string
  status: ModelStatus
  confidence: number        // 0–1
  lastInference: string     // ISO8601
  inferenceMs: number       // latency in ms
}

export interface ModelsHealth {
  weapon: ModelHealth
  emotion: ModelHealth
  tone: ModelHealth
  overallStatus: ModelStatus
  fusionStatus: ModelStatus
}

// ── Detection traits ─────────────────────────

export interface WeaponTrait {
  detected: WeaponType
  confidence: number        // 0–1
  boundingBox?: BoundingBox
  cameraId: string
}

export interface EmotionTrait {
  detected: EmotionType
  confidence: number
  subjectCount: number
  cameraId: string
}

export interface VoiceTrait {
  detected: VoiceStressType
  confidence: number
  audioZone: string
}

export interface BehaviourTrait {
  detected: BehaviourType
  confidence: number
  patternType: string       // e.g. 'rapid_movement', 'loitering'
  cameraId: string
}

export interface BoundingBox {
  x: number                 // normalised 0–1
  y: number
  width: number
  height: number
  label: string
}

// ── Fused event ──────────────────────────────

export interface FusedEvent {
  eventId: string
  siteId: string
  timestamp: string         // ISO8601
  durationSeconds: number
  weapon: WeaponTrait
  emotion: EmotionTrait
  voice: VoiceTrait
  behaviour: BehaviourTrait
  sourceIds: string[]       // camera/sensor IDs involved
  mode: SystemMode
}

// ── Scenario predictions ─────────────────────

export interface ScenarioPrediction {
  rank: number              // 1, 2, 3
  name: string
  probability: number       // 0–1, null in degraded mode
  isAvailable: boolean      // false when learning agent offline
}

// ── Decision output ──────────────────────────

export interface DecisionOutput {
  action: DecisionAction
  threatLevel: ThreatLevel
  confidence: number        // 0–1
  rationale: string
  isDegraded: boolean       // true when SOP-only, no ML enrichment
  generatedAt: string       // ISO8601
}

// ── Zone state ───────────────────────────────

export interface ZoneState {
  zoneId: string
  zoneName: string
  floor: number
  threatLevel: ThreatLevel
  activeAlerts: number
  lastUpdated: string       // ISO8601
}

// ── Camera state ─────────────────────────────

export interface CameraState {
  cameraId: string
  label: string
  location: string
  isOnline: boolean
  threatLevel: ThreatLevel
  activeBoundingBoxes: BoundingBox[]
  streamUrl?: string        // populated when real stream available
}

// ── Alert ────────────────────────────────────

export interface Alert {
  alertId: string
  type: string
  detail: string
  level: ThreatLevel
  cameraId?: string
  timestamp: string         // ISO8601
  acknowledged: boolean
  acknowledgedBy?: string
  acknowledgedAt?: string
  isDegraded: boolean
}

// ── Full WebSocket message ───────────────────
// This is the top-level shape of every message
// received from the backend WebSocket.

export interface SentinelMessage {
  messageId: string
  timestamp: string
  mode: SystemMode
  fusedEvent: FusedEvent
  decision: DecisionOutput
  scenarios: ScenarioPrediction[]
  alerts: Alert[]
  zones: ZoneState[]
  cameras: CameraState[]
  models: ModelsHealth
  learningAgent: LearningAgentStatus
}

// ── Learning agent ───────────────────────────

export interface LearningAgentStatus {
  status: ModelStatus
  modelVersion: string
  lastTrained: string       // ISO8601
  predictionCount: number
  accuracySevenDay: number  // 0–1
  sopVersion: string
  isAvailable: boolean
}

// ── Audit log ────────────────────────────────

export interface AuditEntry {
  entryId: string
  timestamp: string         // ISO8601
  operatorId: string
  operatorName: string
  action: string
  detail: string
  alertId?: string
}

// ── Operator action payload ──────────────────
// Sent TO the backend when operator takes action

export interface OperatorActionPayload {
  actionType: DecisionAction | 'acknowledge_alert' | 'mark_false_positive'
  alertId?: string
  operatorId: string
  note?: string
  timestamp: string
}

// ── System stats (derived, for stat cards) ───

export interface SystemStats {
  totalActiveAlerts: number
  criticalCount: number
  highCount: number
  mediumCount: number
  lowCount: number
  currentThreatLevel: ThreatLevel
  threatLevelChangedAt: string
  avgModelConfidence: number
  mode: SystemMode
  incidentsToday: number
  resolvedToday: number
}
