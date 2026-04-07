import { z } from 'zod'

export const ThreatLevelSchema = z.enum(['critical', 'high', 'medium', 'low', 'clear'])
export const SystemModeSchema = z.enum(['cloud', 'degraded', 'incident'])
export const WeaponTypeSchema = z.enum(['knife', 'gun', 'unarmed', 'none'])
export const EmotionTypeSchema = z.enum(['fearful', 'angry', 'neutral', 'none'])
export const VoiceStressSchema = z.enum(['panic', 'elevated', 'calm', 'none'])
export const BehaviourSchema = z.enum(['anomalous', 'normal'])
export const DecisionActionSchema = z.enum(['dispatch', 'escalate', 'lockdown', 'monitor', 'standby'])
export const ModelStatusSchema = z.enum(['online', 'offline', 'degraded'])

export const BoundingBoxSchema = z.object({
  x: z.number().min(0).max(1),
  y: z.number().min(0).max(1),
  width: z.number().min(0).max(1),
  height: z.number().min(0).max(1),
  label: z.string(),
})

export const AlertSchema = z.object({
  alertId: z.string().uuid(),
  type: z.string(),
  detail: z.string(),
  level: ThreatLevelSchema,
  cameraId: z.string().optional(),
  timestamp: z.string().datetime(),
  acknowledged: z.boolean(),
  acknowledgedBy: z.string().optional(),
  acknowledgedAt: z.string().datetime().optional(),
  isDegraded: z.boolean(),
})

export const ZoneStateSchema = z.object({
  zoneId: z.string(),
  zoneName: z.string(),
  floor: z.number(),
  threatLevel: ThreatLevelSchema,
  activeAlerts: z.number(),
  lastUpdated: z.string().datetime(),
})

export const ScenarioPredictionSchema = z.object({
  rank: z.number().int().min(1).max(3),
  name: z.string(),
  probability: z.number().min(0).max(1),
  isAvailable: z.boolean(),
})

export const DecisionOutputSchema = z.object({
  action: DecisionActionSchema,
  threatLevel: ThreatLevelSchema,
  confidence: z.number().min(0).max(1),
  rationale: z.string(),
  isDegraded: z.boolean(),
  generatedAt: z.string().datetime(),
})

export const ModelHealthSchema = z.object({
  name: z.string(),
  status: ModelStatusSchema,
  confidence: z.number().min(0).max(1),
  lastInference: z.string().datetime(),
  inferenceMs: z.number(),
})

export const SentinelMessageSchema = z.object({
  messageId: z.string().uuid(),
  timestamp: z.string().datetime(),
  mode: SystemModeSchema,
  decision: DecisionOutputSchema,
  scenarios: z.array(ScenarioPredictionSchema),
  alerts: z.array(AlertSchema),
  zones: z.array(ZoneStateSchema),
})

export type ValidatedSentinelMessage = z.infer<typeof SentinelMessageSchema>
