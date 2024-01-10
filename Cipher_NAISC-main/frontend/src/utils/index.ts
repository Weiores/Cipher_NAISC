import { clsx, type ClassValue } from 'clsx'
import { formatDistanceToNow, format } from 'date-fns'
import type { ThreatLevel, SystemMode, DecisionAction } from '@/types'

// ── Class name builder ───────────────────────
export function cn(...inputs: ClassValue[]) {
  return clsx(inputs)
}

// ── Threat level helpers ─────────────────────
export function threatLevelColor(level: ThreatLevel): string {
  const map: Record<ThreatLevel, string> = {
    critical: 'text-critical',
    high: 'text-high',
    medium: 'text-medium',
    low: 'text-low',
    clear: 'text-text-4',
  }
  return map[level]
}

export function threatLevelBg(level: ThreatLevel): string {
  const map: Record<ThreatLevel, string> = {
    critical: 'bg-critical/10 border-critical/30',
    high: 'bg-high/10 border-high/25',
    medium: 'bg-medium/10 border-medium/25',
    low: 'bg-low/10 border-low/20',
    clear: 'bg-bg-3 border-border',
  }
  return map[level]
}

export function threatLevelBarColor(level: ThreatLevel): string {
  const map: Record<ThreatLevel, string> = {
    critical: 'bg-critical',
    high: 'bg-high',
    medium: 'bg-medium',
    low: 'bg-low',
    clear: 'bg-text-4',
  }
  return map[level]
}

export function threatLevelLabel(level: ThreatLevel): string {
  return level.toUpperCase()
}

// ── System mode helpers ──────────────────────
export function modeColor(mode: SystemMode): string {
  const map: Record<SystemMode, string> = {
    cloud: 'text-cloud',
    degraded: 'text-degraded',
    incident: 'text-critical',
  }
  return map[mode]
}

export function modeBorderColor(mode: SystemMode): string {
  const map: Record<SystemMode, string> = {
    cloud: 'border-cloud/20',
    degraded: 'border-degraded/25',
    incident: 'border-critical/30',
  }
  return map[mode]
}

// ── Decision action helpers ──────────────────
export function actionLabel(action: DecisionAction): string {
  const map: Record<DecisionAction, string> = {
    dispatch: 'Dispatch security',
    escalate: 'Escalate to supervisor',
    lockdown: 'Initiate lockdown',
    monitor: 'Continue monitoring',
    standby: 'Stand by',
  }
  return map[action]
}

export function actionColor(action: DecisionAction): string {
  const map: Record<DecisionAction, string> = {
    dispatch: 'text-critical border-critical/25 bg-critical/10',
    escalate: 'text-high border-high/20 bg-high/8',
    lockdown: 'text-critical border-critical/25 bg-critical/10',
    monitor: 'text-text-3 border-border bg-bg-2',
    standby: 'text-text-3 border-border bg-bg-2',
  }
  return map[action]
}

// ── Confidence helpers ───────────────────────
export function confidencePercent(value: number): string {
  return `${Math.round(value * 100)}%`
}

export function confidenceColor(value: number): string {
  if (value >= 0.85) return 'text-critical'
  if (value >= 0.70) return 'text-medium'
  return 'text-low'
}

// ── Time formatters ──────────────────────────
export function formatTimestamp(iso: string): string {
  return format(new Date(iso), 'HH:mm:ss')
}

export function formatRelative(iso: string): string {
  return formatDistanceToNow(new Date(iso), { addSuffix: true })
}

export function formatDuration(seconds: number): string {
  const m = Math.floor(seconds / 60)
  const s = seconds % 60
  return `${m}:${s.toString().padStart(2, '0')}`
}

// ── Derive system stats from message ─────────
import type { Alert, SystemStats, CipherMessage } from '@/types'

export function deriveSystemStats(
  alerts: Alert[],
  message: CipherMessage | null
): SystemStats {
  const active = alerts.filter((a) => !a.acknowledged)
  return {
    totalActiveAlerts: active.length,
    criticalCount: active.filter((a) => a.level === 'critical').length,
    highCount: active.filter((a) => a.level === 'high').length,
    mediumCount: active.filter((a) => a.level === 'medium').length,
    lowCount: active.filter((a) => a.level === 'low').length,
    currentThreatLevel: message?.decision.threatLevel ?? 'clear',
    threatLevelChangedAt: message?.decision.generatedAt ?? new Date().toISOString(),
    avgModelConfidence:
      message?.models
        ? Math.round(
            ((message.models.weapon.confidence +
              message.models.emotion.confidence +
              message.models.tone.confidence) /
              3) *
              100
          )
        : 0,
    mode: message?.mode ?? 'cloud',
    incidentsToday: 7,
    resolvedToday: 3,
  }
}
