import { useQueryClient } from '@tanstack/react-query'
import { QUERY_KEYS } from '@/hooks/useAlertStream'
import { useUIStore } from '@/store/ui.store'
import { threatLevelColor, confidencePercent, formatRelative, cn } from '@/utils'
import type { Alert, DecisionOutput, ModelsHealth } from '@/types'

export function StatCardsRow() {
  const qc = useQueryClient()
  const mode = useUIStore((s) => s.mode)

  const alerts = qc.getQueryData<Alert[]>(QUERY_KEYS.alerts) ?? []
  const decision = qc.getQueryData<DecisionOutput>(QUERY_KEYS.decision)
  const models = qc.getQueryData<ModelsHealth>(QUERY_KEYS.models)

  const active = alerts.filter((a) => !a.acknowledged)
  const critCount = active.filter((a) => a.level === 'critical').length
  const highCount = active.filter((a) => a.level === 'high').length

  const avgConf = models
    ? (models.weapon.confidence + models.emotion.confidence + models.tone.confidence) / 3
    : null

  const modeColor = { cloud: 'text-low', degraded: 'text-degraded', incident: 'text-critical' }[mode]
  const modeSub = { cloud: 'Learning agent active', degraded: 'SOP rules only', incident: 'All systems engaged' }[mode]

  return (
    <div className="grid gap-2" style={{ gridTemplateColumns: 'repeat(5, 1fr)' }}>

      {/* Active alerts */}
      <div className="stat-card stat-card-critical">
        <div className="font-mono text-[9px] text-text-3 uppercase tracking-[0.08em] mb-[6px]">
          Active alerts
        </div>
        <div className="font-mono text-[22px] text-critical leading-none mb-[3px]">
          {active.length}
        </div>
        <div className="font-mono text-[9px] text-text-3">
          <span className="text-text-2">{critCount} critical</span> · {highCount} high
        </div>
      </div>

      {/* Threat level */}
      <div className="stat-card stat-card-high">
        <div className="font-mono text-[9px] text-text-3 uppercase tracking-[0.08em] mb-[6px]">
          Threat level
        </div>
        <div className={cn('font-mono text-[22px] leading-none mb-[3px]', threatLevelColor(decision?.threatLevel ?? 'clear'))}>
          {(decision?.threatLevel ?? 'CLEAR').toUpperCase()}
        </div>
        <div className="font-mono text-[9px] text-text-3">
          {decision ? (
            <span>Escalated <span className="text-text-2">{formatRelative(decision.generatedAt)}</span></span>
          ) : '—'}
        </div>
      </div>

      {/* Avg confidence */}
      <div className="stat-card stat-card-medium">
        <div className="font-mono text-[9px] text-text-3 uppercase tracking-[0.08em] mb-[6px]">
          Avg confidence
        </div>
        <div className="font-mono text-[22px] text-medium leading-none mb-[3px]">
          {avgConf !== null ? confidencePercent(avgConf) : '—'}
        </div>
        <div className="font-mono text-[9px] text-text-3">
          {models?.overallStatus === 'online'
            ? <span>3 models · <span className="text-text-2">fusion OK</span></span>
            : <span className="text-degraded">Models offline</span>
          }
        </div>
      </div>

      {/* System mode */}
      <div className="stat-card stat-card-low">
        <div className="font-mono text-[9px] text-text-3 uppercase tracking-[0.08em] mb-[6px]">
          System mode
        </div>
        <div className={cn('font-mono text-[22px] leading-none mb-[3px]', modeColor)}>
          {mode.toUpperCase()}
        </div>
        <div className="font-mono text-[9px] text-text-3">
          {modeSub}
        </div>
      </div>

      {/* Incidents today */}
      <div className="stat-card stat-card-info">
        <div className="font-mono text-[9px] text-text-3 uppercase tracking-[0.08em] mb-[6px]">
          Incidents today
        </div>
        <div className="font-mono text-[22px] text-info leading-none mb-[3px]">
          7
        </div>
        <div className="font-mono text-[9px] text-text-3">
          <span className="text-text-2">3 resolved</span> · 4 open
        </div>
      </div>

    </div>
  )
}
