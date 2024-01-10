import { useQueryClient } from '@tanstack/react-query'
import { QUERY_KEYS } from '@/hooks/useAlertStream'
import { confidencePercent, cn } from '@/utils'
import type { DecisionOutput, DecisionAction } from '@/types'

const ACTION_STYLES: Record<DecisionAction, string> = {
  dispatch: 'text-critical',
  escalate: 'text-high',
  lockdown: 'text-critical',
  monitor: 'text-low',
  standby: 'text-text-2',
}

const BADGE_STYLES: Record<DecisionAction, string> = {
  dispatch: 'badge-incident',
  escalate: 'badge-degraded',
  lockdown: 'badge-incident',
  monitor: 'badge-live',
  standby: 'badge',
}

const BADGE_TEXT: Record<DecisionAction, string> = {
  dispatch: 'ACTION REQUIRED',
  escalate: 'DEGRADED CONFIDENCE',
  lockdown: 'IMMEDIATE ACTION',
  monitor: 'MONITORING',
  standby: 'STANDBY',
}

export function DecisionPanel() {
  const qc = useQueryClient()
  const decision = qc.getQueryData<DecisionOutput>(QUERY_KEYS.decision)

  return (
    <div className="panel">
      <div className="panel-header">
        <div className="panel-title">
          <LockIcon />
          Decision output
        </div>
        {decision && (
          <span className={BADGE_STYLES[decision.action]}>
            {BADGE_TEXT[decision.action]}
          </span>
        )}
      </div>

      <div className="p-2 flex flex-col gap-[6px]">
        {/* Recommended action */}
        <div className="text-center py-3 bg-bg-2 border border-border rounded-[4px]">
          <div className="font-mono text-[9px] text-text-4 uppercase tracking-[0.08em] mb-[5px]">
            Recommended response
          </div>
          <div className={cn('font-mono text-[18px] font-bold tracking-[0.06em]',
            decision ? ACTION_STYLES[decision.action] : 'text-text-4'
          )}>
            {decision ? decision.action.toUpperCase() : '—'}
          </div>
        </div>

        {/* Rationale */}
        <div className="bg-bg-2 border border-border rounded-[4px] p-[8px_10px]">
          <div className="font-mono text-[8px] text-text-4 uppercase tracking-[0.08em] mb-1">
            Rationale
          </div>
          <div className="font-mono text-[9px] text-text-3 leading-[1.6] tracking-[0.02em]">
            {decision?.rationale ?? 'Awaiting decision data...'}
          </div>
        </div>

        {/* Confidence */}
        <div className={cn(
          'font-mono text-[10px] font-bold text-center py-[6px] tracking-[0.04em]',
          decision?.isDegraded ? 'text-degraded' : 'text-critical'
        )}>
          {decision
            ? `CONFIDENCE: ${decision.isDegraded ? 'DEGRADED' : confidencePercent(decision.confidence)}`
            : '—'
          }
        </div>
      </div>
    </div>
  )
}

function LockIcon() {
  return (
    <svg width="9" height="9" viewBox="0 0 9 9" fill="none">
      <rect x="1" y="3.5" width="7" height="5" rx="1" stroke="currentColor" strokeWidth="1" />
      <path d="M3 3.5V2.5a1.5 1.5 0 013 0v1" stroke="currentColor" strokeWidth="1" />
    </svg>
  )
}
