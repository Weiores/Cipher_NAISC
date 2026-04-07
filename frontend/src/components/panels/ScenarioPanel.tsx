import { useQueryClient } from '@tanstack/react-query'
import { QUERY_KEYS } from '@/hooks/useAlertStream'
import { useUIStore } from '@/store/ui.store'
import { cn } from '@/utils'
import type { ScenarioPrediction } from '@/types'

const PROB_COLORS = ['text-critical', 'text-high', 'text-medium']
const BAR_COLORS = ['bg-critical', 'bg-high', 'bg-medium']

export function ScenarioPanel() {
  const qc = useQueryClient()
  const mode = useUIStore((s) => s.mode)
  const scenarios = qc.getQueryData<ScenarioPrediction[]>(QUERY_KEYS.scenarios) ?? []

  const isDegraded = mode === 'degraded'

  return (
    <div className="panel">
      <div className="panel-header">
        <div className="panel-title">
          <TriangleIcon />
          Scenario predictions
        </div>
        {isDegraded
          ? <span className="badge-sop">SOP ONLY</span>
          : mode === 'incident'
            ? <span className="badge-incident">ACTIVE SCENARIO</span>
            : <span className="badge-info">LEARNING AGENT</span>
        }
      </div>

      <div className="p-2 flex flex-col gap-[6px]">
        {scenarios.length === 0
          ? <EmptyState />
          : scenarios.map((s, i) => (
              <ScenarioItem key={s.rank} scenario={s} index={i} isDegraded={isDegraded} />
            ))
        }
      </div>
    </div>
  )
}

function ScenarioItem({
  scenario,
  index,
  isDegraded,
}: {
  scenario: ScenarioPrediction
  index: number
  isDegraded: boolean
}) {
  const probPercent = Math.round(scenario.probability * 100)

  return (
    <div className="bg-bg-2 border border-border rounded-[4px] p-[8px_10px]">
      <div className="flex justify-between items-center mb-1">
        <span className="font-mono text-[9px] text-text-4 tracking-[0.06em]">
          SCENARIO {String(scenario.rank).padStart(2, '0')}
        </span>
        <span className={cn('font-mono text-[13px] font-bold', PROB_COLORS[index])}>
          {scenario.isAvailable ? `${probPercent}%` : '—'}
        </span>
      </div>

      <div className="font-[Barlow_Condensed] text-[12px] font-semibold text-text-1 mb-[5px] tracking-[0.02em]">
        {scenario.name}
      </div>

      {scenario.isAvailable && !isDegraded ? (
        <div className="h-[2px] bg-bg-4 rounded-[1px] overflow-hidden">
          <div
            className={cn('h-full rounded-[1px] transition-all duration-700', BAR_COLORS[index])}
            style={{ width: `${probPercent}%` }}
          />
        </div>
      ) : (
        <div className="font-mono text-[8px] text-text-4 tracking-[0.05em]">
          {isDegraded ? 'LEARNING AGENT OFFLINE' : 'UNAVAILABLE'}
        </div>
      )}
    </div>
  )
}

function EmptyState() {
  return (
    <div className="font-mono text-[10px] text-text-4 text-center py-4 tracking-[0.05em]">
      AWAITING PREDICTION DATA
    </div>
  )
}

function TriangleIcon() {
  return (
    <svg width="9" height="9" viewBox="0 0 9 9" fill="none">
      <path d="M4.5 1L8.5 8H0.5L4.5 1Z" stroke="currentColor" strokeWidth="1" />
    </svg>
  )
}
