import { useQueryClient } from '@tanstack/react-query'
import { QUERY_KEYS } from '@/hooks/useAlertStream'
import { useUIStore } from '@/store/ui.store'
import { confidencePercent, cn } from '@/utils'
import type { ModelsHealth, ModelHealth } from '@/types'

export function ModelHealthPanel() {
  const qc = useQueryClient()
  const { confidenceHistory } = useUIStore()
  const models = qc.getQueryData<ModelsHealth>(QUERY_KEYS.models)

  const badgeCls = !models || models.overallStatus === 'offline'
    ? 'badge-degraded'
    : models.overallStatus === 'degraded'
      ? 'badge-sop'
      : 'badge-live'

  const badgeText = !models
    ? 'NO DATA'
    : models.overallStatus === 'offline'
      ? '0/3 ONLINE'
      : models.overallStatus === 'degraded'
        ? 'DEGRADED'
        : '3/3 ONLINE'

  return (
    <div className="panel">
      <div className="panel-header">
        <div className="panel-title">
          <PulseIcon />
          Model health
        </div>
        <span className={badgeCls}>{badgeText}</span>
      </div>

      {/* Model bars */}
      <div className="p-2 flex flex-col gap-[5px]">
        {models ? (
          <>
            <ModelRow model={models.weapon} barColor="bg-critical" />
            <ModelRow model={models.emotion} barColor="bg-high" />
            <ModelRow model={models.tone} barColor="bg-medium" />
          </>
        ) : (
          <div className="font-mono text-[10px] text-text-4 text-center py-3 tracking-[0.05em]">
            AWAITING MODEL DATA
          </div>
        )}
      </div>

      {/* Confidence history sparkline */}
      <div className="panel-header" style={{ borderTop: '1px solid var(--border)', borderBottom: 'none', marginTop: '4px' }}>
        <div className="panel-title">Confidence history</div>
      </div>
      <div className="p-2">
        <ConfidenceSparkline history={confidenceHistory} />
      </div>
    </div>
  )
}

function ModelRow({ model, barColor }: { model: ModelHealth; barColor: string }) {
  const isOnline = model.status === 'online'
  const conf = Math.round(model.confidence * 100)

  return (
    <div className="grid items-center gap-2" style={{ gridTemplateColumns: '90px 1fr 36px' }}>
      <div className="font-mono text-[9px] text-text-3 uppercase tracking-[0.05em] truncate">
        {model.name.split(' ')[0]}
      </div>
      <div className="bg-bg-3 rounded-[2px] h-[5px] overflow-hidden">
        <div
          className={cn('h-full rounded-[2px] transition-all duration-500', isOnline ? barColor : 'bg-text-4')}
          style={{ width: `${conf}%` }}
        />
      </div>
      <div className={cn('font-mono text-[9px] text-right', isOnline ? 'text-low' : 'text-text-4')}>
        {isOnline ? confidencePercent(model.confidence) : 'OFF'}
      </div>
    </div>
  )
}

function ConfidenceSparkline({ history }: { history: number[] }) {
  return (
    <div className="flex items-end gap-[2px] h-[50px]">
      {history.map((v, i) => {
        const color = v > 85 ? 'bg-critical' : v > 70 ? 'bg-medium' : 'bg-low'
        return (
          <div
            key={i}
            className={cn('flex-1 rounded-[1px_1px_0_0] min-w-[4px]', color)}
            style={{
              height: `${v * 0.9}%`,
              opacity: 0.4 + (i / history.length) * 0.6,
            }}
          />
        )
      })}
    </div>
  )
}

function PulseIcon() {
  return (
    <svg width="9" height="9" viewBox="0 0 9 9" fill="none">
      <path d="M1 4.5h1.5L3.5 2l2 5L7 4.5H8" stroke="currentColor" strokeWidth="1" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  )
}
