import { useQueryClient } from '@tanstack/react-query'
import { QUERY_KEYS } from '@/hooks/useAlertStream'
import { useOperatorActions } from '@/hooks/useOperatorActions'
import { useUIStore } from '@/store/ui.store'
import { threatLevelColor, formatTimestamp, cn } from '@/utils'
import type { Alert, ThreatLevel } from '@/types'

const LEVEL_BAR: Record<ThreatLevel, string> = {
  critical: 'bg-critical',
  high: 'bg-high',
  medium: 'bg-medium',
  low: 'bg-low',
  clear: 'bg-text-4',
}

export function AlertFeed() {
  const qc = useQueryClient()
  const { alertFilter } = useUIStore()
  const { acknowledgeAlert } = useOperatorActions()

  const allAlerts = qc.getQueryData<Alert[]>(QUERY_KEYS.alerts) ?? []
  const alerts = alertFilter === 'all'
    ? allAlerts
    : allAlerts.filter((a) => a.level === alertFilter)

  return (
    <div className="panel flex flex-col">
      <div className="panel-header">
        <div className="panel-title">
          <ListIcon />
          Alert feed
        </div>
        <span className="badge-live">LIVE</span>
      </div>

      {/* Filter row */}
      <FilterRow />

      {/* Alert list */}
      <div className="flex-1 overflow-y-auto max-h-[220px] p-1">
        {alerts.length === 0 ? (
          <div className="font-mono text-[10px] text-text-4 text-center py-6 tracking-[0.05em]">
            NO ACTIVE ALERTS
          </div>
        ) : (
          alerts.map((alert) => (
            <AlertItem
              key={alert.alertId}
              alert={alert}
              onAck={() => acknowledgeAlert(alert.alertId)}
            />
          ))
        )}
      </div>
    </div>
  )
}

function FilterRow() {
  const { alertFilter, setAlertFilter } = useUIStore()
  const levels: Array<ThreatLevel | 'all'> = ['all', 'critical', 'high', 'medium', 'low']

  return (
    <div className="flex gap-1 px-2 py-[5px] border-b border-border">
      {levels.map((l) => (
        <button
          key={l}
          onClick={() => setAlertFilter(l)}
          className={cn(
            'font-mono text-[8px] px-[6px] py-[2px] rounded-[2px] border cursor-pointer transition-all tracking-[0.05em] uppercase',
            alertFilter === l
              ? l === 'all'
                ? 'bg-cloud/10 text-cloud border-cloud/20'
                : `${LEVEL_BAR[l as ThreatLevel].replace('bg-', 'border-')}/30 text-${l === 'critical' ? 'critical' : l === 'high' ? 'high' : l === 'medium' ? 'medium' : 'low'} bg-transparent border`
              : 'bg-transparent text-text-4 border-border hover:text-text-3'
          )}
        >
          {l}
        </button>
      ))}
    </div>
  )
}

function AlertItem({ alert, onAck }: { alert: Alert; onAck: () => void }) {
  return (
    <div className={cn('alert-item', !alert.acknowledged && 'animate-slide-in')}>
      {/* Severity bar */}
      <div className={cn('alert-bar', LEVEL_BAR[alert.level])} />

      {/* Content */}
      <div>
        <div className="font-mono text-[10px] font-bold text-text-1 mb-[1px] tracking-[0.03em]">
          {alert.type}
        </div>
        <div className="text-[10px] text-text-3 leading-[1.3]">
          {alert.detail}
        </div>
        {!alert.acknowledged ? (
          <button
            onClick={(e) => { e.stopPropagation(); onAck() }}
            className="font-mono text-[8px] px-[5px] py-[1px] rounded-[2px] border border-border-2
                       bg-transparent text-text-4 cursor-pointer mt-[3px] block
                       hover:border-cloud hover:text-cloud transition-all tracking-[0.04em]"
          >
            ACK
          </button>
        ) : (
          <span className="font-mono text-[8px] text-low mt-[3px] block tracking-[0.04em]">
            ACKED · {alert.acknowledgedBy}
          </span>
        )}
      </div>

      {/* Meta */}
      <div className="text-right flex-shrink-0">
        <div className="font-mono text-[9px] text-text-4 mb-[2px]">
          {formatTimestamp(alert.timestamp)}
        </div>
        <div className={cn('font-mono text-[9px] font-bold', threatLevelColor(alert.level))}>
          {alert.level.toUpperCase()}
        </div>
        {alert.isDegraded && (
          <div className="font-mono text-[8px] text-degraded mt-[2px]">SOP</div>
        )}
      </div>
    </div>
  )
}

function ListIcon() {
  return (
    <svg width="9" height="9" viewBox="0 0 9 9" fill="none">
      <rect x="1" y="1" width="7" height="7" rx="1" stroke="currentColor" strokeWidth="1" />
      <line x1="1" y1="3.5" x2="8" y2="3.5" stroke="currentColor" strokeWidth="0.7" />
      <line x1="1" y1="6" x2="8" y2="6" stroke="currentColor" strokeWidth="0.7" />
    </svg>
  )
}
