import { useQueryClient } from '@tanstack/react-query'
import { QUERY_KEYS } from '@/hooks/useAlertStream'
import { useUIStore } from '@/store/ui.store'
import { formatTimestamp, confidencePercent, cn } from '@/utils'
import type { LearningAgentStatus } from '@/types'

export function LearningAgentPanel() {
  const qc = useQueryClient()
  const agent = qc.getQueryData<LearningAgentStatus>(QUERY_KEYS.learningAgent)
  const { auditLog } = useUIStore()

  const badgeCls = !agent || !agent.isAvailable ? 'badge-degraded' : 'badge-info'
  const badgeText = !agent ? 'NO DATA' : !agent.isAvailable ? 'OFFLINE' : 'ACTIVE'

  const rows = agent
    ? [
        { key: 'Status', val: agent.status.toUpperCase(), cls: agent.isAvailable ? 'text-low' : 'text-text-4' },
        { key: 'Model version', val: agent.modelVersion, cls: '' },
        { key: 'Last trained', val: agent.lastTrained ? formatRelativeShort(agent.lastTrained) : '—', cls: '' },
        { key: 'Predictions made', val: agent.predictionCount.toString(), cls: agent.isAvailable ? 'text-low' : 'text-text-4' },
        { key: 'Accuracy (7d)', val: agent.isAvailable ? confidencePercent(agent.accuracySevenDay) : '—', cls: agent.isAvailable ? 'text-low' : 'text-text-4' },
        { key: 'SOP version', val: agent.sopVersion, cls: '' },
      ]
    : []

  return (
    <div className="panel flex flex-col" style={{ minHeight: 0 }}>

      {/* Learning agent section */}
      <div className="panel-header">
        <div className="panel-title">
          <BrainIcon />
          Learning agent
        </div>
        <span className={badgeCls}>{badgeText}</span>
      </div>

      <div className="p-2 flex flex-col gap-[5px]">
        {rows.length === 0 ? (
          <div className="font-mono text-[10px] text-text-4 text-center py-3 tracking-[0.05em]">
            AWAITING AGENT DATA
          </div>
        ) : (
          rows.map((r) => (
            <div key={r.key} className="flex justify-between items-center px-2 py-[5px] bg-bg-2 border border-border rounded-[3px]">
              <span className="font-mono text-[9px] text-text-3 tracking-[0.05em]">{r.key}</span>
              <span className={cn('font-mono text-[10px] font-bold', r.cls || 'text-text-1')}>{r.val}</span>
            </div>
          ))
        )}
      </div>

      {/* Audit log section */}
      <div className="panel-header" style={{ borderTop: '1px solid #1a2d3f', borderBottom: '1px solid #1a2d3f' }}>
        <div className="panel-title">
          <ClipboardIcon />
          Audit log
        </div>
        <span className="font-mono text-[9px] text-text-4">{auditLog.length} entries</span>
      </div>

      <div className="flex-1 overflow-y-auto min-h-0 max-h-[140px] p-1">
        {auditLog.length === 0 ? (
          <div className="font-mono text-[10px] text-text-4 text-center py-4 tracking-[0.05em]">
            NO AUDIT ENTRIES YET
          </div>
        ) : (
          auditLog.map((entry) => (
            <div key={entry.entryId} className="font-mono text-[9px] text-text-3 px-[6px] py-1 border-b border-white/[0.03] last:border-0 leading-[1.4]">
              <span className="text-text-4">[{formatTimestamp(entry.timestamp)}]</span>{' '}
              {entry.detail}
            </div>
          ))
        )}
      </div>

    </div>
  )
}

function formatRelativeShort(iso: string): string {
  const diffMs = Date.now() - new Date(iso).getTime()
  const hours = Math.floor(diffMs / 3600000)
  if (hours < 1) return 'Just now'
  if (hours < 24) return `${hours}h ago`
  return `${Math.floor(hours / 24)}d ago`
}

function BrainIcon() {
  return (
    <svg width="9" height="9" viewBox="0 0 9 9" fill="none">
      <path d="M4.5 1C3 1 2 2 2 3.5c0 .8.4 1.5 1 2V7h3V5.5c.6-.5 1-1.2 1-2C7 2 6 1 4.5 1Z" stroke="currentColor" strokeWidth="1" />
    </svg>
  )
}

function ClipboardIcon() {
  return (
    <svg width="9" height="9" viewBox="0 0 9 9" fill="none">
      <rect x="1.5" y="2" width="6" height="6.5" rx="1" stroke="currentColor" strokeWidth="1" />
      <path d="M3 2V1.5a1 1 0 012 0V2" stroke="currentColor" strokeWidth="1" />
      <line x1="3" y1="4.5" x2="6" y2="4.5" stroke="currentColor" strokeWidth="0.8" strokeLinecap="round" />
      <line x1="3" y1="6" x2="5" y2="6" stroke="currentColor" strokeWidth="0.8" strokeLinecap="round" />
    </svg>
  )
}
