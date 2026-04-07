import { useQueryClient } from '@tanstack/react-query'
import { QUERY_KEYS } from '@/hooks/useAlertStream'
import { confidencePercent, formatTimestamp, cn } from '@/utils'
import type { FusedEvent } from '@/types'

export function FusedEventPanel() {
  const qc = useQueryClient()
  const event = qc.getQueryData<FusedEvent>(QUERY_KEYS.currentEvent)

  const isDegraded = event?.mode === 'degraded'

  const fields = event
    ? [
        {
          key: 'Weapon',
          val: event.weapon.detected === 'none' ? 'NONE' : event.weapon.detected.toUpperCase(),
          cls: event.weapon.detected !== 'none' ? 'text-critical' : 'text-text-2',
          conf: event.weapon.detected !== 'none'
            ? `${confidencePercent(event.weapon.confidence)} confidence`
            : isDegraded ? 'Model offline' : 'No detection',
        },
        {
          key: 'Emotion',
          val: event.emotion.detected === 'none' ? 'N/A' : event.emotion.detected.toUpperCase(),
          cls: event.emotion.detected !== 'none' ? 'text-high' : 'text-text-3',
          conf: event.emotion.detected !== 'none'
            ? `${confidencePercent(event.emotion.confidence)} · ${event.emotion.subjectCount} subjects`
            : isDegraded ? 'Vision model offline' : 'No detection',
        },
        {
          key: 'Voice stress',
          val: event.voice.detected === 'none' ? 'N/A' : event.voice.detected.toUpperCase(),
          cls: event.voice.detected !== 'none' ? 'text-high' : 'text-text-3',
          conf: event.voice.detected !== 'none'
            ? `${confidencePercent(event.voice.confidence)} confidence`
            : isDegraded ? 'Audio model offline' : 'Calm',
        },
        {
          key: 'Behaviour',
          val: event.behaviour.detected.toUpperCase(),
          cls: event.behaviour.detected === 'anomalous' ? 'text-high' : 'text-text-2',
          conf: `${confidencePercent(event.behaviour.confidence)} · ${event.behaviour.patternType.replace('_', ' ')}`,
        },
        {
          key: 'Source',
          val: event.sourceIds[0] ?? '—',
          cls: 'text-text-2',
          conf: `Zone A · ${event.sourceIds.length} source${event.sourceIds.length > 1 ? 's' : ''}`,
        },
        {
          key: 'Duration',
          val: formatDuration(event.durationSeconds),
          cls: 'text-text-2',
          conf: 'Ongoing',
        },
      ]
    : []

  return (
    <div className="panel">
      <div className="panel-header">
        <div className="panel-title">
          <PlusIcon />
          Fused event summary
        </div>
        <span className="font-mono text-[9px] text-text-4">
          {event ? formatTimestamp(event.timestamp) : '--:--:--'}
        </span>
      </div>

      <div className="grid gap-[5px] p-2" style={{ gridTemplateColumns: '1fr 1fr' }}>
        {fields.length === 0
          ? <div className="col-span-2 font-mono text-[10px] text-text-4 text-center py-4 tracking-[0.05em]">AWAITING EVENT DATA</div>
          : fields.map((f) => (
              <div key={f.key} className="bg-bg-2 border border-border rounded-[3px] p-[8px_9px]">
                <div className="font-mono text-[8px] text-text-4 uppercase tracking-[0.08em] mb-1">
                  {f.key}
                </div>
                <div className={cn('font-mono text-[12px] font-bold tracking-[0.04em]', f.cls)}>
                  {f.val}
                </div>
                <div className="font-mono text-[8px] text-text-4 mt-[2px]">
                  {f.conf}
                </div>
              </div>
            ))
        }
      </div>
    </div>
  )
}

function formatDuration(seconds: number): string {
  const m = Math.floor(seconds / 60)
  const s = seconds % 60
  return `${m}:${s.toString().padStart(2, '0')}`
}

function PlusIcon() {
  return (
    <svg width="9" height="9" viewBox="0 0 9 9" fill="none">
      <circle cx="4.5" cy="4.5" r="3.5" stroke="currentColor" strokeWidth="1" />
      <path d="M3 4.5h3M4.5 3v3" stroke="currentColor" strokeWidth="1" strokeLinecap="round" />
    </svg>
  )
}
