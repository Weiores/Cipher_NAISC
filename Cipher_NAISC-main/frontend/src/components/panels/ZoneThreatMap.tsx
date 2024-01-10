import { useQueryClient } from '@tanstack/react-query'
import { QUERY_KEYS } from '@/hooks/useAlertStream'
import { useUIStore } from '@/store/ui.store'
import { threatLevelColor, threatLevelBg, cn } from '@/utils'
import type { ZoneState } from '@/types'

export function ZoneThreatMap() {
  const qc = useQueryClient()
  const { setSelectedZoneId, addAuditEntry, operator } = useUIStore()
  const zones = qc.getQueryData<ZoneState[]>(QUERY_KEYS.zones) ?? []

  const handleZoneClick = (zone: ZoneState) => {
    setSelectedZoneId(zone.zoneId)
    addAuditEntry({
      operatorId: operator?.id ?? 'unknown',
      operatorName: operator?.displayName ?? 'Unknown',
      action: 'ZONE_INSPECT',
      detail: `Zone "${zone.zoneName}" inspected — ${zone.threatLevel.toUpperCase()}`,
    })
  }

  return (
    <div className="panel">
      <div className="panel-header">
        <div className="panel-title">
          <GridIcon />
          Zone threat map
        </div>
        <span className="font-mono text-[9px] text-text-4">BUILDING FLOOR 1</span>
      </div>

      <div className="p-2">
        {zones.length === 0 ? (
          <div className="font-mono text-[10px] text-text-4 text-center py-6 tracking-[0.05em]">
            AWAITING ZONE DATA
          </div>
        ) : (
          <div
            className="grid gap-1"
            style={{
              gridTemplateColumns: 'repeat(3, 1fr)',
              gridTemplateRows: 'repeat(3, 1fr)',
              aspectRatio: '1.4',
            }}
          >
            {zones.map((zone) => (
              <ZoneTile
                key={zone.zoneId}
                zone={zone}
                onClick={() => handleZoneClick(zone)}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

function ZoneTile({ zone, onClick }: { zone: ZoneState; onClick: () => void }) {
  const isCritical = zone.threatLevel === 'critical'

  return (
    <div
      onClick={onClick}
      className={cn(
        'zone-tile transition-all cursor-pointer',
        threatLevelBg(zone.threatLevel),
        isCritical && 'animate-pulse-zone'
      )}
    >
      <div className="zone-name">{zone.zoneName}</div>
      <div className={cn('zone-level', threatLevelColor(zone.threatLevel))}>
        {zone.threatLevel === 'clear' ? 'CLEAR' : zone.threatLevel.toUpperCase().slice(0, 4)}
      </div>
      {zone.activeAlerts > 0 && (
        <div className="font-mono text-[8px] text-text-4">
          {zone.activeAlerts} alert{zone.activeAlerts > 1 ? 's' : ''}
        </div>
      )}
    </div>
  )
}

function GridIcon() {
  return (
    <svg width="9" height="9" viewBox="0 0 9 9" fill="none">
      <rect x="1" y="1" width="3" height="3" rx="0.5" stroke="currentColor" strokeWidth="1" />
      <rect x="5" y="1" width="3" height="3" rx="0.5" stroke="currentColor" strokeWidth="1" />
      <rect x="1" y="5" width="3" height="3" rx="0.5" stroke="currentColor" strokeWidth="1" />
      <rect x="5" y="5" width="3" height="3" rx="0.5" stroke="currentColor" strokeWidth="1" />
    </svg>
  )
}
