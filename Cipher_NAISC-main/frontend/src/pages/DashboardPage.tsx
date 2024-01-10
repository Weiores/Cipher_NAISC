import { useState } from 'react'
import { useAlertStream } from '@/hooks/useAlertStream'
import { AppShell } from '@/components/layout/AppShell'
import { StatCardsRow } from '@/components/panels/StatCardsRow'
import { CameraGrid } from '@/components/panels/CameraGrid'
import { AlertFeed } from '@/components/panels/AlertFeed'
import { ScenarioPanel } from '@/components/panels/ScenarioPanel'
import { FusedEventPanel } from '@/components/panels/FusedEventPanel'
import { DecisionPanel } from '@/components/panels/DecisionPanel'
import { OperatorActionsPanel } from '@/components/panels/OperatorActionsPanel'
import { ModelHealthPanel } from '@/components/panels/ModelHealthPanel'
import { ZoneThreatMap } from '@/components/panels/ZoneThreatMap'
import { LearningAgentPanel } from '@/components/panels/LearningAgentPanel'
import { IncidentsTab } from './IncidentsTab'
import { AnalyticsTab } from './AnalyticsTab'
import { SimulationTab } from './SimulationTab'
import clsx from 'clsx'

const TABS = [
  { id: 'monitor', label: 'Live Monitor' },
  { id: 'incidents', label: 'Incidents' },
  { id: 'analytics', label: 'Analytics' },
  { id: 'simulation', label: 'Simulation' },
] as const

type TabId = typeof TABS[number]['id']

export function DashboardPage() {
  useAlertStream()
  const [activeTab, setActiveTab] = useState<TabId>('monitor')

  return (
    <AppShell>
      {/* Tab bar */}
      <div className="flex gap-1 px-2 pt-2 border-b border-slate-700">
        {TABS.map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={clsx(
              'px-4 py-2 text-sm font-medium rounded-t transition-colors',
              activeTab === tab.id
                ? 'bg-slate-700 text-white border-b-2 border-blue-500'
                : 'text-slate-400 hover:text-white hover:bg-slate-800',
            )}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab content */}
      {activeTab === 'monitor' && (
        <div className="p-2 grid gap-2">
          <StatCardsRow />
          <div className="grid gap-2" style={{ gridTemplateColumns: '2fr 1.2fr 1fr' }}>
            <CameraGrid />
            <AlertFeed />
            <ScenarioPanel />
          </div>
          <div className="grid gap-2" style={{ gridTemplateColumns: '1.4fr 1fr 0.8fr 0.8fr' }}>
            <FusedEventPanel />
            <DecisionPanel />
            <OperatorActionsPanel />
            <ModelHealthPanel />
          </div>
          <div className="grid gap-2" style={{ gridTemplateColumns: '1fr 1fr' }}>
            <ZoneThreatMap />
            <LearningAgentPanel />
          </div>
        </div>
      )}

      {activeTab === 'incidents' && <IncidentsTab />}
      {activeTab === 'analytics' && <AnalyticsTab />}
      {activeTab === 'simulation' && <SimulationTab />}
    </AppShell>
  )
}
