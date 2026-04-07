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

export function DashboardPage() {
  // Initialise WebSocket connection — fires once, lives for the session
  useAlertStream()

  return (
    <AppShell>
      <div className="p-2 grid gap-2">

        {/* Row 1 — Stat cards */}
        <StatCardsRow />

        {/* Row 2 — Cameras + Alert feed + Scenarios */}
        <div className="grid gap-2" style={{ gridTemplateColumns: '2fr 1.2fr 1fr' }}>
          <CameraGrid />
          <AlertFeed />
          <ScenarioPanel />
        </div>

        {/* Row 3 — Fused event + Decision + Actions + Model health */}
        <div className="grid gap-2" style={{ gridTemplateColumns: '1.4fr 1fr 0.8fr 0.8fr' }}>
          <FusedEventPanel />
          <DecisionPanel />
          <OperatorActionsPanel />
          <ModelHealthPanel />
        </div>

        {/* Row 4 — Zone map + Learning agent + Audit log */}
        <div className="grid gap-2" style={{ gridTemplateColumns: '1fr 1fr' }}>
          <ZoneThreatMap />
          <LearningAgentPanel />
        </div>

      </div>
    </AppShell>
  )
}
