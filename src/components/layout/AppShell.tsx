import { useEffect, useState } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { useUIStore } from '@/store/ui.store'
import { useConnectionStatus, QUERY_KEYS } from '@/hooks/useAlertStream'
import { modeColor, modeBorderColor, cn } from '@/utils'
import type { SystemMode } from '@/types'

interface AppShellProps {
  children: React.ReactNode
}

export function AppShell({ children }: AppShellProps) {
  const { mode, operator, sessionStart } = useUIStore()
  const connectionStatus = useConnectionStatus()
  const [clock, setClock] = useState('')
  const [sessionTime, setSessionTime] = useState('00:00')

  // Live clock + session timer
  useEffect(() => {
    const tick = () => {
      const now = new Date()
      setClock(
        [now.getHours(), now.getMinutes(), now.getSeconds()]
          .map((n) => n.toString().padStart(2, '0'))
          .join(':')
      )
      const secs = Math.floor((Date.now() - sessionStart.getTime()) / 1000)
      const m = Math.floor(secs / 60).toString().padStart(2, '0')
      const s = (secs % 60).toString().padStart(2, '0')
      setSessionTime(`${m}:${s}`)
    }
    tick()
    const id = setInterval(tick, 1000)
    return () => clearInterval(id)
  }, [sessionStart])

  const connPillClass = {
    connected: 'bg-cloud/8 border-cloud/20 text-cloud',
    disconnected: 'bg-critical/8 border-critical/20 text-critical',
    reconnecting: 'bg-degraded/8 border-degraded/20 text-degraded',
  }[connectionStatus]

  const connText = {
    connected: mode === 'incident' ? 'INCIDENT STREAM' : 'CLOUD CONNECTED',
    disconnected: 'DISCONNECTED',
    reconnecting: 'RECONNECTING...',
  }[connectionStatus]

  const tickerText = {
    cloud: 'CLOUD MODE ACTIVE · ALL MODELS ONLINE · LEARNING AGENT v3.2.1 · WEAPON DETECTED CAM-01 · OPERATOR RESPONSE PENDING · INCIDENT INC-2847 OPEN',
    degraded: 'DEGRADED MODE — CLOUD CONNECTION LOST · SOP RULES ACTIVE · LEARNING AGENT UNAVAILABLE · MANUAL VERIFICATION REQUIRED · LOCAL PROCESSING ONLY',
    incident: '🔴 ACTIVE INCIDENT — ARMED THREAT DETECTED · FIREARM CAM-01 · KNIFE CAM-02 · MASS PANIC ZONES A–C · LOCKDOWN RECOMMENDED · EMERGENCY SERVICES NOTIFIED',
  }[mode]

  return (
    <div className="flex flex-col min-h-screen">

      {/* ── Topbar ─────────────────────────── */}
      <header className="h-11 bg-bg-1 border-b border-border-2 flex items-center px-3 gap-3 sticky top-0 z-50 flex-shrink-0">

        {/* Logo */}
        <div className="font-mono text-[14px] text-cloud tracking-[0.12em] whitespace-nowrap">
          SENTINEL<span className="text-text-3">/</span>OPS
        </div>

        <div className="w-px h-5 bg-border-2" />

        {/* Live ticker */}
        <div className="flex-1 overflow-hidden h-[18px] relative">
          <div className="font-mono text-[10px] text-text-3 tracking-[0.06em] whitespace-nowrap animate-ticker">
            {tickerText} &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; {tickerText}
          </div>
        </div>

        {/* Right side controls */}
        <div className="flex items-center gap-2 flex-shrink-0">

          {/* Mode buttons — dev only, in prod mode is set by orchestration */}
          {import.meta.env.DEV && <ModeToggle />}

          {/* Connection pill */}
          <div className={cn('flex items-center gap-[5px] font-mono text-[10px] px-2 py-[3px] rounded-[3px] border', connPillClass)}>
            <span className="w-[5px] h-[5px] rounded-full bg-current animate-blink" />
            <span>{connText}</span>
          </div>

          {/* Operator chip */}
          <div className="font-mono text-[10px] text-text-3 tracking-[0.04em]">
            OPR · {operator?.displayName ?? '—'}
          </div>

          {/* Clock */}
          <div className="font-mono text-[11px] text-text-2 tracking-[0.06em] min-w-[62px] text-right tabular-nums">
            {clock}
          </div>
        </div>
      </header>

      {/* ── Degraded banner ────────────────── */}
      {mode === 'degraded' && (
        <div className="flex items-center gap-2 px-3 py-[6px] bg-degraded/7 border-b border-degraded/20 font-mono text-[10px] text-degraded tracking-[0.05em] flex-shrink-0">
          <WarningIcon />
          DEGRADED MODE — Cloud offline. Local SOP reasoning only. Learning agent unavailable. Confidence scores suppressed.
        </div>
      )}

      {/* ── Incident banner ────────────────── */}
      {mode === 'incident' && (
        <div className="flex items-center gap-2 px-3 py-[6px] bg-critical/7 border-b border-critical/25 font-mono text-[10px] text-critical tracking-[0.05em] animate-inc-pulse flex-shrink-0">
          <span className="w-[6px] h-[6px] rounded-full bg-critical animate-blink flex-shrink-0" />
          ACTIVE INCIDENT — Real-time threat detected. All models engaged. Operator action required immediately.
        </div>
      )}

      {/* ── Main content ───────────────────── */}
      <main className="flex-1 overflow-auto">
        {children}
      </main>

      {/* ── Status bar ─────────────────────── */}
      <footer className="flex items-center gap-3 px-3 py-[5px] bg-bg-1 border-t border-border font-mono text-[9px] text-text-4 tracking-[0.04em] flex-wrap flex-shrink-0">
        <span className={connectionStatus === 'connected' ? 'text-low' : 'text-critical'}>
          {connectionStatus === 'connected' ? '● WS CONNECTED' : connectionStatus === 'reconnecting' ? '⟳ RECONNECTING' : '○ DISCONNECTED'}
        </span>
        <span>·</span>
        <span>MODELS: WEAPON ● EMOTION ● AUDIO ●</span>
        <div className="flex-1" />
        <span>SESSION: {sessionTime}</span>
        <span>·</span>
        <span>OPR: {operator?.displayName ?? '—'}</span>
        <span>·</span>
        <span>BUILD: v0.1.0-dev</span>
      </footer>

    </div>
  )
}

// ── Mode toggle (dev only) ───────────────────
function ModeToggle() {
  const { mode, setMode } = useUIStore()
  const modes: SystemMode[] = ['cloud', 'degraded', 'incident']

  const activeClass: Record<SystemMode, string> = {
    cloud: 'text-cloud bg-cloud/8',
    degraded: 'text-degraded bg-degraded/8',
    incident: 'text-critical bg-critical/8',
  }

  return (
    <div className="flex border border-border-2 rounded-[4px] overflow-hidden font-mono text-[10px]">
      {modes.map((m) => (
        <button
          key={m}
          onClick={() => setMode(m)}
          className={cn(
            'px-[10px] py-1 cursor-pointer border-none tracking-[0.05em] transition-all',
            mode === m ? activeClass[m] : 'text-text-4 bg-transparent hover:text-text-2'
          )}
        >
          {m.toUpperCase()}
        </button>
      ))}
    </div>
  )
}

function WarningIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 14 14" fill="none" className="flex-shrink-0">
      <path d="M7 1L13 12H1L7 1Z" stroke="currentColor" strokeWidth="1.2" />
      <line x1="7" y1="5" x2="7" y2="8.5" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" />
      <circle cx="7" cy="10.5" r="0.6" fill="currentColor" />
    </svg>
  )
}
