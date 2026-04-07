import { create } from 'zustand'
import type { SystemMode, ThreatLevel, Operator, AuditEntry } from '@/types'

// ─────────────────────────────────────────────
// UI STATE — never mix with server/WebSocket state
// Server state lives in React Query cache only
// ─────────────────────────────────────────────

interface UIState {
  // System mode — set by orchestration layer in production,
  // manually toggleable in development/demo
  mode: SystemMode
  setMode: (mode: SystemMode) => void

  // Selected alert (for detail panel / focus)
  selectedAlertId: string | null
  setSelectedAlertId: (id: string | null) => void

  // Selected camera (for expanded view)
  selectedCameraId: string | null
  setSelectedCameraId: (id: string | null) => void

  // Selected zone (for zone detail)
  selectedZoneId: string | null
  setSelectedZoneId: (id: string | null) => void

  // Sidebar state
  sidebarOpen: boolean
  setSidebarOpen: (open: boolean) => void

  // Current operator (populated after auth)
  operator: Operator | null
  setOperator: (operator: Operator | null) => void

  // Session
  sessionStart: Date
  resetSession: () => void

  // Audit log (client-side, persisted to backend on each action)
  auditLog: AuditEntry[]
  addAuditEntry: (entry: Omit<AuditEntry, 'entryId' | 'timestamp'>) => void

  // Alert filter
  alertFilter: ThreatLevel | 'all'
  setAlertFilter: (filter: ThreatLevel | 'all') => void

  // Confidence history (rolling 20 readings for sparkline)
  confidenceHistory: number[]
  addConfidenceReading: (value: number) => void
}

export const useUIStore = create<UIState>((set, get) => ({
  mode: 'cloud',
  setMode: (mode) => {
    set({ mode })
    get().addAuditEntry({
      operatorId: get().operator?.id ?? 'system',
      operatorName: get().operator?.displayName ?? 'System',
      action: 'MODE_SWITCH',
      detail: `System mode changed to ${mode.toUpperCase()}`,
    })
  },

  selectedAlertId: null,
  setSelectedAlertId: (id) => set({ selectedAlertId: id }),

  selectedCameraId: null,
  setSelectedCameraId: (id) => set({ selectedCameraId: id }),

  selectedZoneId: null,
  setSelectedZoneId: (id) => set({ selectedZoneId: id }),

  sidebarOpen: false,
  setSidebarOpen: (open) => set({ sidebarOpen: open }),

  operator: null,
  setOperator: (operator) => set({ operator }),

  sessionStart: new Date(),
  resetSession: () => set({ sessionStart: new Date() }),

  auditLog: [],
  addAuditEntry: (entry) => {
    const full: AuditEntry = {
      ...entry,
      entryId: crypto.randomUUID(),
      timestamp: new Date().toISOString(),
    }
    set((state) => ({
      auditLog: [full, ...state.auditLog].slice(0, 100),
    }))
  },

  alertFilter: 'all',
  setAlertFilter: (filter) => set({ alertFilter: filter }),

  confidenceHistory: Array.from({ length: 20 }, () =>
    Math.floor(65 + Math.random() * 25)
  ),
  addConfidenceReading: (value) => {
    set((state) => ({
      confidenceHistory: [...state.confidenceHistory.slice(1), value],
    }))
  },
}))
