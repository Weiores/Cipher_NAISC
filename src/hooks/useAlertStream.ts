import { useEffect, useRef, useCallback } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { useUIStore } from '@/store/ui.store'
import { SentinelMessageSchema } from '@/types/schemas'
import type { Alert, SentinelMessage } from '@/types'

// ─────────────────────────────────────────────
// WebSocket URL — reads from env, falls back to
// mock server for local development
// ─────────────────────────────────────────────
const WS_URL = import.meta.env.VITE_WS_URL ?? 'ws://localhost:8000/ws'

// ─────────────────────────────────────────────
// Query keys — centralised so components and
// hooks always reference the same cache keys
// ─────────────────────────────────────────────
export const QUERY_KEYS = {
  currentEvent: ['currentEvent'] as const,
  alerts: ['alerts'] as const,
  zones: ['zones'] as const,
  cameras: ['cameras'] as const,
  models: ['models'] as const,
  scenarios: ['scenarios'] as const,
  decision: ['decision'] as const,
  learningAgent: ['learningAgent'] as const,
  connectionStatus: ['connectionStatus'] as const,
} as const

type ConnectionStatus = 'connected' | 'disconnected' | 'reconnecting'

const RECONNECT_DELAYS = [1000, 2000, 4000, 8000, 16000] // exponential backoff

export function useAlertStream() {
  const queryClient = useQueryClient()
  const { addAuditEntry, addConfidenceReading, setMode } = useUIStore()
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectAttemptRef = useRef(0)
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const isMountedRef = useRef(true)

  const setConnectionStatus = useCallback(
    (status: ConnectionStatus) => {
      queryClient.setQueryData(QUERY_KEYS.connectionStatus, status)
    },
    [queryClient]
  )

  const handleMessage = useCallback(
    (event: MessageEvent) => {
      let raw: unknown
      try {
        raw = JSON.parse(event.data as string)
      } catch {
        console.error('[WS] Failed to parse message:', event.data)
        return
      }

      // Validate against Zod schema — reject malformed messages
      const result = SentinelMessageSchema.safeParse(raw)
      if (!result.success) {
        console.warn('[WS] Message failed schema validation:', result.error.flatten())
        return
      }

      const msg = raw as SentinelMessage

      // Update React Query cache — each panel reads its own slice
      queryClient.setQueryData(QUERY_KEYS.currentEvent, msg.fusedEvent)
      queryClient.setQueryData(QUERY_KEYS.decision, msg.decision)
      queryClient.setQueryData(QUERY_KEYS.scenarios, msg.scenarios)
      queryClient.setQueryData(QUERY_KEYS.zones, msg.zones)
      queryClient.setQueryData(QUERY_KEYS.models, msg.models)
      queryClient.setQueryData(QUERY_KEYS.learningAgent, msg.learningAgent)

      // Prepend new alerts to the alerts cache (newest first, capped at 200)
      const newAlerts = msg.alerts.filter((a) => !a.acknowledged)
      if (newAlerts.length > 0) {
        queryClient.setQueryData<Alert[]>(QUERY_KEYS.alerts, (prev) =>
          [...newAlerts, ...(prev ?? [])].slice(0, 200)
        )
      }

      // Sync system mode from server (in production the server sets this)
      if (msg.mode) setMode(msg.mode)

      // Feed confidence history sparkline
      if (msg.models?.weapon?.confidence) {
        const avg =
          (msg.models.weapon.confidence +
            msg.models.emotion.confidence +
            msg.models.tone.confidence) /
          3
        addConfidenceReading(Math.round(avg * 100))
      }
    },
    [queryClient, setMode, addConfidenceReading]
  )

  const connect = useCallback(() => {
    if (!isMountedRef.current) return
    if (wsRef.current?.readyState === WebSocket.OPEN) return

    setConnectionStatus('reconnecting')

    const ws = new WebSocket(WS_URL)
    wsRef.current = ws

    ws.onopen = () => {
      if (!isMountedRef.current) return
      reconnectAttemptRef.current = 0
      setConnectionStatus('connected')
      addAuditEntry({
        operatorId: 'system',
        operatorName: 'System',
        action: 'WS_CONNECTED',
        detail: `WebSocket connected to ${WS_URL}`,
      })
    }

    ws.onmessage = handleMessage

    ws.onclose = (event) => {
      if (!isMountedRef.current) return
      setConnectionStatus('disconnected')
      addAuditEntry({
        operatorId: 'system',
        operatorName: 'System',
        action: 'WS_DISCONNECTED',
        detail: `WebSocket closed — code ${event.code}`,
      })

      // Exponential backoff reconnect
      const delay =
        RECONNECT_DELAYS[
          Math.min(reconnectAttemptRef.current, RECONNECT_DELAYS.length - 1)
        ]
      reconnectAttemptRef.current += 1
      reconnectTimerRef.current = setTimeout(connect, delay)
    }

    ws.onerror = (error) => {
      console.error('[WS] Error:', error)
    }
  }, [handleMessage, setConnectionStatus, addAuditEntry])

  useEffect(() => {
    isMountedRef.current = true
    connect()

    return () => {
      isMountedRef.current = false
      if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current)
      wsRef.current?.close()
    }
  }, [connect])

  // Expose send function for operator actions
  const send = useCallback((payload: unknown) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(payload))
    } else {
      console.warn('[WS] Cannot send — not connected')
    }
  }, [])

  return { send }
}

// ─────────────────────────────────────────────
// Selector hooks — components use these, never
// directly access the query cache
// ─────────────────────────────────────────────

export function useConnectionStatus() {
  const queryClient = useQueryClient()
  return (
    queryClient.getQueryData<ConnectionStatus>(QUERY_KEYS.connectionStatus) ??
    'disconnected'
  )
}
