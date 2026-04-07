import { useCallback } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { useUIStore } from '@/store/ui.store'
import { useAlertStream, QUERY_KEYS } from './useAlertStream'
import type { Alert, DecisionAction, OperatorActionPayload } from '@/types'

export function useOperatorActions() {
  const { send } = useAlertStream()
  const { operator, addAuditEntry } = useUIStore()
  const queryClient = useQueryClient()

  const acknowledgeAlert = useCallback(
    (alertId: string) => {
      // Optimistic update — mark acked in cache immediately
      queryClient.setQueryData<Alert[]>(QUERY_KEYS.alerts, (prev) =>
        (prev ?? []).map((a) =>
          a.alertId === alertId
            ? {
                ...a,
                acknowledged: true,
                acknowledgedBy: operator?.displayName ?? 'Unknown',
                acknowledgedAt: new Date().toISOString(),
              }
            : a
        )
      )

      const payload: OperatorActionPayload = {
        actionType: 'acknowledge_alert',
        alertId,
        operatorId: operator?.id ?? 'unknown',
        timestamp: new Date().toISOString(),
      }

      send(payload)

      addAuditEntry({
        operatorId: operator?.id ?? 'unknown',
        operatorName: operator?.displayName ?? 'Unknown',
        action: 'ALERT_ACK',
        detail: `Alert ${alertId} acknowledged`,
        alertId,
      })
    },
    [queryClient, send, operator, addAuditEntry]
  )

  const markFalsePositive = useCallback(
    (alertId: string) => {
      queryClient.setQueryData<Alert[]>(QUERY_KEYS.alerts, (prev) =>
        (prev ?? []).filter((a) => a.alertId !== alertId)
      )

      const payload: OperatorActionPayload = {
        actionType: 'mark_false_positive',
        alertId,
        operatorId: operator?.id ?? 'unknown',
        timestamp: new Date().toISOString(),
      }

      send(payload)

      addAuditEntry({
        operatorId: operator?.id ?? 'unknown',
        operatorName: operator?.displayName ?? 'Unknown',
        action: 'FALSE_POSITIVE',
        detail: `Alert ${alertId} marked as false positive — added to training data`,
        alertId,
      })
    },
    [queryClient, send, operator, addAuditEntry]
  )

  const takeAction = useCallback(
    (action: DecisionAction, note?: string) => {
      const payload: OperatorActionPayload = {
        actionType: action,
        operatorId: operator?.id ?? 'unknown',
        note,
        timestamp: new Date().toISOString(),
      }

      send(payload)

      const actionLabels: Record<DecisionAction, string> = {
        dispatch: 'Security team dispatched',
        escalate: 'Escalated to supervisor',
        lockdown: 'Lockdown initiated',
        monitor: 'Monitoring mode active',
        standby: 'Standing by',
      }

      addAuditEntry({
        operatorId: operator?.id ?? 'unknown',
        operatorName: operator?.displayName ?? 'Unknown',
        action: `ACTION_${action.toUpperCase()}`,
        detail: actionLabels[action] + (note ? ` — ${note}` : ''),
      })
    },
    [send, operator, addAuditEntry]
  )

  return { acknowledgeAlert, markFalsePositive, takeAction }
}
