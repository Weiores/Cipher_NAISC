import { useState } from 'react'
import { useOperatorActions } from '@/hooks/useOperatorActions'
import { actionLabel, actionColor, cn } from '@/utils'
import type { DecisionAction } from '@/types'

const PRIMARY_ACTIONS: DecisionAction[] = ['dispatch', 'escalate', 'lockdown']
const SECONDARY_ACTIONS: DecisionAction[] = ['monitor', 'standby']

export function OperatorActionsPanel() {
  const { takeAction, markFalsePositive } = useOperatorActions()
  const [confirmedAction, setConfirmedAction] = useState<string | null>(null)

  const handleAction = (action: DecisionAction) => {
    takeAction(action)
    setConfirmedAction(action)
    setTimeout(() => setConfirmedAction(null), 2200)
  }

  const handleFalsePositive = () => {
    markFalsePositive('current')
    setConfirmedAction('false_positive')
    setTimeout(() => setConfirmedAction(null), 2200)
  }

  return (
    <div className="panel">
      <div className="panel-header">
        <div className="panel-title">
          <PersonIcon />
          Operator actions
        </div>
      </div>

      <div className="p-2 flex flex-col gap-[5px]">
        {/* Primary actions */}
        {PRIMARY_ACTIONS.map((action) => (
          <ActionButton
            key={action}
            action={action}
            confirmed={confirmedAction === action}
            onClick={() => handleAction(action)}
          />
        ))}

        <div className="h-px bg-border my-[2px]" />

        {/* Secondary actions */}
        {SECONDARY_ACTIONS.map((action) => (
          <ActionButton
            key={action}
            action={action}
            confirmed={confirmedAction === action}
            onClick={() => handleAction(action)}
          />
        ))}

        {/* False positive */}
        <button
          onClick={handleFalsePositive}
          className={cn(
            'action-btn',
            confirmedAction === 'false_positive'
              ? 'bg-low/10 text-low border-low/25'
              : 'bg-bg-2 text-text-3 border-border hover:text-text-2 hover:border-border-2'
          )}
        >
          {confirmedAction === 'false_positive' ? '✓ MARKED FALSE POSITIVE' : 'False positive'}
        </button>
      </div>
    </div>
  )
}

function ActionButton({
  action,
  confirmed,
  onClick,
}: {
  action: DecisionAction
  confirmed: boolean
  onClick: () => void
}) {
  return (
    <button
      onClick={onClick}
      className={cn(
        'action-btn',
        confirmed
          ? 'bg-low/10 text-low border-low/25'
          : actionColor(action)
      )}
    >
      {confirmed ? `✓ CONFIRMED` : actionLabel(action).toUpperCase()}
    </button>
  )
}

function PersonIcon() {
  return (
    <svg width="9" height="9" viewBox="0 0 9 9" fill="none">
      <circle cx="4.5" cy="3" r="1.5" stroke="currentColor" strokeWidth="1" />
      <path d="M1 8c0-1.9 1.6-3.5 3.5-3.5S8 6.1 8 8" stroke="currentColor" strokeWidth="1" fill="none" />
    </svg>
  )
}
