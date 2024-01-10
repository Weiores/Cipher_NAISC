import { useState } from 'react'
import { useIncidents, useSubmitResponse, useAgentReports } from '@/hooks/useApi'
import type { ApiIncident, OfficerResponsePayload } from '@/types/api'
import { formatLabel } from '@/utils/formatLabel'
import clsx from 'clsx'

const ACTIONS = ['DISPATCH_OFFICERS', 'INCREASE_SURVEILLANCE', 'ISSUE_VERBAL_WARNING', 'REVIEW_FOOTAGE', 'FALSE_ALARM']

function threatColour(action: string) {
  if (!action) return 'text-slate-400'
  if (action.includes('DISPATCH')) return 'text-red-400'
  if (action.includes('SURVEILLANCE')) return 'text-orange-400'
  return 'text-yellow-400'
}

function ResponseForm({ incident, onClose }: { incident: ApiIncident; onClose: () => void }) {
  const submit = useSubmitResponse(incident.id)
  const [form, setForm] = useState<OfficerResponsePayload>({
    officer_action: incident.officer_action ?? '',
    final_outcome: incident.final_outcome ?? '',
    feedback: incident.feedback ?? '',
    is_false_positive: incident.is_false_positive ?? false,
  })

  return (
    <form
      className="mt-3 p-3 bg-slate-700 rounded space-y-2 text-sm"
      onSubmit={e => {
        e.preventDefault()
        submit.mutate(form, { onSuccess: onClose })
      }}
    >
      <h4 className="font-semibold text-white">Log Officer Response</h4>

      <div>
        <label className="block text-slate-300 mb-1">Action Taken</label>
        <select
          className="w-full bg-slate-800 text-white rounded px-2 py-1 border border-slate-600"
          value={form.officer_action}
          onChange={e => setForm(f => ({ ...f, officer_action: e.target.value }))}
          required
        >
          <option value="">Select action…</option>
          {ACTIONS.map(a => (
            <option key={a} value={a}>{formatLabel(a)}</option>
          ))}
        </select>
      </div>

      <div>
        <label className="block text-slate-300 mb-1">Final Outcome</label>
        <input
          className="w-full bg-slate-800 text-white rounded px-2 py-1 border border-slate-600"
          placeholder="e.g. Resolved, De-escalated"
          value={form.final_outcome}
          onChange={e => setForm(f => ({ ...f, final_outcome: e.target.value }))}
          required
        />
      </div>

      <div>
        <label className="block text-slate-300 mb-1">Feedback / Notes</label>
        <textarea
          className="w-full bg-slate-800 text-white rounded px-2 py-1 border border-slate-600 resize-none"
          rows={2}
          value={form.feedback}
          onChange={e => setForm(f => ({ ...f, feedback: e.target.value }))}
        />
      </div>

      <label className="flex items-center gap-2 text-slate-300 cursor-pointer">
        <input
          type="checkbox"
          checked={form.is_false_positive}
          onChange={e => setForm(f => ({ ...f, is_false_positive: e.target.checked }))}
        />
        Mark as False Positive
      </label>

      <div className="flex gap-2 pt-1">
        <button
          type="submit"
          disabled={submit.isPending}
          className="px-3 py-1 bg-blue-600 hover:bg-blue-500 text-white rounded text-xs font-semibold disabled:opacity-50"
        >
          {submit.isPending ? 'Saving…' : 'Submit Response'}
        </button>
        <button type="button" onClick={onClose} className="px-3 py-1 bg-slate-600 hover:bg-slate-500 text-white rounded text-xs">
          Cancel
        </button>
      </div>
      {submit.isError && (
        <p className="text-red-400 text-xs">Failed to save. Please try again.</p>
      )}
    </form>
  )
}

const AGENT_LABELS: Record<string, [string, string]> = {
  threat_analyst: ['🔫', 'Threat Analyst'],
  psychologist:   ['🧠', 'Psychologist'],
  crowd_expert:   ['👥', 'Crowd Expert'],
  historian:      ['📋', 'Historian'],
  tactician:      ['🎯', 'Tactician'],
  coordinator:    ['📊', 'Coordinator'],
}

function AgentReportsPanel({ incidentId }: { incidentId: string }) {
  const [open, setOpen] = useState(false)
  const { data, isLoading, isError } = useAgentReports(incidentId, open)

  return (
    <div className="mt-2">
      <button
        className="text-xs text-blue-400 hover:text-blue-300 underline"
        onClick={() => setOpen(o => !o)}
      >
        {open ? '▲ Hide agent reports' : '▼ Show agent reports'}
      </button>

      {open && (
        <div className="mt-2 space-y-2">
          {isLoading && <p className="text-slate-500 text-xs">Loading agent reports…</p>}
          {isError && <p className="text-slate-500 text-xs">No agent reports for this incident.</p>}
          {data && Object.entries(data).map(([key, report]) => {
            const [icon, label] = AGENT_LABELS[key] ?? ['🤖', formatLabel(key)]
            return (
              <details key={key} className="bg-slate-800 rounded">
                <summary className="px-3 py-1.5 cursor-pointer text-xs font-semibold text-slate-300 select-none">
                  {icon} {label}
                </summary>
                <pre className="px-3 pb-2 text-xs text-slate-400 overflow-auto max-h-40 whitespace-pre-wrap">
                  {JSON.stringify(report, null, 2)}
                </pre>
              </details>
            )
          })}
        </div>
      )}
    </div>
  )
}

function IncidentRow({ incident }: { incident: ApiIncident }) {
  const [expanded, setExpanded] = useState(false)
  const [showForm, setShowForm] = useState(false)

  const detections = typeof incident.detections === 'string'
    ? (() => { try { return JSON.parse(incident.detections) } catch { return incident.detections } })()
    : incident.detections

  const shortId = `#${incident.id.slice(0, 8)}`
  const actionLabel = incident.recommended_action
    ? formatLabel(incident.recommended_action)
    : 'Pending'

  return (
    <div className={clsx('border rounded p-3 mb-2', incident.is_false_positive ? 'border-slate-600 opacity-60' : 'border-slate-700')}>
      <div
        className="flex items-center justify-between cursor-pointer"
        onClick={() => setExpanded(e => !e)}
      >
        <div className="flex items-center gap-3">
          <span className={clsx('text-xs font-bold', threatColour(incident.recommended_action ?? ''))}>
            {actionLabel}
          </span>
          <span className="text-slate-400 text-xs font-mono">{shortId}</span>
          {incident.is_false_positive && (
            <span className="text-xs bg-slate-700 px-1 rounded text-slate-400">False Positive</span>
          )}
          {!incident.officer_action && !incident.is_false_positive && (
            <span className="text-xs bg-yellow-900/50 px-1 rounded text-yellow-400">Pending</span>
          )}
          {incident.officer_action && (
            <span className="text-xs bg-green-900/50 px-1 rounded text-green-400">Responded</span>
          )}
        </div>
        <div className="flex items-center gap-3">
          <span className="text-slate-500 text-xs">{new Date(incident.created_at).toLocaleString()}</span>
          <span className="text-slate-500 text-sm">{expanded ? '▲' : '▼'}</span>
        </div>
      </div>

      {expanded && (
        <div className="mt-2 text-xs text-slate-300 space-y-1">
          <p><span className="text-slate-400">ID:</span> <span className="font-mono">{incident.id}</span></p>
          <p><span className="text-slate-400">Summary:</span> {incident.perception_summary || '—'}</p>
          <p><span className="text-slate-400">Confidence:</span> {(incident.reasoning_confidence * 100).toFixed(0)}%</p>
          {incident.officer_action && (
            <p><span className="text-slate-400">Officer Action:</span> {formatLabel(incident.officer_action)}</p>
          )}
          {incident.final_outcome && (
            <p><span className="text-slate-400">Outcome:</span> {formatLabel(incident.final_outcome)}</p>
          )}
          {detections && (
            <pre className="bg-slate-800 rounded p-2 text-xs overflow-auto max-h-24 text-slate-300">
              {JSON.stringify(detections, null, 2)}
            </pre>
          )}
          <AgentReportsPanel incidentId={incident.id} />
          {!showForm && (
            <button
              className="mt-2 px-3 py-1 bg-blue-700 hover:bg-blue-600 text-white rounded text-xs font-semibold"
              onClick={() => setShowForm(true)}
            >
              Log Response
            </button>
          )}
          {showForm && <ResponseForm incident={incident} onClose={() => setShowForm(false)} />}
        </div>
      )}
    </div>
  )
}

export function IncidentsTab() {
  const { data, isLoading, isError } = useIncidents(100)

  if (isLoading) return <p className="text-slate-400 p-4">Loading incidents…</p>
  if (isError) return <p className="text-red-400 p-4">Failed to load incidents. Is the API running on port 8000?</p>

  const incidents = data ?? []

  return (
    <div className="p-4">
      <h2 className="text-white font-bold text-lg mb-3">Incidents ({incidents.length})</h2>
      {incidents.length === 0 ? (
        <p className="text-slate-400 text-sm">No incidents recorded yet.</p>
      ) : (
        incidents.map(inc => <IncidentRow key={inc.id} incident={inc} />)
      )}
    </div>
  )
}
