/**
 * Mirrors the Python format_swarm_output() function.
 * Produces the same war-room breakdown rendered in the dashboard as is
 * sent to Telegram — no extra API call needed.
 */

const AGENT_META: Record<string, [string, string, string]> = {
  threat_analyst: ['🔫', 'SECURITY THREAT ANALYST',       'threat_level'],
  psychologist:   ['🧠', 'BEHAVIOURAL PSYCHOLOGIST',      'behaviour_risk'],
  crowd_expert:   ['👥', 'CROWD SAFETY EXPERT',           'crowd_risk'],
  historian:      ['📋', 'HISTORICAL INCIDENT ANALYST',   'pattern_match'],
  tactician:      ['🎯', 'TACTICAL RESPONSE SPECIALIST',  'urgency'],
}

type AnyReport = Record<string, unknown>

export function formatSwarmOutput(incident: AnyReport): string {
  const divider = '━'.repeat(36)
  const ts = String(incident.timestamp ?? incident.created_at ?? '').slice(11, 19)
  const id = String(incident.id ?? '').slice(0, 8)
  const reports = (incident.agent_reports ?? {}) as Record<string, AnyReport>
  const coordinator = (reports.coordinator ?? {}) as AnyReport

  const lines: string[] = [
    divider,
    '🚨 CIPHER NAISC — INCIDENT ANALYSIS',
    `Incident: ${id}  |  ${ts}`,
    divider,
    '',
  ]

  for (const [key, [icon, label, riskField]] of Object.entries(AGENT_META)) {
    const report = (reports[key] ?? {}) as AnyReport
    if (report.error) {
      lines.push(`${icon} ${label}   [UNAVAILABLE]`)
      lines.push(`"${report.error}"`)
    } else {
      const risk = String(report[riskField] ?? '—')
      const findings = String(report.findings ?? report.reasoning ?? 'No findings.')
      const paddedLabel = label.padEnd(32)
      lines.push(`${icon} ${paddedLabel} [${risk}]`)
      if (key === 'tactician') {
        lines.push(`" Primary: ${String(report.primary_action ?? '—')}`)
        if (report.secondary_action) {
          lines.push(`  Secondary: ${String(report.secondary_action)}`)
        }
        lines.push(` ${findings}"`)
      } else {
        lines.push(`"${findings}"`)
      }
    }
    lines.push('')
  }

  // Coordinator verdict — prefer coordinator report, fall back to incident fields
  const overallThreat = String(
    coordinator.overall_threat_level ?? incident.overall_threat_level ?? '—'
  )
  const confidence = Math.round(
    ((coordinator.confidence as number) ?? (incident.reasoning_confidence as number) ?? 0) * 100
  )
  const fpLikelihood = Math.round(
    ((coordinator.false_positive_likelihood as number) ?? 0) * 100
  )
  const summary = String(
    coordinator.incident_summary ?? incident.perception_summary ?? ''
  )
  const action = String(
    coordinator.final_action ?? incident.recommended_action ?? '—'
  )
  const secondaryAction = coordinator.secondary_action
    ? String(coordinator.secondary_action)
    : null
  const urgency = String(coordinator.urgency ?? incident.urgency ?? '—')

  lines.push(
    divider,
    '📊 COORDINATOR VERDICT',
    `Overall threat: ${overallThreat}  |  Confidence: ${confidence}%`,
    `False positive likelihood: ${fpLikelihood}%`,
    '',
    'SUMMARY:',
    summary,
    '',
    `ACTION: ${action}`,
  )
  if (secondaryAction) lines.push(`SECONDARY: ${secondaryAction}`)
  lines.push(`URGENCY: ${urgency}`, divider)

  return lines.join('\n')
}
