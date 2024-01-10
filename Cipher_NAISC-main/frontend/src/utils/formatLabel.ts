/**
 * Converts snake_case or SCREAMING_SNAKE_CASE action strings to Title Case.
 * "DISPATCH_OFFICERS" → "Dispatch Officers"
 * "MONITOR_ONLY"      → "Monitor Only"
 * "false_alarm"       → "False Alarm"
 */
export function formatLabel(raw: string | null | undefined): string {
  if (!raw) return '—'
  return raw
    .toLowerCase()
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase())
}
