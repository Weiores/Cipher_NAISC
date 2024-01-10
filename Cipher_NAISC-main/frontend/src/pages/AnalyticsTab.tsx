import { useAnalytics, useMLStats, useFeedbackSummary, useMLRetrain } from '@/hooks/useApi'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend, LineChart, Line, CartesianGrid,
} from 'recharts'
import { formatLabel } from '@/utils/formatLabel'

const COLOURS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#06b6d4']

function StatCard({ label, value, sub }: { label: string; value: string | number; sub?: string }) {
  return (
    <div className="bg-slate-800 rounded p-4 flex flex-col gap-1">
      <p className="text-slate-400 text-xs uppercase tracking-wider">{label}</p>
      <p className="text-white text-2xl font-bold">{value}</p>
      {sub && <p className="text-slate-500 text-xs">{sub}</p>}
    </div>
  )
}

export function AnalyticsTab() {
  const { data, isLoading, isError } = useAnalytics()
  const { data: mlStats } = useMLStats()
  const { data: fbSummary } = useFeedbackSummary()
  const retrain = useMLRetrain()

  if (isLoading) return <p className="text-slate-400 p-4">Loading analytics…</p>
  if (isError) return <p className="text-red-400 p-4">Failed to load analytics. Is the API running on port 8000?</p>

  const d = data!
  const accuracy = d.learning?.recommendation_accuracy ?? 0
  const fpRate = (d.false_positive_rate * 100).toFixed(1)

  // Most common threat type — formatted for display
  const topAction = formatLabel(d.most_common_threat) || '—'

  const dailyData = (d.incidents_per_day ?? []).map(row => ({
    day: row.day.slice(5),   // MM-DD
    count: row.count,
  }))

  const pieData = (d.action_distribution ?? []).map(row => ({
    name: formatLabel(row.recommended_action) || 'Unknown',
    value: row.count,
  }))

  return (
    <div className="p-4 space-y-6">
      <h2 className="text-white font-bold text-lg">Analytics</h2>

      {/* Stat cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <StatCard label="Total Incidents" value={d.total_incidents} />
        <StatCard label="False Positive Rate" value={`${fpRate}%`} sub={`${d.false_positive_count} incidents`} />
        <StatCard label="Most Common Threat" value={topAction} />
        <StatCard label="Recommendation Accuracy" value={`${(accuracy * 100).toFixed(0)}%`} sub={`${d.learning?.responded ?? 0} responded`} />
      </div>

      {/* Bar chart: incidents per day */}
      <div className="bg-slate-800 rounded p-4">
        <h3 className="text-slate-300 text-sm font-semibold mb-3">Incidents per Day (last 30 days)</h3>
        {dailyData.length === 0 ? (
          <p className="text-slate-500 text-sm">No data yet.</p>
        ) : (
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={dailyData} margin={{ top: 4, right: 8, left: -20, bottom: 0 }}>
              <XAxis dataKey="day" tick={{ fill: '#94a3b8', fontSize: 11 }} />
              <YAxis allowDecimals={false} tick={{ fill: '#94a3b8', fontSize: 11 }} />
              <Tooltip
                contentStyle={{ background: '#1e293b', border: 'none', borderRadius: 6 }}
                labelStyle={{ color: '#e2e8f0' }}
                itemStyle={{ color: '#60a5fa' }}
              />
              <Bar dataKey="count" fill="#3b82f6" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        )}
      </div>

      {/* Pie chart: action distribution */}
      <div className="bg-slate-800 rounded p-4">
        <h3 className="text-slate-300 text-sm font-semibold mb-3">Recommended Action Distribution</h3>
        {pieData.length === 0 ? (
          <p className="text-slate-500 text-sm">No data yet.</p>
        ) : (
          <ResponsiveContainer width="100%" height={220}>
            <PieChart>
              <Pie
                data={pieData}
                cx="50%"
                cy="50%"
                outerRadius={80}
                dataKey="value"
                label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                labelLine={false}
              >
                {pieData.map((_, i) => (
                  <Cell key={i} fill={COLOURS[i % COLOURS.length]} />
                ))}
              </Pie>
              <Legend wrapperStyle={{ color: '#94a3b8', fontSize: 12 }} />
              <Tooltip
                contentStyle={{ background: '#1e293b', border: 'none', borderRadius: 6 }}
                itemStyle={{ color: '#e2e8f0' }}
              />
            </PieChart>
          </ResponsiveContainer>
        )}
      </div>

      {/* ── Model Performance ─────────────────────────────────────────── */}
      <div>
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-3">
            <h2 className="text-white font-bold text-lg">Model Performance</h2>
            {/* ML status badge */}
            {(() => {
              const trained = d.ml_stats?.samples_trained ?? mlStats?.samples_seen ?? 0
              const active  = d.ml_stats?.is_active ?? mlStats?.is_fitted ?? false
              if (active && trained >= 10) {
                return (
                  <span className="text-xs bg-emerald-900 text-emerald-300 border border-emerald-600 px-2 py-0.5 rounded-full font-mono">
                    ● ML Model Active
                  </span>
                )
              }
              return (
                <span className="text-xs bg-slate-700 text-slate-400 border border-slate-600 px-2 py-0.5 rounded-full font-mono">
                  ○ Warming Up ({trained} samples)
                </span>
              )
            })()}
          </div>
          <button
            onClick={() => retrain.mutate()}
            disabled={retrain.isPending}
            className="text-xs bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white px-3 py-1 rounded font-mono"
          >
            {retrain.isPending ? 'Retraining…' : '⚡ Retrain Model'}
          </button>
        </div>

        {/* ML stat cards — inline data from /analytics takes priority, fallback to /ml-stats */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4">
          <StatCard
            label="Total Feedback"
            value={fbSummary?.total ?? '—'}
            sub={`${fbSummary?.confirmed ?? 0} confirmed · ${fbSummary?.false_alarm ?? 0} false alarms`}
          />
          <StatCard
            label="Trained on Incidents"
            value={d.ml_stats?.samples_trained ?? mlStats?.samples_seen ?? '—'}
            sub={d.ml_stats?.is_active ? 'Model is active' : 'Not yet trained'}
          />
          <StatCard
            label="False Positives Prevented"
            value={d.false_positives_prevented ?? '—'}
            sub={`of ${fbSummary?.false_alarm ?? 0} total false alarms`}
          />
          <StatCard
            label="Rec. Approval"
            value={fbSummary ? `${((fbSummary.recommendation_approval_rate) * 100).toFixed(1)}%` : '—'}
            sub={`${fbSummary?.good_rec ?? 0} good · ${fbSummary?.bad_rec ?? 0} bad`}
          />
        </div>

        {/* Model accuracy progress bar */}
        {(() => {
          const acc = d.ml_stats?.accuracy ?? mlStats?.accuracy ?? 0
          const accPct = Math.round(acc * 100)
          const trained = d.ml_stats?.samples_trained ?? mlStats?.samples_seen ?? 0
          return trained > 0 ? (
            <div className="bg-slate-800 rounded p-4 mb-4">
              <div className="flex justify-between items-center mb-2">
                <h3 className="text-slate-300 text-sm font-semibold">Model Accuracy</h3>
                <span className="text-white font-bold text-lg">{accPct}%</span>
              </div>
              <div className="w-full bg-slate-700 rounded-full h-3">
                <div
                  className={`h-3 rounded-full transition-all duration-700 ${
                    accPct >= 80 ? 'bg-emerald-500' : accPct >= 60 ? 'bg-yellow-500' : 'bg-red-500'
                  }`}
                  style={{ width: `${accPct}%` }}
                />
              </div>
              <p className="text-slate-500 text-xs mt-1">
                Trained on {trained} feedback samples
                {d.ml_stats?.last_updated
                  ? ` · Last updated ${new Date(d.ml_stats.last_updated).toLocaleString()}`
                  : ''}
              </p>
            </div>
          ) : null
        })()}


        {/* Accuracy over time line chart */}
        <div className="bg-slate-800 rounded p-4">
          <h3 className="text-slate-300 text-sm font-semibold mb-3">Model Accuracy Over Time</h3>
          {!mlStats?.accuracy_history?.length ? (
            <p className="text-slate-500 text-sm">
              {mlStats?.is_fitted
                ? 'No accuracy history yet — submit more feedback to build a trend.'
                : 'Model not trained yet. Officers can provide feedback via Telegram after each alert.'}
            </p>
          ) : (
            <ResponsiveContainer width="100%" height={200}>
              <LineChart
                data={mlStats.accuracy_history.map(p => ({
                  time: p.timestamp.slice(11, 16),
                  accuracy: Math.round(p.accuracy * 1000) / 10,
                  samples: p.samples_seen,
                }))}
                margin={{ top: 4, right: 8, left: -20, bottom: 0 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis dataKey="time" tick={{ fill: '#94a3b8', fontSize: 11 }} />
                <YAxis domain={[0, 100]} unit="%" tick={{ fill: '#94a3b8', fontSize: 11 }} />
                <Tooltip
                  contentStyle={{ background: '#1e293b', border: 'none', borderRadius: 6 }}
                  labelStyle={{ color: '#e2e8f0' }}
                  formatter={(v: number, name: string) =>
                    name === 'accuracy' ? [`${v}%`, 'Accuracy'] : [v, 'Samples']}
                />
                <Line
                  type="monotone"
                  dataKey="accuracy"
                  stroke="#3b82f6"
                  strokeWidth={2}
                  dot={{ fill: '#3b82f6', r: 3 }}
                  activeDot={{ r: 5 }}
                />
              </LineChart>
            </ResponsiveContainer>
          )}
          {mlStats?.last_updated && (
            <p className="text-slate-600 text-xs mt-2">
              Last updated: {new Date(mlStats.last_updated).toLocaleString()}
            </p>
          )}
        </div>
      </div>
    </div>
  )
}
