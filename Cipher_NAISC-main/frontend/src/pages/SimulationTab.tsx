import { useState, useRef } from 'react'
import axios from 'axios'
import type { ApiIncident } from '@/types/api'

const API_BASE = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

interface FrameResult {
  frame_id: number
  timestamp: string
  is_danger: boolean
  danger_reasons: string[]
  detections: unknown[]
}

function exportCsv(frames: FrameResult[], filename = 'simulation_report.csv') {
  const rows = [
    ['frame_id', 'timestamp', 'is_danger', 'danger_reasons', 'detections'],
    ...frames.map(f => [
      f.frame_id,
      f.timestamp,
      f.is_danger,
      f.danger_reasons.join(' | '),
      JSON.stringify(f.detections).replace(/,/g, ';'),
    ]),
  ]
  const csv = rows.map(r => r.join(',')).join('\n')
  const blob = new Blob([csv], { type: 'text/csv' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.click()
  URL.revokeObjectURL(url)
}

export function SimulationTab() {
  const fileRef = useRef<HTMLInputElement>(null)
  const [uploading, setUploading] = useState(false)
  const [progress, setProgress] = useState('')
  const [frames, setFrames] = useState<FrameResult[]>([])
  const [error, setError] = useState('')

  async function handleUpload() {
    const file = fileRef.current?.files?.[0]
    if (!file) return

    setUploading(true)
    setError('')
    setFrames([])
    setProgress('Uploading video…')

    const formData = new FormData()
    formData.append('file', file)

    try {
      // Try the perception-layer video API server first
      const uploadRes = await axios.post(
        `${API_BASE}/upload-video`,
        formData,
        { headers: { 'Content-Type': 'multipart/form-data' } }
      )
      const videoId: string = uploadRes.data.video_id ?? uploadRes.data.id ?? file.name

      setProgress('Processing frames…')
      // Poll for results
      let attempts = 0
      while (attempts < 120) {
        await new Promise(r => setTimeout(r, 2000))
        attempts++
        try {
          const statusRes = await axios.get(`${API_BASE}/results/${videoId}`)
          const data = statusRes.data
          if (data.status === 'completed') {
            setFrames(data.detections ?? [])
            setProgress(`Done — ${data.processed_frames ?? 0} frames analysed.`)
            break
          } else if (data.status === 'error') {
            setError(data.error ?? 'Processing failed.')
            break
          } else {
            setProgress(`Processing… ${data.progress_pct ?? 0}%`)
          }
        } catch {
          // Polling may fail transiently
        }
      }
      if (attempts >= 120) setError('Processing timed out after 4 minutes.')
    } catch (err: unknown) {
      const msg = axios.isAxiosError(err)
        ? err.response?.data?.detail ?? err.message
        : String(err)
      setError(`Upload failed: ${msg}. Make sure the API is running.`)
    } finally {
      setUploading(false)
    }
  }

  const dangerFrames = frames.filter(f => f.is_danger)

  return (
    <div className="p-4 space-y-4">
      <h2 className="text-white font-bold text-lg">Simulation / Training</h2>

      {/* Testing notice */}
      <div className="border-l-4 border-blue-500 bg-blue-950/40 px-4 py-3 rounded-r text-sm text-blue-200">
        ℹ️ Upload a video file to test the detection pipeline. Results are for demonstration purposes only.
      </div>

      <p className="text-slate-400 text-sm">
        Upload a video file for offline analysis. The system will process each frame and
        display detection results below.
      </p>

      <div className="flex items-center gap-3">
        <input
          ref={fileRef}
          type="file"
          accept="video/*"
          className="text-slate-300 text-sm file:mr-3 file:py-1.5 file:px-3 file:rounded file:border-0 file:bg-slate-700 file:text-white file:text-xs file:cursor-pointer hover:file:bg-slate-600"
        />
        <button
          onClick={handleUpload}
          disabled={uploading}
          className="px-4 py-1.5 bg-blue-600 hover:bg-blue-500 text-white rounded text-sm font-semibold disabled:opacity-50"
        >
          {uploading ? 'Processing…' : 'Analyse Video'}
        </button>
        {frames.length > 0 && (
          <button
            onClick={() => exportCsv(frames)}
            className="px-3 py-1.5 bg-green-700 hover:bg-green-600 text-white rounded text-sm font-semibold"
          >
            Export CSV
          </button>
        )}
      </div>

      {progress && <p className="text-slate-400 text-xs">{progress}</p>}
      {error && <p className="text-red-400 text-sm">{error}</p>}

      {frames.length > 0 && (
        <div>
          <p className="text-slate-300 text-sm mb-2">
            {frames.length} frames analysed · <span className="text-red-400">{dangerFrames.length} danger frames</span>
          </p>

          <div className="overflow-auto max-h-96">
            <table className="w-full text-xs text-left border-collapse">
              <thead>
                <tr className="bg-slate-800 text-slate-400">
                  <th className="p-2 border border-slate-700">Frame</th>
                  <th className="p-2 border border-slate-700">Time</th>
                  <th className="p-2 border border-slate-700">Danger</th>
                  <th className="p-2 border border-slate-700">Reasons</th>
                  <th className="p-2 border border-slate-700">Detections</th>
                </tr>
              </thead>
              <tbody>
                {frames.map(f => (
                  <tr
                    key={f.frame_id}
                    className={f.is_danger ? 'bg-red-950/40' : 'bg-slate-900/40'}
                  >
                    <td className="p-2 border border-slate-700 text-slate-300">{f.frame_id}</td>
                    <td className="p-2 border border-slate-700 text-slate-400">
                      {new Date(f.timestamp).toLocaleTimeString()}
                    </td>
                    <td className="p-2 border border-slate-700">
                      {f.is_danger
                        ? <span className="text-red-400 font-bold">⚠ YES</span>
                        : <span className="text-green-500">CLEAR</span>}
                    </td>
                    <td className="p-2 border border-slate-700 text-orange-300">
                      {f.danger_reasons.join(', ') || '—'}
                    </td>
                    <td className="p-2 border border-slate-700 text-slate-400 max-w-xs truncate">
                      {JSON.stringify(f.detections)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}
