import { useEffect, useRef, useState } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { QUERY_KEYS } from '@/hooks/useAlertStream'
import { cn } from '@/utils'
import type { CameraState } from '@/types'

const API_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

type FeedMode = 'standby' | 'webcam' | 'video' | 'demo'

export function CameraGrid() {
  const qc = useQueryClient()
  const cameras = qc.getQueryData<CameraState[]>(QUERY_KEYS.cameras) ?? []
  const cam01 = cameras.find((c) => c.cameraId === 'CAM-01')
  const hasThreat = cam01?.threatLevel === 'critical' || cam01?.threatLevel === 'high'

  const [feedMode, setFeedMode] = useState<FeedMode>('standby')

  return (
    <div className="panel">
      <div className="panel-header">
        <div className="panel-title">
          <CameraIcon />
          Live cameras
        </div>
        <span
          className={cn(
            'font-mono text-[8px] px-[5px] py-[1px] rounded-[2px] font-bold tracking-[0.04em]',
            feedMode !== 'standby'
              ? 'bg-green-700/80 text-green-300'
              : 'bg-white/10 text-white/40'
          )}
        >
          {feedMode !== 'standby' ? 'LIVE' : 'STANDBY'}
        </span>
      </div>

      <div className="p-2">
        <LiveCameraFeed
          label="CAM-01 · ENTRANCE"
          hasThreat={hasThreat ?? false}
          feedMode={feedMode}
          onModeChange={setFeedMode}
        />
      </div>
    </div>
  )
}

interface LiveCameraFeedProps {
  label: string
  hasThreat: boolean
  feedMode: FeedMode
  onModeChange: (mode: FeedMode) => void
}

function LiveCameraFeed({ label, hasThreat, feedMode, onModeChange }: LiveCameraFeedProps) {
  const [src, setSrc] = useState('')
  const [offline, setOffline] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [uploading, setUploading] = useState(false)
  const [uploadError, setUploadError] = useState('')
  const [demoAvailable, setDemoAvailable] = useState(false)
  const [showFileInput, setShowFileInput] = useState(false)
  const fileInputRef = useRef<HTMLInputElement | null>(null)
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null)

  // Check if demo video exists on mount
  useEffect(() => {
    fetch(`${API_URL}/feed/demo-available`)
      .then(r => r.json())
      .then(d => setDemoAvailable(d.available))
      .catch(() => {})
  }, [])

  function startFramePolling() {
    if (intervalRef.current) clearInterval(intervalRef.current)
    setSrc(`${API_URL}/stream/frame?t=${Date.now()}`)
    intervalRef.current = setInterval(() => {
      setSrc(`${API_URL}/stream/frame?t=${Date.now()}`)
    }, 500)
  }

  async function activateWebcam() {
    try { await fetch(`${API_URL}/stream/start`) } catch { /* ignore */ }
    onModeChange('webcam')
    startFramePolling()
  }

  async function stopFeed() {
    if (feedMode === 'webcam') {
      try { await fetch(`${API_URL}/stream/stop`) } catch { /* ignore */ }
    } else {
      try { await fetch(`${API_URL}/feed/stop-video`, { method: 'POST' }) } catch { /* ignore */ }
    }
    if (intervalRef.current) clearInterval(intervalRef.current)
    setSrc('')
    onModeChange('standby')
    setShowFileInput(false)
    setUploadProgress(0)
    setUploadError('')
  }

  async function uploadAndStartVideo(file: File) {
    setUploading(true)
    setUploadProgress(0)
    setUploadError('')

    const form = new FormData()
    form.append('file', file)

    try {
      // POST /feed/upload saves the file AND auto-starts playback in one call.
      // Use XHR so we get upload progress events.
      await new Promise<void>((resolve, reject) => {
        const xhr = new XMLHttpRequest()
        xhr.open('POST', `${API_URL}/feed/upload`)
        xhr.upload.onprogress = e => {
          if (e.lengthComputable)
            setUploadProgress(Math.round((e.loaded / e.total) * 100))
        }
        xhr.onload = () =>
          xhr.status < 400 ? resolve() : reject(new Error(`Server error ${xhr.status}`))
        xhr.onerror = () => reject(new Error('Network error — check backend is running'))
        xhr.send(form)
      })

      setUploadProgress(100)
      // Give the video processor 1.5 s to open the file and buffer the first frame
      await new Promise(r => setTimeout(r, 1500))
      setUploading(false)
      setShowFileInput(false)
      onModeChange('video')
      startFramePolling()
    } catch (err) {
      setUploadError(err instanceof Error ? err.message : 'Upload failed')
      setUploading(false)
      setUploadProgress(0)
    }
  }

  async function startDemoVideo() {
    try {
      await fetch(`${API_URL}/feed/start-video`, { method: 'POST' })
      onModeChange('demo')
      startFramePolling()
    } catch { /* ignore — feed/start-video may 404 if no demo video */ }
  }

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    if (file) uploadAndStartVideo(file)
  }

  useEffect(() => {
    return () => { if (intervalRef.current) clearInterval(intervalRef.current) }
  }, [])

  const tagCls = hasThreat ? 'bg-red-600 text-white' : 'bg-cyan-900/60 text-cyan-300 border border-cyan-700/40'
  const isActive = feedMode !== 'standby'

  // ── STANDBY ──────────────────────────────────────────────────────────────
  if (!isActive) {
    return (
      <div
        className="relative bg-[#111] rounded-[4px] overflow-hidden border border-border flex flex-col items-center justify-center gap-[5px] px-3"
        style={{ aspectRatio: '16/9' }}
      >
        {/* STANDBY badge */}
        <div className="absolute top-1 right-1 font-mono text-[8px] px-[5px] py-[1px] rounded-[2px] font-bold tracking-[0.04em] bg-white/10 text-white/40">
          STANDBY
        </div>

        {/* Option A — Webcam */}
        <button
          onClick={activateWebcam}
          className="flex flex-col items-center gap-[3px] cursor-pointer hover:opacity-80 transition-opacity"
        >
          <span className="text-[18px] leading-none select-none">📷</span>
          <span className="font-mono text-[8px] text-white/60 tracking-[0.04em]">Activate Live Webcam</span>
          <span className="font-mono text-[7px] text-amber-400/70 tracking-[0.03em]">⚠️ For testing purposes only</span>
        </button>

        {/* Divider */}
        <div className="flex items-center gap-[6px] w-full max-w-[120px]">
          <div className="flex-1 h-px bg-white/10" />
          <span className="font-mono text-[7px] text-white/20 tracking-[0.06em]">or</span>
          <div className="flex-1 h-px bg-white/10" />
        </div>

        {/* Option B — Upload video */}
        {uploadError ? (
          <div className="flex flex-col items-center gap-[3px] w-full max-w-[150px]">
            <span className="font-mono text-[7px] text-red-400 tracking-[0.03em] text-center leading-tight">
              ✕ {uploadError}
            </span>
            <button
              onClick={() => { setUploadError(''); setShowFileInput(true) }}
              className="font-mono text-[7px] px-[6px] py-[2px] rounded-[2px] bg-white/5 text-white/40 border border-white/10 hover:text-white/60 transition-colors cursor-pointer tracking-[0.03em]"
            >
              Try again
            </button>
          </div>
        ) : uploading ? (
          <div className="flex flex-col items-center gap-[4px] w-full max-w-[120px]">
            <span className="font-mono text-[8px] text-cyan-400/80 tracking-[0.04em]">
              {uploadProgress >= 100 ? 'Starting feed…' : 'Uploading…'}
            </span>
            <div className="w-full bg-white/10 rounded-full h-[4px] overflow-hidden">
              <div
                className="bg-cyan-500 h-full rounded-full transition-all duration-200"
                style={{ width: `${uploadProgress}%` }}
              />
            </div>
            <span className="font-mono text-[7px] text-white/30">{uploadProgress}%</span>
          </div>
        ) : showFileInput ? (
          <div className="flex flex-col items-center gap-[3px]">
            <span className="font-mono text-[7px] text-white/40 tracking-[0.04em]">Select a video file:</span>
            <input
              ref={fileInputRef}
              type="file"
              accept=".mp4,.avi,.mov,.mkv"
              onChange={handleFileChange}
              className="hidden"
            />
            <div className="flex gap-[4px]">
              <button
                onClick={() => fileInputRef.current?.click()}
                className="font-mono text-[7px] px-[6px] py-[2px] rounded-[2px] bg-cyan-900/60 text-cyan-300 border border-cyan-700/40 hover:bg-cyan-800/60 transition-colors cursor-pointer tracking-[0.03em]"
              >
                Browse…
              </button>
              <button
                onClick={() => setShowFileInput(false)}
                className="font-mono text-[7px] px-[5px] py-[2px] rounded-[2px] bg-white/5 text-white/30 border border-white/10 hover:text-white/50 transition-colors cursor-pointer"
              >
                ✕
              </button>
            </div>
          </div>
        ) : (
          <div className="flex flex-col items-center gap-[3px]">
            <button
              onClick={() => setShowFileInput(true)}
              className="flex items-center gap-[4px] cursor-pointer hover:opacity-80 transition-opacity"
            >
              <span className="text-[14px] leading-none select-none">🎬</span>
              <div className="flex flex-col items-start">
                <span className="font-mono text-[8px] text-white/60 tracking-[0.04em]">Upload Video Feed</span>
                <span className="font-mono text-[7px] text-white/30 tracking-[0.03em]">Play a video file as feed</span>
              </div>
            </button>

            {demoAvailable && (
              <button
                onClick={startDemoVideo}
                className="font-mono text-[7px] px-[6px] py-[2px] rounded-[2px] bg-indigo-900/50 text-indigo-300 border border-indigo-700/40 hover:bg-indigo-800/50 transition-colors cursor-pointer tracking-[0.03em] mt-[2px]"
              >
                ▶ Use Demo Video
              </button>
            )}
          </div>
        )}

        {/* CAM label */}
        <div
          className="absolute bottom-1 left-[5px] font-mono text-[8px] text-white/40 bg-black/60 px-[5px] py-[1px] rounded-[2px] tracking-[0.04em]"
          title="This live feed is for demonstration and testing only."
        >
          {label}
        </div>
        <div className="absolute bottom-[5px] right-[5px]">
          <span className="w-[5px] h-[5px] rounded-full bg-green-500 animate-pulse block" />
        </div>
      </div>
    )
  }

  // ── ACTIVE FEED (webcam / video / demo) ──────────────────────────────────
  const bannerText =
    feedMode === 'demo'  ? '⚠️ Demo video feed — for testing purposes only' :
    feedMode === 'video' ? '⚠️ Uploaded video feed — for testing purposes only' :
                           '⚠️ Live feed active — for testing purposes only'

  return (
    <div
      className={cn(
        'relative bg-bg-0 rounded-[4px] overflow-hidden border transition-colors',
        hasThreat ? 'border-red-500 shadow-[inset_0_0_0_1px_rgba(255,34,68,0.3)]' : 'border-border'
      )}
      style={{ aspectRatio: '16/9' }}
    >
      {offline ? (
        <div className="w-full h-full flex items-center justify-center bg-bg-0 text-[9px] font-mono text-white/30">
          CAMERA OFFLINE
        </div>
      ) : (
        <img
          src={src}
          alt="camera feed"
          className="w-full h-full object-cover block"
          onError={() => setOffline(true)}
          onLoad={() => setOffline(false)}
        />
      )}

      {/* Warning banner */}
      <div className="absolute bottom-0 left-0 right-0 bg-amber-500/20 border-t border-amber-500/30 px-[6px] py-[2px]">
        <span className="font-mono text-[7px] text-amber-300 tracking-[0.03em]">{bannerText}</span>
      </div>

      {/* CAM label */}
      <div
        className="absolute bottom-[16px] left-[5px] font-mono text-[8px] text-white/50 bg-black/60 px-[5px] py-[1px] rounded-[2px] tracking-[0.04em]"
        title="For demonstration and testing only."
      >
        {label}
      </div>

      {/* Feed type badge */}
      <div className={cn(
        'absolute top-1 left-1 font-mono text-[8px] px-[5px] py-[1px] rounded-[2px] font-bold tracking-[0.04em]',
        feedMode === 'demo'  ? 'bg-purple-700/80 text-purple-200' :
        feedMode === 'video' ? 'bg-indigo-700/80 text-indigo-200' :
                               'bg-green-700/80 text-green-200'
      )}>
        {feedMode === 'demo' ? 'DEMO FEED' : feedMode === 'video' ? 'VIDEO FEED' : 'LIVE'}
      </div>

      {/* THREAT/CLEAR + Stop */}
      <div className="absolute top-1 right-1 flex items-center gap-[4px]">
        <div className={cn('font-mono text-[8px] px-[5px] py-[1px] rounded-[2px] font-bold tracking-[0.04em]', tagCls)}>
          {hasThreat ? 'THREAT' : 'CLEAR'}
        </div>
        <button
          onClick={stopFeed}
          className="font-mono text-[7px] px-[5px] py-[1px] rounded-[2px] bg-black/70 text-white/60 border border-white/20 hover:text-white hover:border-white/50 transition-colors tracking-[0.03em] cursor-pointer"
        >
          ⏹ Stop
        </button>
      </div>
    </div>
  )
}

function CameraIcon() {
  return (
    <svg width="9" height="9" viewBox="0 0 9 9" fill="none">
      <circle cx="4.5" cy="4.5" r="3.5" stroke="currentColor" strokeWidth="1" />
      <circle cx="4.5" cy="4.5" r="1.5" fill="currentColor" />
    </svg>
  )
}
