import { useEffect, useRef } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { QUERY_KEYS } from '@/hooks/useAlertStream'
import { useUIStore } from '@/store/ui.store'
import { cn } from '@/utils'
import type { CameraState } from '@/types'

const CAMERAS = [
  { id: 'CAM-01', label: 'CAM-01 · ENTRANCE' },
  { id: 'CAM-02', label: 'CAM-02 · LOBBY' },
  { id: 'CAM-03', label: 'CAM-03 · CORRIDOR' },
  { id: 'CAM-04', label: 'CAM-04 · STAIRWELL' },
]

export function CameraGrid() {
  const qc = useQueryClient()
  const mode = useUIStore((s) => s.mode)
  const cameras = qc.getQueryData<CameraState[]>(QUERY_KEYS.cameras) ?? []

  const getCamState = (id: string) => cameras.find((c) => c.cameraId === id)

  return (
    <div className="panel">
      <div className="panel-header">
        <div className="panel-title">
          <CameraIcon />
          Live cameras
        </div>
        <span className="badge-live">{CAMERAS.length} FEEDS ACTIVE</span>
      </div>

      <div className="grid gap-[5px] p-2" style={{ gridTemplateColumns: '1fr 1fr' }}>
        {CAMERAS.map((cam, idx) => {
          const state = getCamState(cam.id)
          const hasThreat = state?.threatLevel === 'critical' || state?.threatLevel === 'high'
          const isIncidentSecondary = mode === 'incident' && idx === 1

          return (
            <CameraFeed
              key={cam.id}
              cameraIndex={idx}
              label={cam.label}
              hasThreat={hasThreat ?? (idx === 0)}
              hasSecondaryThreat={isIncidentSecondary}
              threatLevel={state?.threatLevel ?? 'clear'}
              mode={mode}
            />
          )
        })}
      </div>
    </div>
  )
}

interface CameraFeedProps {
  cameraIndex: number
  label: string
  hasThreat: boolean
  hasSecondaryThreat: boolean
  threatLevel: string
  mode: string
}

function CameraFeed({
  cameraIndex,
  label,
  hasThreat,
  hasSecondaryThreat,
  threatLevel,
  mode,
}: CameraFeedProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const tickRef = useRef(0)
  const rafRef = useRef<number>(0)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const draw = () => {
      const ctx = canvas.getContext('2d')
      if (!ctx) return

      const w = canvas.offsetWidth || 200
      const h = canvas.offsetHeight || 112
      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w
        canvas.height = h
      }

      const t = tickRef.current + cameraIndex * 17
      tickRef.current++

      // Noise background
      ctx.fillStyle = '#040810'
      ctx.fillRect(0, 0, w, h)

      function rng(s: number) {
        const x = Math.sin(s * 127.1 + 3.9478) * 43758.5453
        return x - Math.floor(x)
      }

      // Background noise blocks
      for (let i = 0; i < 12; i++) {
        const x = rng(t * 0.01 + i) * w
        const y = rng(t * 0.01 + i + 50) * h
        const s = 3 + rng(t * 0.01 + i + 100) * 6
        ctx.fillStyle = `rgba(${15 + rng(i + 200) * 20},${25 + rng(i + 300) * 25},${35 + rng(i + 400) * 30},0.7)`
        ctx.fillRect(Math.floor(x), Math.floor(y), Math.floor(s), Math.floor(s * 1.8))
      }

      // Primary threat bounding box
      if (hasThreat) {
        const bx = w * 0.12, by = h * 0.08, bw = w * 0.42, bh = h * 0.78
        const pulse = (Math.sin(t * 0.12) + 1) * 0.5
        ctx.strokeStyle = `rgba(255,34,68,${0.7 + pulse * 0.3})`
        ctx.lineWidth = 1.5
        ctx.setLineDash([5, 3])
        ctx.strokeRect(bx, by, bw, bh)
        ctx.setLineDash([])
        ctx.fillStyle = 'rgba(255,34,68,0.9)'
        ctx.fillRect(bx, by - 12, mode === 'incident' ? 52 : 52, 12)
        ctx.fillStyle = '#fff'
        ctx.font = 'bold 8px "Share Tech Mono", monospace'
        ctx.fillText(mode === 'incident' ? 'FIREARM' : 'WEAPON', bx + 3, by - 2)
      }

      // Secondary threat bounding box (incident mode)
      if (hasSecondaryThreat) {
        const bx = w * 0.55, by = h * 0.2, bw = w * 0.35, bh = h * 0.6
        ctx.strokeStyle = 'rgba(255,140,0,0.8)'
        ctx.lineWidth = 1.2
        ctx.setLineDash([4, 3])
        ctx.strokeRect(bx, by, bw, bh)
        ctx.setLineDash([])
        ctx.fillStyle = 'rgba(255,140,0,0.85)'
        ctx.fillRect(bx, by - 11, 36, 11)
        ctx.fillStyle = '#fff'
        ctx.font = 'bold 7px "Share Tech Mono", monospace'
        ctx.fillText('KNIFE', bx + 3, by - 2)
      }

      // Scanlines
      ctx.fillStyle = `rgba(255,255,255,${0.015 + rng(t + 999) * 0.01})`
      for (let y = 0; y < h; y += 2) ctx.fillRect(0, y, w, 1)

      rafRef.current = requestAnimationFrame(draw)
    }

    rafRef.current = requestAnimationFrame(draw)
    return () => cancelAnimationFrame(rafRef.current)
  }, [cameraIndex, hasThreat, hasSecondaryThreat, mode])

  const tagText = hasThreat
    ? mode === 'incident' && cameraIndex === 0
      ? 'FIREARM 93%'
      : 'WEAPON 91%'
    : hasSecondaryThreat
      ? 'KNIFE 87%'
      : 'CLEAR'

  const tagCls = hasThreat || hasSecondaryThreat
    ? 'bg-critical/90 text-white'
    : 'bg-cloud/10 text-cloud border border-cloud/20'

  return (
    <div
      className={cn(
        'relative bg-bg-0 rounded-[4px] overflow-hidden border cursor-pointer transition-colors',
        hasThreat || hasSecondaryThreat
          ? 'border-critical shadow-[inset_0_0_0_1px_rgba(255,34,68,0.3)]'
          : 'border-border hover:border-border-3'
      )}
      style={{ aspectRatio: '16/9' }}
    >
      <canvas ref={canvasRef} className="w-full h-full block" />
      <div className="absolute bottom-1 left-[5px] font-mono text-[8px] text-white/50 bg-black/60 px-[5px] py-[1px] rounded-[2px] tracking-[0.04em]">
        {label}
      </div>
      <div className={cn('absolute top-1 right-1 font-mono text-[8px] px-[5px] py-[1px] rounded-[2px] font-bold tracking-[0.04em]', tagCls)}>
        {tagText}
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
