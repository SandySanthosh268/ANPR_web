import { useEffect, useRef } from 'react'

const VEHICLE_COLOR = '#22c55e'
const PLATE_COLOR = '#eab308'
const NEAREST_FRAME_TOLERANCE_SEC = 2

export default function CanvasOverlay({ frames, videoRef }) {
  const canvasRef = useRef(null)
  // Read via ref inside the draw loop instead of restarting the effect on
  // every `frames` update — avoids tearing down/recreating the rAF loop
  // every time a new segment's detections arrive.
  const framesRef = useRef(frames)
  framesRef.current = frames

  useEffect(() => {
    const video = videoRef.current
    const canvas = canvasRef.current
    if (!video || !canvas) return

    let rafId
    const draw = () => {
      const rect = video.getBoundingClientRect()
      if (canvas.width !== Math.round(rect.width) || canvas.height !== Math.round(rect.height)) {
        canvas.width = Math.round(rect.width)
        canvas.height = Math.round(rect.height)
      }
      const ctx = canvas.getContext('2d')
      ctx.clearRect(0, 0, canvas.width, canvas.height)

      const vidW = video.videoWidth
      const vidH = video.videoHeight
      const currentFrames = framesRef.current

      if (currentFrames?.length && vidW > 0) {
        let best = null
        let bestDiff = Infinity
        for (const f of currentFrames) {
          const d = Math.abs(f.video_timestamp_sec - video.currentTime)
          if (d < bestDiff) {
            bestDiff = d
            best = f
          }
        }

        if (best && bestDiff < NEAREST_FRAME_TOLERANCE_SEC) {
          drawFrame(ctx, canvas, vidW, vidH, best.detections ?? [])
        }
      }
      rafId = requestAnimationFrame(draw)
    }
    rafId = requestAnimationFrame(draw)
    return () => cancelAnimationFrame(rafId)
  }, [videoRef])

  return <canvas ref={canvasRef} className="pointer-events-none absolute inset-0 h-full w-full" />
}

function drawFrame(ctx, canvas, vidW, vidH, detections) {
  // object-contain letterbox/pillarbox offset — the decoded video may not
  // exactly fill the canvas (e.g. fullscreen with a different aspect ratio
  // than the source), so boxes must be positioned relative to the actual
  // rendered video area, not the raw canvas/container size.
  const videoAspect = vidW / vidH
  const canvasAspect = canvas.width / canvas.height
  let renderW, renderH, offsetX, offsetY
  if (videoAspect > canvasAspect) {
    renderW = canvas.width
    renderH = canvas.width / videoAspect
    offsetX = 0
    offsetY = (canvas.height - renderH) / 2
  } else {
    renderH = canvas.height
    renderW = canvas.height * videoAspect
    offsetX = (canvas.width - renderW) / 2
    offsetY = 0
  }
  const sx = renderW / vidW
  const sy = renderH / vidH

  ctx.lineWidth = Math.max(2, canvas.width / 400)
  ctx.font = `${Math.max(14, canvas.width / 60)}px system-ui, sans-serif`
  ctx.textBaseline = 'bottom'

  for (const d of detections) {
    // OCR still runs and saves to the DB in the background — the plate text
    // just isn't shown live here (surfaced later via a detection history
    // view instead). Track ID labels both boxes so viewers can still
    // visually correlate a vehicle to its plate box.
    drawBox(ctx, d.vehicle_bbox, VEHICLE_COLOR, `${d.vehicle_type} #${d.track_id}`, offsetX, offsetY, sx, sy)
    if (d.plate_bbox) {
      drawBox(ctx, d.plate_bbox, PLATE_COLOR, `#${d.track_id}`, offsetX, offsetY, sx, sy)
    }
  }
}

function drawBox(ctx, bbox, color, label, offsetX, offsetY, sx, sy) {
  const [x1, y1, x2, y2] = bbox
  const rx = offsetX + x1 * sx
  const ry = offsetY + y1 * sy
  const rw = (x2 - x1) * sx
  const rh = (y2 - y1) * sy

  ctx.strokeStyle = color
  ctx.strokeRect(rx, ry, rw, rh)

  const textWidth = ctx.measureText(label).width
  ctx.fillStyle = color
  ctx.fillRect(rx, ry - 20, textWidth + 8, 20)
  ctx.fillStyle = '#0b0d12'
  ctx.fillText(label, rx + 4, ry - 4)
}
