import { useEffect, useRef, useState } from 'react'
import Hls from 'hls.js'
import { getDetections } from '../services/api'
import CanvasOverlay from './CanvasOverlay'

const RETRY_DELAY_MS = 1200
const MAX_ATTEMPTS = 3

export default function LivePlayer({ hlsUrl, detectionsUrl, onEnded, onPlateResult }) {
  const videoRef = useRef(null)
  const [frames, setFrames] = useState([])
  const [buffering, setBuffering] = useState(true)
  // Tracks the most recently requested segment so a slow retry for an old
  // segment can't clobber a newer one's already-applied result.
  const latestSegmentRef = useRef(-1)

  useEffect(() => {
    const video = videoRef.current
    if (!video) return

    // Under CPU load the detection pipeline can temporarily fall behind
    // real-time video playback, so a segment the player just reached may not
    // have detection data *yet* even though it will moments later — retry a
    // few times before giving up, rather than permanently showing nothing
    // for that segment.
    const fetchWithRetry = (segment, attempt = 1) => {
      getDetections(detectionsUrl, segment)
        .then((result) => {
          if (latestSegmentRef.current !== segment) return
          setFrames(result.frames)
          // OCR results ride along in the same per-frame detections used for
          // the live overlay boxes — surface any with a resolved plate to
          // the table below the player, rather than opening a separate
          // channel for something the data already carries.
          for (const frame of result.frames) {
            for (const d of frame.detections ?? []) {
              if (d.plate) onPlateResult?.(d)
            }
          }
        })
        .catch(() => {
          if (latestSegmentRef.current !== segment) return
          if (attempt < MAX_ATTEMPTS) {
            setTimeout(() => fetchWithRetry(segment, attempt + 1), RETRY_DELAY_MS)
          } else {
            setFrames([])
          }
        })
    }

    let hls
    if (Hls.isSupported()) {
      hls = new Hls()
      hls.loadSource(hlsUrl)
      hls.attachMedia(video)
      // Fetching a segment's detections right when hls.js moves into that
      // segment is inherently synchronized — no separate timeline-matching
      // protocol needed, unlike the previous WebSocket + video_time approach.
      hls.on(Hls.Events.FRAG_CHANGED, (_event, data) => {
        latestSegmentRef.current = data.frag.sn
        fetchWithRetry(data.frag.sn)
      })
    } else if (video.canPlayType('application/vnd.apple.mpegurl')) {
      video.src = hlsUrl
    }

    // Connect + start loading immediately (above), but hold playback back a
    // few seconds so detection data and HLS segments have a head start —
    // starting immediately meant the first several seconds had no matching
    // detections yet, since inference lags behind the raw video feed.
    const onCanPlay = () => {
      setBuffering(false)
      video.play().catch((err) => console.warn('Autoplay blocked:', err))
    }
    video.addEventListener('canplay', onCanPlay)

    const stopAll = () => {
      video.pause()
      onEnded?.()
    }
    video.addEventListener('ended', stopAll)

    return () => {
      video.removeEventListener('canplay', onCanPlay)
      video.removeEventListener('ended', stopAll)
      hls?.destroy()
    }
  }, [hlsUrl, detectionsUrl])

  return (
    <div className="relative inline-block max-w-full">
      <video ref={videoRef} controls muted className="max-w-full" />
      <CanvasOverlay frames={frames} videoRef={videoRef} />
      {buffering && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/80 text-lg text-white">
          Buffering video and detection data...
        </div>
      )}
    </div>
  )
}
