import { useEffect, useRef, useState } from 'react'
import Hls from 'hls.js'
import { getDetections } from '../services/api'
import CanvasOverlay from './CanvasOverlay'

export default function LivePlayer({ hlsUrl, detectionsUrl, onEnded }) {
  const videoRef = useRef(null)
  const [frames, setFrames] = useState([])
  const [buffering, setBuffering] = useState(true)

  useEffect(() => {
    const video = videoRef.current
    if (!video) return

    let hls
    if (Hls.isSupported()) {
      hls = new Hls()
      hls.loadSource(hlsUrl)
      hls.attachMedia(video)
      // Fetching a segment's detections right when hls.js moves into that
      // segment is inherently synchronized — no separate timeline-matching
      // protocol needed, unlike the previous WebSocket + video_time approach.
      hls.on(Hls.Events.FRAG_CHANGED, (_event, data) => {
        getDetections(detectionsUrl, data.frag.sn)
          .then((result) => setFrames(result.frames))
          .catch(() => setFrames([])) // segment not processed by the pipeline yet
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
