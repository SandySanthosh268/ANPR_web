import { useEffect, useState } from 'react'
import { getCamera, getPlateResults, startCamera } from '../services/api'
import LivePlayer from '../components/LivePlayer'
import DetectionTable from '../components/DetectionTable'

const PLATE_POLL_MS = 2000

export default function CameraView({ cameraId }) {
  const [camera, setCamera] = useState(null)
  const [error, setError] = useState(null)
  const [started, setStarted] = useState(false)
  // Every OCR attempt (accepted, rejected, or no readable text) — the
  // backend already returns the full bounded, most-recent-first list, so
  // this is just replaced wholesale on each poll rather than merged.
  const [plateResults, setPlateResults] = useState([])

  useEffect(() => {
    getCamera(cameraId)
      .then(setCamera)
      .catch((err) => setError(err.message))
  }, [cameraId])

  // frame_width/frame_height (the source resolution detection bboxes are in)
  // are only known once the backend's pipeline has processed its first frame
  // — null at the moment Play is clicked. Poll briefly until populated so
  // CanvasOverlay can scale boxes correctly instead of assuming they're in
  // the HLS preview stream's (possibly downscaled) decoded resolution.
  useEffect(() => {
    if (!started || !camera || camera.frame_width) return
    const interval = setInterval(() => {
      getCamera(cameraId)
        .then((data) => {
          if (data.frame_width) {
            setCamera(data)
            clearInterval(interval)
          }
        })
        .catch(() => {})
    }, 500)
    return () => clearInterval(interval)
  }, [started, camera, cameraId])

  // OCR runs asynchronously on the backend and can resolve well after its
  // triggering segment was already fetched for the live overlay, so results
  // are polled from a separate endpoint instead of riding along with
  // per-segment detections.
  useEffect(() => {
    if (!started || !camera) return
    const poll = () => {
      getPlateResults(camera.detections_url)
        .then(({ results }) => setPlateResults(results))
        .catch(() => {})
    }
    poll()
    const interval = setInterval(poll, PLATE_POLL_MS)
    return () => clearInterval(interval)
  }, [started, camera])

  // vehicle_count climbs as PlateTracker mints new track ids — same poll
  // cadence as plate results, just re-fetching the camera endpoint since
  // that's where the backend surfaces it (no dedicated count endpoint).
  useEffect(() => {
    if (!started) return
    const poll = () => {
      getCamera(cameraId)
        .then((data) => setCamera((prev) => (prev ? { ...prev, vehicle_count: data.vehicle_count } : prev)))
        .catch(() => {})
    }
    poll()
    const interval = setInterval(poll, PLATE_POLL_MS)
    return () => clearInterval(interval)
  }, [started, cameraId])

  if (error) return <p className="text-red-400">Failed to load camera: {error}</p>
  if (!camera) return <p className="text-gray-400">Loading camera {cameraId}...</p>

  return (
    <div className="p-4">
      <h1 className="text-xl font-semibold mb-3">Camera: {camera.camera_id}</h1>
      {started ? (
        <>
          <p className="mb-2 text-gray-300">
            Vehicles crossed: <span className="font-semibold text-white">{camera.vehicle_count ?? 0}</span>
          </p>
          <LivePlayer
            hlsUrl={camera.hls_url}
            detectionsUrl={camera.detections_url}
            frameWidth={camera.frame_width}
            frameHeight={camera.frame_height}
            onEnded={() => {
              setStarted(false)
              setPlateResults([])
            }}
          />
          <DetectionTable results={plateResults} />
        </>
      ) : (
        <div className="flex h-96 w-full max-w-3xl items-center justify-center rounded bg-black/40">
          <button
            onClick={async () => {
              try {
                // Tells the backend to actually begin reading/processing the
                // source — before this, its models are loaded and idle, not
                // running the detection loop regardless of the process
                // having been launched.
                await startCamera(cameraId)
                setStarted(true)
              } catch (err) {
                setError(err.message)
              }
            }}
            className="rounded bg-green-600 px-8 py-4 text-lg font-medium text-white hover:bg-green-500"
          >
            ▶ Play
          </button>
        </div>
      )}
    </div>
  )
}
