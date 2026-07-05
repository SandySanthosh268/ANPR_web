import { useEffect, useState } from 'react'
import { getCamera, getPlateResults, startCamera } from '../services/api'
import LivePlayer from '../components/LivePlayer'
import DetectionTable from '../components/DetectionTable'

const PLATE_POLL_MS = 2000

export default function CameraView({ cameraId }) {
  const [camera, setCamera] = useState(null)
  const [error, setError] = useState(null)
  const [started, setStarted] = useState(false)
  // Keyed by track_id so a track's row updates in place as OCR retries
  // produce a better reading, instead of accumulating duplicate rows.
  const [plateResults, setPlateResults] = useState({})

  useEffect(() => {
    getCamera(cameraId)
      .then(setCamera)
      .catch((err) => setError(err.message))
  }, [cameraId])

  // OCR runs asynchronously on the backend and can resolve well after its
  // triggering segment was already fetched for the live overlay, so results
  // are polled from a separate endpoint instead of riding along with
  // per-segment detections.
  useEffect(() => {
    if (!started || !camera) return
    const poll = () => {
      getPlateResults(camera.detections_url)
        .then(({ results }) => {
          setPlateResults((prev) => {
            const next = { ...prev }
            for (const r of results) next[r.track_id] = r
            return next
          })
        })
        .catch(() => {})
    }
    poll()
    const interval = setInterval(poll, PLATE_POLL_MS)
    return () => clearInterval(interval)
  }, [started, camera])

  if (error) return <p className="text-red-400">Failed to load camera: {error}</p>
  if (!camera) return <p className="text-gray-400">Loading camera {cameraId}...</p>

  return (
    <div className="p-4">
      <h1 className="text-xl font-semibold mb-3">Camera: {camera.camera_id}</h1>
      {started ? (
        <>
          <LivePlayer
            hlsUrl={camera.hls_url}
            detectionsUrl={camera.detections_url}
            onEnded={() => {
              setStarted(false)
              setPlateResults({})
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
