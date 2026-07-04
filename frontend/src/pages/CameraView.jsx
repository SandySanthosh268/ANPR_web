import { useEffect, useState } from 'react'
import { getCamera } from '../services/api'
import LivePlayer from '../components/LivePlayer'
import DetectionTable from '../components/DetectionTable'

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

  const handlePlateResult = (detection) => {
    setPlateResults((prev) => {
      const existing = prev[detection.track_id]
      if (existing && existing.ocr_confidence >= detection.ocr_confidence) return prev
      return { ...prev, [detection.track_id]: detection }
    })
  }

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
            onPlateResult={handlePlateResult}
          />
          <DetectionTable results={plateResults} />
        </>
      ) : (
        <div className="flex h-96 w-full max-w-3xl items-center justify-center rounded bg-black/40">
          <button
            onClick={() => setStarted(true)}
            className="rounded bg-green-600 px-8 py-4 text-lg font-medium text-white hover:bg-green-500"
          >
            ▶ Play
          </button>
        </div>
      )}
    </div>
  )
}
