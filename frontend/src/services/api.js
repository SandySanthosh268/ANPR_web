import axios from 'axios'

const client = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL,
})

export async function getCamera(cameraId) {
  const { data } = await client.get(`/api/cameras/${cameraId}`)
  return data
}

// detectionsUrl is already absolute (returned by getCamera) — axios ignores
// baseURL when given an absolute url, so this just uses the shared client.
export async function getDetections(detectionsUrl, segment) {
  const { data } = await client.get(detectionsUrl, { params: { segment } })
  return data
}
