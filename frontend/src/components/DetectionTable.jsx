export default function DetectionTable({ results }) {
  const rows = Object.values(results).sort((a, b) => b.track_id - a.track_id)

  if (rows.length === 0) {
    return <p className="mt-3 text-sm text-gray-400">No plate reads yet.</p>
  }

  return (
    <table className="mt-3 w-full max-w-3xl border-collapse text-sm">
      <thead>
        <tr className="border-b border-gray-700 text-left text-gray-400">
          <th className="py-1.5 pr-4">Track</th>
          <th className="py-1.5 pr-4">Vehicle</th>
          <th className="py-1.5 pr-4">Plate</th>
          <th className="py-1.5 pr-4">Confidence</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((r) => (
          <tr key={r.track_id} className="border-b border-gray-800">
            <td className="py-1.5 pr-4 text-gray-300">#{r.track_id}</td>
            <td className="py-1.5 pr-4 text-gray-300 capitalize">{r.vehicle_type}</td>
            <td className="py-1.5 pr-4 font-mono font-semibold text-green-400">{r.plate}</td>
            <td className="py-1.5 pr-4 text-gray-400">{(r.ocr_confidence * 100).toFixed(0)}%</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}
