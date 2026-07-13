// Every OCR attempt is shown, not just successfully-validated plates — the
// status column/color makes it clear which is which so failures stay
// visible for debugging instead of silently disappearing.
const STATUS_STYLE = {
  accepted: { label: 'OK', className: 'text-green-400' },
  rejected: { label: 'Rejected', className: 'text-yellow-500' },
  no_text: { label: 'No text', className: 'text-gray-500' },
}

export default function DetectionTable({ results }) {
  const rows = results

  if (rows.length === 0) {
    return <p className="mt-3 text-sm text-gray-400">No plate reads yet.</p>
  }

  return (
    <table className="mt-3 w-full max-w-3xl border-collapse text-sm">
      <thead>
        <tr className="border-b border-gray-700 text-left text-gray-400">
          <th className="py-1.5 pr-4">Track</th>
          <th className="py-1.5 pr-4">Image</th>
          <th className="py-1.5 pr-4">Vehicle</th>
          <th className="py-1.5 pr-4">Plate</th>
          <th className="py-1.5 pr-4">Confidence</th>
          <th className="py-1.5 pr-4">Status</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((r) => {
          const status = STATUS_STYLE[r.status] ?? STATUS_STYLE.rejected
          return (
            <tr key={r.id} className="border-b border-gray-800">
              <td className="py-1.5 pr-4 text-gray-300">#{r.track_id}</td>
              <td className="py-1.5 pr-4">
                {r.image ? (
                  <img src={r.image} alt={`Plate crop for track ${r.track_id}`} className="h-8 rounded" />
                ) : (
                  <span className="text-gray-600">—</span>
                )}
              </td>
              <td className="py-1.5 pr-4 text-gray-300 capitalize">{r.vehicle_type}</td>
              <td className={`py-1.5 pr-4 font-mono font-semibold ${status.className}`}>{r.plate ?? '—'}</td>
              <td className="py-1.5 pr-4 text-gray-400">
                {r.ocr_confidence != null ? `${(r.ocr_confidence * 100).toFixed(0)}%` : '—'}
              </td>
              <td className={`py-1.5 pr-4 ${status.className}`}>{status.label}</td>
            </tr>
          )
        })}
      </tbody>
    </table>
  )
}
