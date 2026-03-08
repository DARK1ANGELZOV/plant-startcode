'use client'

import { useState } from 'react'
import type { GrowthTrackResponse } from '@/lib/types'
import { CLASS_COLORS, CLASS_LABELS } from '@/lib/types'
import { ChevronDown, ChevronUp, TrendingUp } from 'lucide-react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'

interface GrowthResultProps {
  result: GrowthTrackResponse
}

export function GrowthResult({ result }: GrowthResultProps) {
  const [showFrames, setShowFrames] = useState(true)

  const chartData = result.frames.map((frame) => {
    const dataPoint: Record<string, number | string> = {
      name: `Кадр ${frame.frame_index + 1}`,
      timestamp: frame.timestamp,
    }

    result.tracks.forEach((track) => {
      const point = track.points.find((p) => p.frame_index === frame.frame_index)
      if (point?.length_mm != null) {
        dataPoint[`track_${track.track_id}_length`] = point.length_mm
      }
    })
    return dataPoint
  })

  const getTrackColor = (className: string, index: number) => {
    const color = CLASS_COLORS[className]
    if (color) return color
    const fallback = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6']
    return fallback[index % fallback.length]
  }

  const getPhiStatusColor = (status: string) => {
    const normalized = String(status || '').toLowerCase()
    if (normalized === 'healthy') return 'text-emerald-400'
    if (normalized === 'risk' || normalized === 'warning') return 'text-amber-400'
    if (normalized === 'critical') return 'text-red-400'
    return 'text-white/60'
  }

  return (
    <div className="glass-card space-y-4">
      <div className="flex items-center gap-3">
        <div className="p-2 bg-emerald-500/20 rounded-lg">
          <TrendingUp size={20} className="text-emerald-500" />
        </div>
        <div>
          <h3 className="text-lg font-medium text-white">Отслеживание роста</h3>
          <p className="text-sm text-white/60">
            {result.frames.length} кадров • {result.tracks.length} треков
          </p>
        </div>
      </div>

      {result.tracks.length > 0 && chartData.some((d) => Object.keys(d).length > 2) && (
        <div className="h-64 mt-4">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData} margin={{ top: 5, right: 20, left: 10, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
              <XAxis dataKey="name" stroke="rgba(255,255,255,0.5)" tick={{ fill: 'rgba(255,255,255,0.5)', fontSize: 12 }} />
              <YAxis
                stroke="rgba(255,255,255,0.5)"
                tick={{ fill: 'rgba(255,255,255,0.5)', fontSize: 12 }}
                label={{
                  value: 'Длина (мм)',
                  angle: -90,
                  position: 'insideLeft',
                  style: { fill: 'rgba(255,255,255,0.5)', fontSize: 12 },
                }}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: 'rgba(20, 20, 20, 0.95)',
                  border: '1px solid rgba(255,255,255,0.1)',
                  borderRadius: '8px',
                  color: '#fff',
                }}
                labelStyle={{ color: 'rgba(255,255,255,0.8)' }}
              />
              <Legend wrapperStyle={{ color: 'rgba(255,255,255,0.8)' }} />
              {result.tracks.map((track, index) => (
                <Line
                  key={track.track_id}
                  type="monotone"
                  dataKey={`track_${track.track_id}_length`}
                  stroke={getTrackColor(track.class_name, index)}
                  strokeWidth={2}
                  dot={{ r: 4, fill: getTrackColor(track.class_name, index) }}
                  activeDot={{ r: 6 }}
                  name={`${CLASS_LABELS[track.class_name] || track.class_name} #${track.track_id}`}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {result.tracks.length > 0 && (
        <div className="flex flex-wrap gap-3">
          {result.tracks.map((track, index) => (
            <div key={track.track_id} className="flex items-center gap-2 px-3 py-1.5 bg-white/5 rounded-full border border-white/10">
              <div className="w-3 h-3 rounded-full" style={{ backgroundColor: getTrackColor(track.class_name, index) }} />
              <span className="text-sm text-white/80">
                {CLASS_LABELS[track.class_name] || track.class_name} #{track.track_id}
              </span>
            </div>
          ))}
        </div>
      )}

      <div>
        <button onClick={() => setShowFrames(!showFrames)} className="flex items-center justify-between w-full py-2 text-white/80 hover:text-white transition-colors">
          <span className="font-medium">Кадры ({result.frames.length})</span>
          {showFrames ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
        </button>

        {showFrames && result.frames.length > 0 && (
          <div className="overflow-x-auto mt-2">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-white/10">
                  <th className="px-3 py-2 text-left text-white/60 font-medium">Кадр</th>
                  <th className="px-3 py-2 text-left text-white/60 font-medium">Время</th>
                  <th className="px-3 py-2 text-right text-white/60 font-medium">Растений</th>
                  <th className="px-3 py-2 text-center text-white/60 font-medium">PHI статус</th>
                </tr>
              </thead>
              <tbody>
                {result.frames.map((frame) => (
                  <tr key={frame.frame_index} className="border-b border-white/5">
                    <td className="px-3 py-2 text-white/80">{frame.frame_index + 1}</td>
                    <td className="px-3 py-2 text-white/80">{new Date(frame.timestamp).toLocaleString('ru-RU')}</td>
                    <td className="px-3 py-2 text-right text-white/80">{frame.count}</td>
                    <td className={`px-3 py-2 text-center ${getPhiStatusColor(frame.phi?.status || '')}`}>{frame.phi?.status || 'N/A'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}
