'use client'

import { useState } from 'react'
import type { PredictResponse } from '@/lib/types'
import { CLASS_COLORS, CLASS_LABELS } from '@/lib/types'
import { toAssetUrl } from '@/lib/api'
import { CheckCircle, AlertTriangle, XCircle, Info, ChevronDown, ChevronUp, Maximize2 } from 'lucide-react'

interface AnalysisResultProps {
  result: PredictResponse
}

type ExplainabilityTab = 'confidence' | 'attention' | 'gradcam' | 'uncertainty'

function toConfidenceLabel(level?: string): { className: string; label: string } {
  const normalized = String(level || '').toLowerCase()
  if (normalized === 'high') return { className: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30', label: 'Высокая' }
  if (normalized === 'medium') return { className: 'bg-amber-500/20 text-amber-400 border-amber-500/30', label: 'Средняя' }
  if (normalized === 'low') return { className: 'bg-red-500/20 text-red-400 border-red-500/30', label: 'Низкая' }
  return { className: 'bg-white/10 text-white/70 border-white/20', label: level || 'Нет данных' }
}

export function AnalysisResult({ result }: AnalysisResultProps) {
  const [showMeasurements, setShowMeasurements] = useState(true)
  const [showExplainability, setShowExplainability] = useState(false)
  const [activeTab, setActiveTab] = useState<ExplainabilityTab>('confidence')
  const [showFullImage, setShowFullImage] = useState(false)

  const overlayUrl = toAssetUrl(result.files.overlay)
  const imageQualityNotes = result.summary?.image_quality?.notes || []
  const confidenceBadge = toConfidenceLabel(result.summary?.measurement_trust_level)
  const phiStatus = String(result.phi?.status || '').toLowerCase()

  const getPhiStatusIcon = () => {
    if (phiStatus === 'healthy') return <CheckCircle size={18} className="text-emerald-500" />
    if (phiStatus === 'risk' || phiStatus === 'warning') return <AlertTriangle size={18} className="text-amber-500" />
    if (phiStatus === 'critical') return <XCircle size={18} className="text-red-500" />
    return <Info size={18} className="text-white/60" />
  }

  const explainabilityTabs: { key: ExplainabilityTab; label: string; available: boolean; url: string }[] = [
    {
      key: 'confidence',
      label: 'Confidence Map',
      available: Boolean(result.explainability?.confidence_map || result.files?.xai_confidence_map),
      url: toAssetUrl(result.explainability?.confidence_map || result.files?.xai_confidence_map),
    },
    {
      key: 'attention',
      label: 'Attention',
      available: Boolean(result.explainability?.attention_heatmap || result.files?.xai_attention),
      url: toAssetUrl(result.explainability?.attention_heatmap || result.files?.xai_attention),
    },
    {
      key: 'gradcam',
      label: 'Grad-CAM',
      available: Boolean(result.explainability?.gradcam || result.files?.xai_gradcam),
      url: toAssetUrl(result.explainability?.gradcam || result.files?.xai_gradcam),
    },
    {
      key: 'uncertainty',
      label: 'Uncertainty',
      available: Boolean(result.explainability?.uncertainty_map || result.files?.xai_uncertainty),
      url: toAssetUrl(result.explainability?.uncertainty_map || result.files?.xai_uncertainty),
    },
  ]

  const hasExplainability = explainabilityTabs.some((tab) => tab.available)
  const activeExplainability = explainabilityTabs.find((tab) => tab.key === activeTab && tab.available)

  return (
    <div className="glass-card space-y-4">
      <div className="relative">
        <div className="relative rounded-lg overflow-hidden border border-white/10 cursor-pointer" onClick={() => setShowFullImage(true)}>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={overlayUrl} alt="Результат анализа" className="w-full h-auto" />
          <button
            className="absolute top-2 right-2 p-2 bg-black/50 hover:bg-black/70 rounded-lg transition-colors"
            onClick={(e) => {
              e.stopPropagation()
              setShowFullImage(true)
            }}
          >
            <Maximize2 size={16} className="text-white" />
          </button>
        </div>

        <div className="flex flex-wrap gap-3 mt-3">
          {Object.entries(CLASS_COLORS).map(([key, color]) => (
            <div key={key} className="flex items-center gap-2">
              <div className="w-4 h-4 rounded-sm" style={{ backgroundColor: color }} />
              <span className="text-sm text-white/70">{CLASS_LABELS[key] || key}</span>
            </div>
          ))}
        </div>
      </div>

      <div className="flex flex-wrap gap-4">
        <div className={`px-3 py-1.5 rounded-full border ${confidenceBadge.className}`}>
          <span className="text-sm">Доверие к измерениям: {confidenceBadge.label}</span>
        </div>
        <div className="flex items-center gap-2 px-3 py-1.5 bg-white/5 rounded-full border border-white/10">
          {getPhiStatusIcon()}
          <span className="text-sm text-white/80">
            PHI: {result.phi?.score?.toFixed(1) ?? 'N/A'} ({result.phi?.status || 'N/A'})
          </span>
        </div>
      </div>

      {imageQualityNotes.length > 0 && (
        <div className="p-3 bg-amber-500/10 border border-amber-500/20 rounded-lg">
          <p className="text-sm text-amber-400 font-medium mb-1">Качество изображения</p>
          <ul className="text-sm text-amber-300/80 space-y-1">
            {imageQualityNotes.map((note, index) => (
              <li key={index}>- {note}</li>
            ))}
          </ul>
        </div>
      )}

      <div>
        <button onClick={() => setShowMeasurements(!showMeasurements)} className="flex items-center justify-between w-full py-2 text-white/80 hover:text-white transition-colors">
          <span className="font-medium">Измерения ({result.measurements?.length || 0})</span>
          {showMeasurements ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
        </button>

        {showMeasurements && result.measurements && result.measurements.length > 0 && (
          <div className="overflow-x-auto mt-2">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-white/10">
                  <th className="px-3 py-2 text-left text-white/60 font-medium">ID</th>
                  <th className="px-3 py-2 text-left text-white/60 font-medium">Класс</th>
                  <th className="px-3 py-2 text-right text-white/60 font-medium">Длина</th>
                  <th className="px-3 py-2 text-right text-white/60 font-medium">Площадь</th>
                  <th className="px-3 py-2 text-right text-white/60 font-medium">Уверенность</th>
                  <th className="px-3 py-2 text-center text-white/60 font-medium">Надежно</th>
                </tr>
              </thead>
              <tbody>
                {result.measurements.map((m) => (
                  <tr key={m.instance_id} className="border-b border-white/5">
                    <td className="px-3 py-2 text-white/80">{m.instance_id}</td>
                    <td className="px-3 py-2">
                      <span className="inline-flex items-center gap-1.5" style={{ color: CLASS_COLORS[m.class_name] || '#fff' }}>
                        <span className="w-2 h-2 rounded-full" style={{ backgroundColor: CLASS_COLORS[m.class_name] || '#fff' }} />
                        {CLASS_LABELS[m.class_name] || m.class_name}
                      </span>
                    </td>
                    <td className="px-3 py-2 text-right text-white/80">
                      {m.length_mm != null ? `${m.length_mm.toFixed(2)} мм` : `${m.length_px.toFixed(2)} px`}
                    </td>
                    <td className="px-3 py-2 text-right text-white/80">
                      {m.area_mm2 != null ? `${m.area_mm2.toFixed(2)} мм²` : `${m.area_px.toFixed(0)} px²`}
                    </td>
                    <td className="px-3 py-2 text-right text-white/80">{(m.confidence * 100).toFixed(1)}%</td>
                    <td className="px-3 py-2 text-center">
                      {m.reliable ? <CheckCircle size={16} className="inline text-emerald-500" /> : <XCircle size={16} className="inline text-red-500" />}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {hasExplainability && (
        <div>
          <button onClick={() => setShowExplainability(!showExplainability)} className="flex items-center justify-between w-full py-2 text-white/80 hover:text-white transition-colors">
            <span className="font-medium">Объяснимость модели</span>
            {showExplainability ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
          </button>

          {showExplainability && (
            <div className="mt-2">
              <div className="flex flex-wrap gap-2 mb-3">
                {explainabilityTabs
                  .filter((tab) => tab.available)
                  .map((tab) => (
                    <button
                      key={tab.key}
                      onClick={() => setActiveTab(tab.key)}
                      className={`px-3 py-1.5 text-sm rounded-lg transition-colors ${
                        activeTab === tab.key
                          ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                          : 'bg-white/5 text-white/60 border border-white/10 hover:bg-white/10'
                      }`}
                    >
                      {tab.label}
                    </button>
                  ))}
              </div>

              {activeExplainability?.url && (
                <div className="rounded-lg overflow-hidden border border-white/10">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src={activeExplainability.url} alt={activeExplainability.label} className="w-full h-auto" />
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {showFullImage && (
        <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4" onClick={() => setShowFullImage(false)}>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={overlayUrl} alt="Результат анализа" className="max-w-full max-h-full object-contain" onClick={(e) => e.stopPropagation()} />
        </div>
      )}
    </div>
  )
}
