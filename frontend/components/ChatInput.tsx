'use client'

import { useState, useRef, useCallback } from 'react'
import { Paperclip, Settings, Send, X, Loader2 } from 'lucide-react'
import { AnalysisSettings, DEFAULT_ANALYSIS_SETTINGS, CROP_OPTIONS } from '@/lib/types'
import Link from 'next/link'

interface ChatInputProps {
  onSend: (message: string, images: File[], settings: AnalysisSettings) => void
  isLoading: boolean
  isDisabled: boolean
  guestTrialUsed: boolean
  isGuest: boolean
}

type ModePreset = 'fast' | 'balanced' | 'strict'

const MODE_PRESETS: Record<ModePreset, { conf: number; iou: number; max_det: number; label: string }> = {
  fast: { conf: 0.12, iou: 0.45, max_det: 40, label: 'Быстрый' },
  balanced: { conf: 0.08, iou: 0.45, max_det: 60, label: 'Сбалансированный' },
  strict: { conf: 0.18, iou: 0.5, max_det: 35, label: 'Точный' },
}

export function ChatInput({ onSend, isLoading, isDisabled, guestTrialUsed, isGuest }: ChatInputProps) {
  const [message, setMessage] = useState('')
  const [images, setImages] = useState<File[]>([])
  const [imagePreviews, setImagePreviews] = useState<string[]>([])
  const [showSettings, setShowSettings] = useState(false)
  const [modePreset, setModePreset] = useState<ModePreset>('balanced')
  const [settings, setSettings] = useState<AnalysisSettings>(DEFAULT_ANALYSIS_SETTINGS)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || [])
    if (files.length === 0) return

    setImages((prev) => [...prev, ...files])
    files.forEach((file) => {
      const reader = new FileReader()
      reader.onload = (event) => {
        setImagePreviews((prev) => [...prev, event.target?.result as string])
      }
      reader.readAsDataURL(file)
    })

    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }, [])

  const removeImage = (index: number) => {
    setImages((prev) => prev.filter((_, i) => i !== index))
    setImagePreviews((prev) => prev.filter((_, i) => i !== index))
  }

  const handleSubmit = () => {
    if (isLoading || isDisabled) return
    if (!message.trim() && images.length === 0) return
    if (isGuest && guestTrialUsed && images.length > 0) return

    const preset = MODE_PRESETS[modePreset]
    onSend(message, images, {
      ...settings,
      conf: preset.conf,
      iou: preset.iou,
      max_det: preset.max_det,
    })

    setMessage('')
    setImages([])
    setImagePreviews([])
    setShowSettings(false)
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit()
    }
  }

  const canSend = Boolean(message.trim() || images.length > 0) && !isLoading && !isDisabled
  const showTrialWarning = isGuest && guestTrialUsed && images.length > 0

  return (
    <div className="border-t border-white/10 bg-[#0f0f0f] p-4">
      {showTrialWarning && (
        <div className="mb-3 px-4 py-3 bg-amber-500/10 border border-amber-500/30 rounded-lg">
          <p className="text-amber-400 text-sm">
            Пробный анализ фото уже использован.{' '}
            <Link href="/register" className="underline hover:text-amber-300">
              Зарегистрируйтесь
            </Link>{' '}
            для полного доступа.
          </p>
        </div>
      )}

      {imagePreviews.length > 0 && (
        <div className="flex flex-wrap gap-2 mb-3">
          {imagePreviews.map((preview, index) => (
            <div key={index} className="relative group w-16 h-16 rounded-lg overflow-hidden border border-white/10">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={preview} alt={`Изображение ${index + 1}`} className="w-full h-full object-cover" />
              <div className="absolute top-0 left-0 bg-emerald-500 text-white text-xs px-1.5 py-0.5 rounded-br">{index + 1}</div>
              <button onClick={() => removeImage(index)} className="absolute top-0 right-0 p-1 bg-black/70 rounded-bl opacity-0 group-hover:opacity-100 transition-opacity">
                <X size={12} className="text-white" />
              </button>
            </div>
          ))}
        </div>
      )}

      {showSettings && (
        <div className="mb-3 p-4 glass rounded-lg settings-panel-enter">
          <h4 className="text-sm font-medium text-white mb-3">Параметры анализа</h4>
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-xs text-white/60 mb-1">Культура</label>
              <select
                value={settings.crop}
                onChange={(e) => setSettings((prev) => ({ ...prev, crop: e.target.value }))}
                className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white focus:outline-none focus:border-emerald-500"
              >
                {CROP_OPTIONS.map((crop) => (
                  <option key={crop.value} value={crop.value} className="bg-[#1a1a1a]">
                    {crop.label}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-xs text-white/60 mb-1">Режим</label>
              <select
                value={modePreset}
                onChange={(e) => setModePreset(e.target.value as ModePreset)}
                className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white focus:outline-none focus:border-emerald-500"
              >
                {Object.entries(MODE_PRESETS).map(([key, value]) => (
                  <option key={key} value={key} className="bg-[#1a1a1a]">
                    {value.label}
                  </option>
                ))}
              </select>
            </div>
          </div>
        </div>
      )}

      <div className="flex items-end gap-2">
        <button
          onClick={() => fileInputRef.current?.click()}
          disabled={isLoading || (isGuest && guestTrialUsed)}
          className="p-3 hover:bg-white/10 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex-shrink-0 h-12"
          title="Прикрепить изображения"
        >
          <Paperclip size={20} className="text-white/60" />
        </button>

        <input ref={fileInputRef} type="file" multiple accept="image/*" onChange={handleFileSelect} className="hidden" />

        <div className="flex-1 flex items-end">
          <textarea
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Напишите сообщение..."
            rows={1}
            disabled={isLoading}
            className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-lg text-white placeholder:text-white/40 focus:outline-none focus:border-emerald-500 resize-none min-h-[48px] max-h-[200px]"
          />
        </div>

        <button
          onClick={() => setShowSettings((v) => !v)}
          className={`p-3 rounded-lg transition-colors flex-shrink-0 h-12 ${showSettings ? 'bg-emerald-500/20 text-emerald-500' : 'hover:bg-white/10 text-white/60'}`}
          title="Параметры"
        >
          <Settings size={20} />
        </button>

        <button
          onClick={handleSubmit}
          disabled={!canSend || showTrialWarning}
          className="p-3 bg-emerald-500 hover:bg-emerald-600 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex-shrink-0 h-12"
          title="Отправить"
        >
          {isLoading ? <Loader2 size={20} className="text-white animate-spin" /> : <Send size={20} className="text-white" />}
        </button>
      </div>
    </div>
  )
}
