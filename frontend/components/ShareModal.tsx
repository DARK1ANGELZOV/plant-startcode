'use client'

import { useState, useEffect } from 'react'
import { api } from '@/lib/api'
import type { ShareData } from '@/lib/types'
import { X, Copy, Check, Loader2, ExternalLink } from 'lucide-react'
import toast from 'react-hot-toast'

interface ShareModalProps {
  sessionId: string
  onClose: () => void
}

export function ShareModal({ sessionId, onClose }: ShareModalProps) {
  const [shareData, setShareData] = useState<ShareData | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [copied, setCopied] = useState(false)

  useEffect(() => {
    async function loadShareData() {
      try {
        const data = await api.getShareData(sessionId)
        setShareData(data)
      } catch {
        toast.error('Не удалось получить ссылку для шаринга')
        onClose()
      } finally {
        setIsLoading(false)
      }
    }

    loadShareData()
  }, [sessionId, onClose])

  const handleCopy = async () => {
    if (!shareData?.share_url) return
    try {
      await navigator.clipboard.writeText(shareData.share_url)
      setCopied(true)
      toast.success('Ссылка скопирована')
      setTimeout(() => setCopied(false), 2000)
    } catch {
      toast.error('Ошибка копирования')
    }
  }

  return (
    <div className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="glass-card w-full max-w-md animate-fade-in" onClick={(e) => e.stopPropagation()}>
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold text-white">Поделиться чатом</h2>
          <button onClick={onClose} className="p-2 hover:bg-white/10 rounded-lg transition-colors">
            <X size={20} className="text-white/60" />
          </button>
        </div>

        {isLoading ? (
          <div className="flex items-center justify-center py-8">
            <Loader2 size={32} className="animate-spin text-emerald-500" />
          </div>
        ) : shareData ? (
          <>
            <div className="mb-4">
              <p className="text-sm text-white/60 mb-1">Название чата</p>
              <p className="text-white">{shareData.title || 'Без названия'}</p>
            </div>

            <div className="mb-4">
              <p className="text-sm text-white/60 mb-1">Количество сообщений</p>
              <p className="text-white">{shareData.messages.length}</p>
            </div>

            <div className="mb-4">
              <p className="text-sm text-white/60 mb-2">Ссылка</p>
              <div className="flex items-center gap-2">
                <input
                  type="text"
                  value={shareData.share_url}
                  readOnly
                  className="flex-1 px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white/80"
                />
                <button onClick={handleCopy} className="p-2 bg-emerald-500 hover:bg-emerald-600 rounded-lg transition-colors" title="Копировать">
                  {copied ? <Check size={20} className="text-white" /> : <Copy size={20} className="text-white" />}
                </button>
              </div>
            </div>

            <a
              href={shareData.share_url}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center justify-center gap-2 w-full py-3 border border-emerald-500/50 hover:bg-emerald-500/10 text-emerald-500 rounded-lg transition-colors"
            >
              <ExternalLink size={18} />
              <span>Открыть ссылку</span>
            </a>
          </>
        ) : (
          <p className="text-center text-white/60 py-4">Данные недоступны</p>
        )}
      </div>
    </div>
  )
}
