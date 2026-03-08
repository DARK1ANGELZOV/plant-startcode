'use client'

import type { ChatMessage as ChatMessageType, PredictResponse, GrowthTrackResponse } from '@/lib/types'
import { AnalysisResult } from './AnalysisResult'
import { GrowthResult } from './GrowthResult'
import { User, Bot } from 'lucide-react'

interface ChatMessageProps {
  message: ChatMessageType
}

export function ChatMessage({ message }: ChatMessageProps) {
  const isUser = message.role === 'user'

  return (
    <div className={`flex gap-3 animate-fade-in ${isUser ? 'flex-row-reverse' : ''}`}>
      <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${isUser ? 'bg-emerald-500/20' : 'bg-white/10'}`}>
        {isUser ? <User size={16} className="text-emerald-500" /> : <Bot size={16} className="text-white/80" />}
      </div>

      <div className={`flex-1 max-w-[80%] ${isUser ? 'text-right' : ''}`}>
        {message.content && (
          <div className={`inline-block px-4 py-3 rounded-2xl ${isUser ? 'message-user rounded-tr-md' : 'message-assistant rounded-tl-md'}`}>
            <p className="text-white/90 text-sm whitespace-pre-wrap leading-relaxed">{message.content}</p>
          </div>
        )}

        {isUser && message.images && message.images.length > 0 && (
          <div className={`flex flex-wrap gap-2 mt-2 ${isUser ? 'justify-end' : ''}`}>
            {message.images.map((img, index) => (
              <div key={index} className="w-24 h-24 rounded-lg overflow-hidden border border-white/10">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={img} alt={`Изображение ${index + 1}`} className="w-full h-full object-cover" />
              </div>
            ))}
          </div>
        )}

        {!isUser && message.analysis_result && (
          <div className="mt-3">
            <AnalysisResult result={message.analysis_result as PredictResponse} />
          </div>
        )}

        {!isUser && message.growth_result && (
          <div className="mt-3">
            <GrowthResult result={message.growth_result as GrowthTrackResponse} />
          </div>
        )}

        <p className={`text-xs text-white/40 mt-2 ${isUser ? 'text-right' : ''}`}>
          {new Date(message.created_at).toLocaleTimeString('ru-RU', {
            hour: '2-digit',
            minute: '2-digit',
          })}
        </p>
      </div>
    </div>
  )
}
