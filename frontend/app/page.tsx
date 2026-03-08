'use client'

import { useState, useEffect, useRef, useCallback } from 'react'
import { useRouter } from 'next/navigation'
import { useAuth } from '@/contexts/AuthContext'
import { api } from '@/lib/api'
import type { ChatMessage as ChatMessageType, AnalysisSettings, GuestSession } from '@/lib/types'
import { Navbar } from '@/components/Navbar'
import { Sidebar } from '@/components/Sidebar'
import { ChatMessage } from '@/components/ChatMessage'
import { ChatInput } from '@/components/ChatInput'
import { ShareModal } from '@/components/ShareModal'
import { Leaf, Loader2 } from 'lucide-react'
import toast from 'react-hot-toast'

export default function ChatPage() {
  const router = useRouter()
  const { isAuthenticated, isGuest, isLoading: authLoading, checkGuestTrialUsed, setGuestTrialUsed } = useAuth()
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null)
  const [messages, setMessages] = useState<ChatMessageType[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [shareSessionId, setShareSessionId] = useState<string | null>(null)
  const [guestTrialUsed, setLocalGuestTrialUsed] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (isGuest) {
      setLocalGuestTrialUsed(checkGuestTrialUsed())
    }
  }, [isGuest, checkGuestTrialUsed])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  useEffect(() => {
    if (!authLoading && !isAuthenticated && !isGuest) {
      router.push('/login')
    }
  }, [authLoading, isAuthenticated, isGuest, router])

  const loadSessionMessages = useCallback(
    async (sessionId: string) => {
      if (isAuthenticated) {
        try {
          const msgs = await api.getSessionMessages(sessionId)
          setMessages(msgs)
        } catch {
          toast.error('Ошибка загрузки сообщений')
        }
      } else if (isGuest) {
        const stored = localStorage.getItem('guest_sessions')
        if (stored) {
          try {
            const sessions: GuestSession[] = JSON.parse(stored)
            const session = sessions.find((s) => s.id === sessionId)
            if (session) {
              setMessages(session.messages)
            }
          } catch {
            setMessages([])
          }
        }
      }
    },
    [isAuthenticated, isGuest],
  )

  const handleSelectSession = (sessionId: string) => {
    setCurrentSessionId(sessionId)
    loadSessionMessages(sessionId)
  }

  const handleNewChat = async () => {
    if (isAuthenticated) {
      try {
        const session = await api.createSession()
        setCurrentSessionId(session.id)
        setMessages([])
      } catch {
        toast.error('Ошибка создания чата')
      }
    } else if (isGuest) {
      const newSessionId = `guest_${Date.now()}`
      setCurrentSessionId(newSessionId)
      setMessages([])
    }
  }

  const saveGuestSession = (sessionId: string, msgs: ChatMessageType[]) => {
    const stored = localStorage.getItem('guest_sessions')
    let sessions: GuestSession[] = []

    if (stored) {
      try {
        sessions = JSON.parse(stored)
      } catch {
        sessions = []
      }
    }

    const existingIndex = sessions.findIndex((s) => s.id === sessionId)
    const title = msgs[0]?.content?.slice(0, 50) || 'Новый чат'
    const now = new Date().toISOString()

    if (existingIndex >= 0) {
      sessions[existingIndex] = {
        ...sessions[existingIndex],
        messages: msgs,
        title,
        updated_at: now,
      }
    } else {
      sessions.unshift({
        id: sessionId,
        title,
        messages: msgs,
        created_at: now,
        updated_at: now,
      })
    }

    localStorage.setItem('guest_sessions', JSON.stringify(sessions))
  }

  const handleSend = async (message: string, images: File[], settings: AnalysisSettings) => {
    if (!message.trim() && images.length === 0) return

    setIsLoading(true)

    const userMessage: ChatMessageType = {
      id: `temp_${Date.now()}`,
      role: 'user',
      content: message || 'Проанализируй изображение растения.',
      created_at: new Date().toISOString(),
      images:
        images.length > 0
          ? await Promise.all(
              images.map((img) => {
                return new Promise<string>((resolve) => {
                  const reader = new FileReader()
                  reader.onload = (e) => resolve(e.target?.result as string)
                  reader.readAsDataURL(img)
                })
              }),
            )
          : undefined,
    }

    setMessages((prev) => [...prev, userMessage])

    try {
      let assistantMessage: ChatMessageType

      if (images.length === 0) {
        const response = await api.sendMessage({
          message,
          session_id: currentSessionId || undefined,
        })

        if (response.session_id != null) {
          setCurrentSessionId(String(response.session_id))
        }

        assistantMessage = {
          id: `assistant_${Date.now()}`,
          role: 'assistant',
          content: response.assistant_reply,
          created_at: new Date().toISOString(),
        }
      } else if (images.length === 1) {
        const response = await api.analyzeImage(images[0], message, currentSessionId || undefined, settings)

        if (response.session_id != null) {
          setCurrentSessionId(String(response.session_id))
        }

        assistantMessage = {
          id: `assistant_${Date.now()}`,
          role: 'assistant',
          content: response.assistant_reply,
          created_at: new Date().toISOString(),
          analysis_result: response.result,
        }

        if (isGuest && !guestTrialUsed) {
          setGuestTrialUsed()
          setLocalGuestTrialUsed(true)
        }
      } else {
        const response = await api.trackGrowthSeries(images, settings.crop, settings.camera_id)

        assistantMessage = {
          id: `growth_${Date.now()}`,
          role: 'assistant',
          content: `Отслеживание роста завершено: обработано ${response.frames.length} кадров, найдено ${response.tracks.length} треков.`,
          created_at: new Date().toISOString(),
          growth_result: response,
        }

        if (isGuest && !guestTrialUsed) {
          setGuestTrialUsed()
          setLocalGuestTrialUsed(true)
        }
      }

      setMessages((prev) => [...prev, assistantMessage])

      if (isGuest && currentSessionId) {
        saveGuestSession(currentSessionId, [...messages, userMessage, assistantMessage])
      } else if (isGuest && !currentSessionId) {
        const newSessionId = `guest_${Date.now()}`
        setCurrentSessionId(newSessionId)
        saveGuestSession(newSessionId, [userMessage, assistantMessage])
      }
    } catch (error) {
      console.error('[frontend] send message error:', error)
      const errorMessage = error instanceof Error ? error.message : 'Ошибка отправки сообщения'
      toast.error(errorMessage)
      setMessages((prev) => prev.filter((m) => m.id !== userMessage.id))
    } finally {
      setIsLoading(false)
    }
  }

  const handleShareSession = (sessionId: string) => {
    setShareSessionId(sessionId)
  }

  if (authLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-[#0f0f0f]">
        <Loader2 size={40} className="animate-spin text-emerald-500" />
      </div>
    )
  }

  if (!isAuthenticated && !isGuest) {
    return null
  }

  return (
    <div className="min-h-screen bg-[#0f0f0f] flex flex-col">
      <Navbar onMenuClick={() => setSidebarOpen(!sidebarOpen)} />

      <div className="flex flex-1 pt-14">
        <Sidebar
          isOpen={sidebarOpen}
          onClose={() => setSidebarOpen(false)}
          currentSessionId={currentSessionId}
          onSelectSession={handleSelectSession}
          onNewChat={handleNewChat}
          onShareSession={handleShareSession}
        />

        <main className="flex-1 flex flex-col md:ml-[260px]">
          <div className="flex-1 overflow-y-auto p-4">
            {messages.length === 0 ? (
              <div className="h-full flex flex-col items-center justify-center text-center animate-fade-in">
                <div className="w-20 h-20 rounded-full bg-emerald-500/20 flex items-center justify-center mb-6 animate-pulse-glow">
                  <Leaf size={40} className="text-emerald-500" />
                </div>
                <h1 className="text-2xl font-semibold text-white mb-2">Готов, когда вы готовы</h1>
                <p className="text-white/60 max-w-md">
                  Отправьте фото растения для анализа или задайте вопрос текстом.
                  {isGuest && !guestTrialUsed && (
                    <span className="block mt-2 text-emerald-400">
                      В гостевом режиме доступен один пробный анализ изображения.
                    </span>
                  )}
                </p>
              </div>
            ) : (
              <div className="max-w-3xl mx-auto space-y-4">
                {messages.map((msg) => (
                  <ChatMessage key={msg.id} message={msg} />
                ))}

                {isLoading && (
                  <div className="flex gap-3 animate-fade-in">
                    <div className="w-8 h-8 rounded-full bg-white/10 flex items-center justify-center">
                      <Loader2 size={16} className="animate-spin text-white/60" />
                    </div>
                    <div className="flex items-center px-4 py-3 message-assistant rounded-2xl rounded-tl-md">
                      <div className="flex gap-1">
                        <span className="w-2 h-2 bg-white/60 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                        <span className="w-2 h-2 bg-white/60 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                        <span className="w-2 h-2 bg-white/60 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                      </div>
                    </div>
                  </div>
                )}

                <div ref={messagesEndRef} />
              </div>
            )}
          </div>

          <ChatInput
            onSend={handleSend}
            isLoading={isLoading}
            isDisabled={false}
            guestTrialUsed={guestTrialUsed}
            isGuest={isGuest}
          />
        </main>
      </div>

      {shareSessionId && <ShareModal sessionId={shareSessionId} onClose={() => setShareSessionId(null)} />}
    </div>
  )
}
