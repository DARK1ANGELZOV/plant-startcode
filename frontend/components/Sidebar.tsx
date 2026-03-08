'use client'

import { useState, useEffect, useCallback, useMemo } from 'react'
import { useAuth } from '@/contexts/AuthContext'
import { api } from '@/lib/api'
import type { ChatSession, GuestSession, SearchResult } from '@/lib/types'
import { Plus, Search, MessageSquare, Share2, Pencil, Pin, Trash2, MoreHorizontal, X, Loader2 } from 'lucide-react'
import toast from 'react-hot-toast'

interface SidebarProps {
  isOpen: boolean
  onClose: () => void
  currentSessionId: string | null
  onSelectSession: (sessionId: string) => void
  onNewChat: () => void
  onShareSession: (sessionId: string) => void
}

type SidebarMeta = {
  pinned: string[]
  hidden: string[]
  titles: Record<string, string>
}

const DEFAULT_META: SidebarMeta = {
  pinned: [],
  hidden: [],
  titles: {},
}

function textContains(text: string | undefined, query: string): boolean {
  return String(text || '').toLowerCase().includes(query.toLowerCase())
}

function dateValue(value: string | undefined): number {
  const ts = Date.parse(String(value || ''))
  return Number.isFinite(ts) ? ts : 0
}

export function Sidebar({
  isOpen,
  onClose,
  currentSessionId,
  onSelectSession,
  onNewChat,
  onShareSession,
}: SidebarProps) {
  const { isAuthenticated, isGuest, user } = useAuth()
  const [sessions, setSessions] = useState<ChatSession[]>([])
  const [guestSessions, setGuestSessions] = useState<GuestSession[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')
  const [searchResults, setSearchResults] = useState<SearchResult[]>([])
  const [isSearching, setIsSearching] = useState(false)
  const [menuSessionId, setMenuSessionId] = useState<string | null>(null)
  const [meta, setMeta] = useState<SidebarMeta>(DEFAULT_META)

  const storageKey = useMemo(() => {
    if (isAuthenticated) return `sidebar_meta_user_${user?.id ?? 'auth'}`
    return 'sidebar_meta_guest'
  }, [isAuthenticated, user?.id])

  const loadSessions = useCallback(async () => {
    if (isAuthenticated) {
      setIsLoading(true)
      try {
        const data = await api.getSessions()
        setSessions(data)
      } catch {
        toast.error('Ошибка загрузки чатов')
      } finally {
        setIsLoading(false)
      }
    } else if (isGuest) {
      const stored = localStorage.getItem('guest_sessions')
      if (stored) {
        try {
          setGuestSessions(JSON.parse(stored))
        } catch {
          setGuestSessions([])
        }
      } else {
        setGuestSessions([])
      }
    }
  }, [isAuthenticated, isGuest])

  useEffect(() => {
    loadSessions()
  }, [loadSessions, currentSessionId])

  useEffect(() => {
    try {
      const raw = localStorage.getItem(storageKey)
      if (!raw) {
        setMeta(DEFAULT_META)
        return
      }
      const parsed = JSON.parse(raw) as Partial<SidebarMeta>
      setMeta({
        pinned: Array.isArray(parsed.pinned) ? parsed.pinned.map(String) : [],
        hidden: Array.isArray(parsed.hidden) ? parsed.hidden.map(String) : [],
        titles: parsed.titles && typeof parsed.titles === 'object' ? parsed.titles : {},
      })
    } catch {
      setMeta(DEFAULT_META)
    }
  }, [storageKey])

  useEffect(() => {
    localStorage.setItem(storageKey, JSON.stringify(meta))
  }, [meta, storageKey])

  useEffect(() => {
    const query = searchQuery.trim()
    if (!isAuthenticated || !query) {
      setSearchResults([])
      setIsSearching(false)
      return
    }

    const timer = setTimeout(async () => {
      setIsSearching(true)
      try {
        const results = await api.searchMessages(query)
        setSearchResults(results)
      } catch {
        setSearchResults([])
      } finally {
        setIsSearching(false)
      }
    }, 260)

    return () => clearTimeout(timer)
  }, [isAuthenticated, searchQuery])

  useEffect(() => {
    const onPointer = (event: MouseEvent) => {
      const target = event.target as HTMLElement
      if (!target.closest('.chat-menu') && !target.closest('.chat-menu-trigger')) {
        setMenuSessionId(null)
      }
    }
    document.addEventListener('mousedown', onPointer)
    return () => document.removeEventListener('mousedown', onPointer)
  }, [])

  const handleSessionClick = (sessionId: string) => {
    onSelectSession(sessionId)
    onClose()
  }

  const handleNewChat = () => {
    onNewChat()
    onClose()
  }

  const renameSession = (sessionId: string, currentTitle: string) => {
    const value = window.prompt('Введите новое название чата', currentTitle)
    if (value == null) return
    const title = value.trim()
    if (!title) {
      toast.error('Название не может быть пустым')
      return
    }
    setMeta((prev) => ({
      ...prev,
      titles: {
        ...prev.titles,
        [sessionId]: title,
      },
    }))
    setMenuSessionId(null)
  }

  const togglePin = (sessionId: string) => {
    setMeta((prev) => {
      const pinned = prev.pinned.includes(sessionId)
        ? prev.pinned.filter((id) => id !== sessionId)
        : [...prev.pinned, sessionId]
      return { ...prev, pinned }
    })
    setMenuSessionId(null)
  }

  const deleteSession = async (sessionId: string) => {
    if (!window.confirm('Удалить чат навсегда?')) return
    setMenuSessionId(null)

    if (isAuthenticated) {
      try {
        await api.deleteSession(sessionId)
        setSessions((prev) => prev.filter((s) => String(s.id) !== sessionId))
      } catch {
        toast.error('Не удалось удалить чат')
        return
      }
    } else if (isGuest) {
      const next = guestSessions.filter((s) => String(s.id) !== sessionId)
      setGuestSessions(next)
      localStorage.setItem('guest_sessions', JSON.stringify(next))
    }

    setMeta((prev) => {
      const titles = { ...prev.titles }
      delete titles[sessionId]
      return {
        ...prev,
        hidden: prev.hidden.filter((id) => id !== sessionId),
        pinned: prev.pinned.filter((id) => id !== sessionId),
        titles,
      }
    })

    if (currentSessionId === sessionId) {
      onNewChat()
    }
  }

  const shareSession = (sessionId: string) => {
    if (!isAuthenticated) {
      toast('Поделиться чатом можно после входа в аккаунт')
      setMenuSessionId(null)
      return
    }
    onShareSession(sessionId)
    setMenuSessionId(null)
  }

  const hasSearch = searchQuery.trim().length > 0
  const messageMatchIds = useMemo(
    () => new Set(searchResults.map((item) => String(item.session_id))),
    [searchResults],
  )

  const authRows = useMemo(() => {
    const query = searchQuery.trim()
    const rows = sessions
      .map((s) => {
        const id = String(s.id)
        return {
          id,
          title: meta.titles[id] || s.title || 'Новый чат',
          updatedAt: s.updated_at || s.created_at,
        }
      })
      .filter((row) => !meta.hidden.includes(row.id))

    const filtered = query
      ? rows.filter((row) => textContains(row.title, query) || messageMatchIds.has(row.id))
      : rows

    return [...filtered].sort((a, b) => {
      const aPinned = meta.pinned.includes(a.id) ? 1 : 0
      const bPinned = meta.pinned.includes(b.id) ? 1 : 0
      if (aPinned !== bPinned) return bPinned - aPinned
      return dateValue(b.updatedAt) - dateValue(a.updatedAt)
    })
  }, [sessions, meta, searchQuery, messageMatchIds])

  const guestRows = useMemo(() => {
    const query = searchQuery.trim()
    const rows = guestSessions
      .map((s) => {
        const id = String(s.id)
        return {
          id,
          title: meta.titles[id] || s.title || 'Новый чат',
          updatedAt: s.updated_at || s.created_at,
          messages: s.messages || [],
        }
      })
      .filter((row) => !meta.hidden.includes(row.id))

    const filtered = query
      ? rows.filter(
          (row) =>
            textContains(row.title, query) ||
            row.messages.some((m) => textContains(m.content, query)),
        )
      : rows

    return [...filtered].sort((a, b) => {
      const aPinned = meta.pinned.includes(a.id) ? 1 : 0
      const bPinned = meta.pinned.includes(b.id) ? 1 : 0
      if (aPinned !== bPinned) return bPinned - aPinned
      return dateValue(b.updatedAt) - dateValue(a.updatedAt)
    })
  }, [guestSessions, meta, searchQuery])

  const renderRow = (sessionId: string, title: string) => {
    const isPinned = meta.pinned.includes(sessionId)

    return (
      <div
        key={sessionId}
        className={`group relative flex items-center gap-2 px-3 py-2 rounded-lg cursor-pointer transition-all ${
          currentSessionId === sessionId
            ? 'bg-emerald-500/20 text-emerald-400'
            : 'hover:bg-white/10 text-white/80'
        }`}
      >
        <button
          onClick={() => handleSessionClick(sessionId)}
          className="flex items-center gap-2 flex-1 min-w-0 text-left"
        >
          <MessageSquare size={16} className="flex-shrink-0" />
          <span className="text-sm truncate">{isPinned ? `📌 ${title}` : title}</span>
        </button>

        <button
          onClick={(e) => {
            e.stopPropagation()
            setMenuSessionId((prev) => (prev === sessionId ? null : sessionId))
          }}
          className="chat-menu-trigger opacity-0 group-hover:opacity-100 p-1 hover:bg-white/10 rounded transition-all"
          title="Действия чата"
        >
          <MoreHorizontal size={15} />
        </button>

        {menuSessionId === sessionId && (
          <div className="chat-menu absolute right-1 top-10 z-20 w-44 rounded-xl border border-white/10 bg-[#1a1a1d] shadow-xl p-1">
            <button
              onClick={() => shareSession(sessionId)}
              className="w-full px-3 py-2 rounded-lg hover:bg-white/10 text-sm text-left inline-flex items-center gap-2"
            >
              <Share2 size={14} />
              Поделиться
            </button>
            <button
              onClick={() => renameSession(sessionId, title)}
              className="w-full px-3 py-2 rounded-lg hover:bg-white/10 text-sm text-left inline-flex items-center gap-2"
            >
              <Pencil size={14} />
              Переименовать
            </button>
            <button
              onClick={() => togglePin(sessionId)}
              className="w-full px-3 py-2 rounded-lg hover:bg-white/10 text-sm text-left inline-flex items-center gap-2"
            >
              <Pin size={14} />
              {isPinned ? 'Открепить чат' : 'Закрепить чат'}
            </button>
            <button
              onClick={() => deleteSession(sessionId)}
              className="w-full px-3 py-2 rounded-lg hover:bg-red-500/15 text-red-300 text-sm text-left inline-flex items-center gap-2"
            >
              <Trash2 size={14} />
              Удалить
            </button>
          </div>
        )}
      </div>
    )
  }

  return (
    <>
      {isOpen && <div className="fixed inset-0 bg-black/50 z-40 md:hidden" onClick={onClose} />}

      <aside
        className={`fixed top-14 left-0 bottom-0 w-[260px] glass border-r border-white/10 z-50 transform transition-transform duration-300 ${
          isOpen ? 'translate-x-0' : '-translate-x-full md:translate-x-0'
        }`}
      >
        <div className="flex flex-col h-full p-3">
          <button
            onClick={handleNewChat}
            className="flex items-center gap-3 w-full px-4 py-3 mb-3 border border-emerald-500/50 hover:bg-emerald-500/10 rounded-lg transition-all duration-200 text-emerald-500"
          >
            <Plus size={20} />
            <span className="font-medium">Новый чат</span>
          </button>

          <div className="relative mb-3">
            <Search size={18} className="absolute left-3 top-1/2 -translate-y-1/2 text-white/40" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Поиск по чатам"
              className="w-full pl-10 pr-10 py-2 bg-white/5 border border-white/10 rounded-lg text-white placeholder:text-white/40 focus:outline-none focus:border-emerald-500 transition-all text-sm"
            />
            {searchQuery && (
              <button
                onClick={() => {
                  setSearchQuery('')
                  setSearchResults([])
                }}
                className="absolute right-2 top-1/2 -translate-y-1/2 text-white/40 hover:text-white/80"
                aria-label="Очистить поиск"
              >
                <X size={16} />
              </button>
            )}
          </div>

          <p className="px-3 pb-2 text-[11px] uppercase tracking-wide text-white/40">
            История чатов {isSearching && <Loader2 size={12} className="inline ml-1 animate-spin" />}
          </p>

          <div className="flex-1 overflow-y-auto">
            {isAuthenticated ? (
              isLoading ? (
                <div className="flex items-center justify-center py-8">
                  <Loader2 size={24} className="animate-spin text-white/60" />
                </div>
              ) : authRows.length > 0 ? (
                <div className="space-y-1">{authRows.map((row) => renderRow(row.id, row.title))}</div>
              ) : (
                <p className="text-center text-white/40 text-sm py-3">
                  {hasSearch ? 'Чаты не найдены' : 'Чатов пока нет'}
                </p>
              )
            ) : isGuest ? (
              guestRows.length > 0 ? (
                <div className="space-y-1">{guestRows.map((row) => renderRow(row.id, row.title))}</div>
              ) : (
                <p className="text-center text-white/40 text-sm py-3">
                  {hasSearch ? 'Совпадений нет' : 'История пуста'}
                </p>
              )
            ) : (
              <p className="text-center text-white/40 text-sm py-8">
                Войдите, чтобы сохранять историю
              </p>
            )}

            {isAuthenticated && hasSearch && (
              <div className="mt-3 border-t border-white/10 pt-2">
                <p className="px-3 py-1 text-[11px] uppercase tracking-wide text-white/40">
                  Совпадения в сообщениях
                </p>
                {searchResults.length > 0 ? (
                  <div className="space-y-1">
                    {searchResults.map((result) => (
                      <button
                        key={result.message_id}
                        onClick={() => handleSessionClick(String(result.session_id))}
                        className="w-full px-3 py-2 text-left hover:bg-white/10 rounded-lg transition-colors"
                      >
                        <p className="text-sm text-white/80 line-clamp-1">{result.title || 'Чат'}</p>
                        <p className="text-xs text-white/60 line-clamp-2">{result.excerpt}</p>
                      </button>
                    ))}
                  </div>
                ) : (
                  <p className="px-3 py-2 text-xs text-white/50">Совпадений по сообщениям нет</p>
                )}
              </div>
            )}
          </div>
        </div>
      </aside>
    </>
  )
}
