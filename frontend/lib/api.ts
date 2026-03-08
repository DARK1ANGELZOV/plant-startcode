import axios, { AxiosError, AxiosInstance } from 'axios'
import type {
  AnalyticsRun,
  AnalysisSettings,
  AuthResponse,
  ChangePasswordRequest,
  ChatAnalyzeResponse,
  ChatMessage,
  ChatMessageRequest,
  ChatSession,
  ChatTextResponse,
  GrowthTrackResponse,
  LoginRequest,
  RegisterRequest,
  SearchResult,
  ShareData,
} from './types'

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

function asStringId(value: string | number | null | undefined): string {
  return String(value ?? '')
}

function asIntSessionId(value: string | number | null | undefined): number | null {
  if (value == null) return null
  if (typeof value === 'number') {
    return Number.isInteger(value) && value > 0 ? value : null
  }
  const raw = String(value).trim()
  if (!/^\d+$/.test(raw)) return null
  const parsed = Number(raw)
  return Number.isInteger(parsed) && parsed > 0 ? parsed : null
}

function asSafeString(value: unknown, fallback = ''): string {
  if (typeof value === 'string') {
    const trimmed = value.trim()
    return trimmed.length > 0 ? trimmed : fallback
  }
  if (value == null) return fallback
  return String(value)
}

function asFiniteNumber(value: unknown, fallback: number): number {
  if (typeof value === 'number' && Number.isFinite(value)) return value
  if (typeof value === 'string') {
    const normalized = value.trim().replace(',', '.')
    const parsed = Number(normalized)
    if (Number.isFinite(parsed)) return parsed
  }
  return fallback
}

function asFiniteInt(value: unknown, fallback: number): number {
  const n = asFiniteNumber(value, fallback)
  const int = Math.round(n)
  return Number.isFinite(int) && int > 0 ? int : fallback
}

function extractApiDetail(data: unknown): string {
  if (!data) return ''
  if (typeof data === 'string') return data
  if (typeof data === 'object') {
    const detail = (data as { detail?: unknown }).detail
    if (typeof detail === 'string') return detail
    if (Array.isArray(detail)) {
      const parts = detail
        .map((item) => {
          if (typeof item === 'string') return item
          if (item && typeof item === 'object') {
            const message = (item as { msg?: unknown }).msg
            return typeof message === 'string' ? message : ''
          }
          return ''
        })
        .filter(Boolean)
      return parts.join('; ')
    }
  }
  return ''
}

function toClientError(error: unknown, fallback: string): Error {
  if (axios.isAxiosError(error)) {
    const status = error.response?.status
    const detail = extractApiDetail(error.response?.data)
    if (!error.response) return new Error('Сервер недоступен. Проверьте, что backend запущен на порту 8000.')
    if (detail) return new Error(detail)
    return new Error(`${fallback} (HTTP ${status ?? 'unknown'})`)
  }
  if (error instanceof Error) return error
  return new Error(fallback)
}

export function toAssetUrl(rawPath?: string | null): string {
  if (!rawPath) return ''
  if (rawPath.startsWith('http://') || rawPath.startsWith('https://')) return rawPath

  const normalized = String(rawPath).replaceAll('\\', '/')
  if (normalized.startsWith('/outputs/')) return `${API_BASE_URL}${normalized}`
  if (normalized.startsWith('outputs/')) return `${API_BASE_URL}/${normalized}`

  const idx = normalized.toLowerCase().lastIndexOf('/outputs/')
  if (idx >= 0) return `${API_BASE_URL}${normalized.slice(idx)}`
  return `${API_BASE_URL}/${normalized.replace(/^\/+/, '')}`
}

class ApiClient {
  private client: AxiosInstance
  private token: string | null = null
  private userId: number | null = null

  constructor() {
    this.client = axios.create({
      baseURL: API_BASE_URL,
      timeout: 60_000,
    })

    this.client.interceptors.request.use((config) => {
      if (this.token) {
        config.headers.Authorization = `Bearer ${this.token}`
      }

      if (this.userId) {
        config.headers['X-Tenant-ID'] = `user_${this.userId}`
      }
      return config
    })

    this.client.interceptors.response.use(
      (response) => response,
      async (error: AxiosError) => {
        const hadToken = !!this.token
        const status = error.response?.status
        const requestConfig = error.config as (typeof error.config & { _guestRetry?: boolean }) | undefined
        const url = requestConfig?.url || ''
        const isGuestAllowedEndpoint = typeof url === 'string' && (url.includes('/chat/analyze') || url.includes('/chat/message'))

        if (status === 401 && hadToken && requestConfig && isGuestAllowedEndpoint && !requestConfig._guestRetry) {
          this.clearAuth()
          requestConfig._guestRetry = true
          if (requestConfig.headers) {
            const headers = requestConfig.headers as Record<string, unknown> & { set?: (name: string, value: unknown) => void }
            if (typeof headers.set === 'function') {
              headers.set('Authorization', undefined)
            }
            delete headers.Authorization
          }
          return this.client.request(requestConfig)
        }

        if (status === 401) {
          this.clearAuth()
          if (hadToken && typeof window !== 'undefined' && !window.location.pathname.startsWith('/login')) {
            window.location.href = '/login'
          }
        }
        return Promise.reject(error)
      },
    )

    if (typeof window !== 'undefined') {
      const savedToken = localStorage.getItem('auth_token')
      const savedUserId = localStorage.getItem('user_id')
      if (savedToken) this.token = savedToken
      if (savedUserId) this.userId = parseInt(savedUserId, 10)
    }
  }

  setAuth(token: string, userId: number) {
    this.token = token
    this.userId = userId
    if (typeof window !== 'undefined') {
      localStorage.setItem('auth_token', token)
      localStorage.setItem('user_id', userId.toString())
    }
  }

  clearAuth() {
    this.token = null
    this.userId = null
    if (typeof window !== 'undefined') {
      localStorage.removeItem('auth_token')
      localStorage.removeItem('user_id')
    }
  }

  isAuthenticated(): boolean {
    return !!this.token
  }

  getBaseUrl(): string {
    return API_BASE_URL
  }

  async register(data: RegisterRequest): Promise<AuthResponse> {
    const response = await this.client.post<AuthResponse>('/auth/register', data)
    return response.data
  }

  async login(data: LoginRequest): Promise<AuthResponse> {
    const response = await this.client.post<AuthResponse>('/auth/login', data)
    return response.data
  }

  async getMe(): Promise<{ id: number; email: string }> {
    const response = await this.client.get('/auth/me')
    return response.data
  }

  async changePassword(data: ChangePasswordRequest): Promise<void> {
    await this.client.post('/auth/change-password', data)
  }

  async getSessions(): Promise<ChatSession[]> {
    const response = await this.client.get<ChatSession[]>('/chat/sessions', { params: { limit: 300 } })
    return (response.data || []).map((row) => ({
      ...row,
      id: asStringId((row as unknown as { id: string | number }).id),
    }))
  }

  async createSession(title = 'Новый чат'): Promise<ChatSession> {
    const response = await this.client.post<ChatSession>('/chat/sessions', { title })
    return {
      ...response.data,
      id: asStringId((response.data as unknown as { id: string | number }).id),
    }
  }

  async deleteSession(sessionId: string): Promise<void> {
    await this.client.delete(`/chat/sessions/${sessionId}`)
  }

  async getSessionMessages(sessionId: string): Promise<ChatMessage[]> {
    const response = await this.client.get<ChatMessage[]>(`/chat/sessions/${sessionId}/messages`, {
      params: { limit: 500 },
    })
    return (response.data || []).map((msg) => ({
      ...msg,
      id: asStringId((msg as unknown as { id: string | number }).id),
    }))
  }

  async sendMessage(data: ChatMessageRequest): Promise<ChatTextResponse> {
    const payload: { message: string; session_id?: number } = { message: data.message }
    const sessionId = asIntSessionId(data.session_id)
    if (sessionId !== null) payload.session_id = sessionId
    try {
      const response = await this.client.post<ChatTextResponse>('/chat/message', payload)
      return response.data
    } catch (error) {
      if (axios.isAxiosError(error) && sessionId !== null) {
        const status = error.response?.status
        const detail = extractApiDetail(error.response?.data)
        if (status === 404 && /chat session not found/i.test(detail)) {
          const retryPayload: { message: string } = { message: data.message }
          const retryResponse = await this.client.post<ChatTextResponse>('/chat/message', retryPayload)
          return retryResponse.data
        }
      }
      throw toClientError(error, 'Ошибка отправки сообщения')
    }
  }

  async analyzeImage(
    image: File,
    message: string,
    sessionId: string | undefined,
    settings: AnalysisSettings,
  ): Promise<ChatAnalyzeResponse> {
    const validSessionId = asIntSessionId(sessionId)

    const createFormData = (sid: number | null): FormData => {
      const formData = new FormData()
      formData.append('image', image)
      formData.append('message', asSafeString(message, 'Проведи диагностику растения по фото.'))
      if (sid !== null) {
        formData.append('session_id', String(sid))
      }
      formData.append('crop', asSafeString(settings.crop, 'Unknown'))
      formData.append('conf', asFiniteNumber(settings.conf, 0.08).toString())
      formData.append('iou', asFiniteNumber(settings.iou, 0.45).toString())
      formData.append('max_det', asFiniteInt(settings.max_det, 60).toString())
      formData.append('use_ensemble', Boolean(settings.use_ensemble).toString())
      formData.append('source_type', asSafeString(settings.source_type, 'lab_camera'))
      formData.append('camera_id', asSafeString(settings.camera_id, 'default'))
      return formData
    }

    try {
      const response = await this.client.post<ChatAnalyzeResponse>('/chat/analyze', createFormData(validSessionId))
      return response.data
    } catch (error) {
      if (axios.isAxiosError(error) && validSessionId !== null) {
        const status = error.response?.status
        const detail = extractApiDetail(error.response?.data)
        if (status === 404 && /chat session not found/i.test(detail)) {
          const retryResponse = await this.client.post<ChatAnalyzeResponse>('/chat/analyze', createFormData(null))
          return retryResponse.data
        }
      }
      throw toClientError(error, 'Ошибка анализа изображения')
    }
  }

  async trackGrowthSeries(images: File[], crop: string, cameraId: string): Promise<GrowthTrackResponse> {
    const formData = new FormData()
    images.forEach((image) => {
      formData.append('images', image)
    })
    formData.append('crop', crop)
    formData.append('camera_id', cameraId)

    try {
      const response = await this.client.post<GrowthTrackResponse>('/growth/track-series', formData)
      return response.data
    } catch (error) {
      throw toClientError(error, 'Ошибка отслеживания роста')
    }
  }

  async searchMessages(query: string): Promise<SearchResult[]> {
    const response = await this.client.get<SearchResult[]>('/chat/search', {
      params: { query, limit: 80 },
    })
    return (response.data || []).map((row) => ({
      ...row,
      session_id: asStringId((row as unknown as { session_id: string | number }).session_id),
      message_id: asStringId((row as unknown as { message_id: string | number }).message_id),
    }))
  }

  async getShareData(sessionId: string): Promise<ShareData> {
    const response = await this.client.get<Omit<ShareData, 'share_url'>>(`/chat/share/${sessionId}`)
    const data = response.data
    const shareUrl =
      typeof window !== 'undefined'
        ? `${window.location.origin}/?share=${data.session_id}`
        : `${API_BASE_URL}/chat/share/${data.session_id}`

    return {
      ...data,
      session_id: asStringId((data as unknown as { session_id: string | number }).session_id),
      messages: (data.messages || []).map((msg) => ({
        ...msg,
        id: asStringId((msg as unknown as { id: string | number }).id),
      })),
      share_url: shareUrl,
    }
  }

  async getAnalyticsRuns(): Promise<AnalyticsRun[]> {
    const response = await this.client.get<AnalyticsRun[]>('/analytics/runs')
    return response.data
  }
}

export const api = new ApiClient()
export { API_BASE_URL }

