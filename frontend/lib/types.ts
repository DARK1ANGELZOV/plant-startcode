// Auth types
export interface User {
  id: number
  email: string
}

export interface AuthResponse {
  access_token: string
  token_type: string
  user: User
}

export interface RegisterRequest {
  email: string
  password: string
}

export interface LoginRequest {
  email: string
  password: string
}

export interface ChangePasswordRequest {
  old_password: string
  new_password: string
}

// Chat types
export interface ChatSession {
  id: string
  title: string
  created_at: string
  updated_at: string
}

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  created_at: string
  run_id?: string | null
  images?: string[]
  analysis_result?: PredictResponse
  growth_result?: GrowthTrackResponse
}

export interface ChatMessageRequest {
  message: string
  session_id?: string
}

export interface ChatTextResponse {
  assistant_reply: string
  session_id?: number | null
}

// Predict types
export interface PredictResponse {
  run_id: string
  scale_mm_per_px: number
  scale_source: string
  measurements: Measurement[]
  summary: PredictSummary
  recommendations: Recommendation[]
  disease_analysis?: DiseaseAnalysis
  phi: PHIResult
  explainability?: ExplainabilityArtifacts
  files: PredictFiles
}

export interface PredictFiles {
  input?: string
  overlay: string
  csv?: string
  distribution_plot?: string
  pdf_report?: string
  xai_confidence_map?: string | null
  xai_attention?: string | null
  xai_gradcam?: string | null
  xai_uncertainty?: string | null
}

export interface Recommendation {
  severity: 'ok' | 'warning' | 'critical'
  message: string
  action: string
}

export interface DiseaseAnalysis {
  risk_level?: string
  confidence?: number
  findings?: string[]
  actions?: string[]
}

export interface ExplainabilityArtifacts {
  confidence_map?: string | null
  attention_heatmap?: string | null
  gradcam?: string | null
  uncertainty_map?: string | null
}

export interface PHIResult {
  score: number
  status: string
  reasons?: string[]
}

export interface Measurement {
  instance_id: number
  crop: string
  class_name: string
  confidence: number
  area_px: number
  area_mm2?: number | null
  length_px: number
  length_mm?: number | null
  reliable: boolean
}

export interface PredictSummary {
  count?: number
  image_quality?: ImageQuality
  measurement_trust_level?: string
  measurement_trust_score?: number
  mm_conversion_possible?: boolean
  calibration_note?: string
}

export interface ImageQuality {
  blur_score?: number
  brightness?: number
  contrast?: number
  notes: string[]
}

// Growth tracking types
export interface GrowthTrackResponse {
  series_id: string
  crop: string
  frames: GrowthFrame[]
  tracks: GrowthTrack[]
}

export interface GrowthFrame {
  frame_index: number
  timestamp: string
  count: number
  phi: PHIResult
}

export interface GrowthTrack {
  track_id: number
  class_name: string
  points: GrowthDataPoint[]
  growth_rate_mm_per_day: number
  smoothed_length_mm: number[]
}

export interface GrowthDataPoint {
  frame_index: number
  timestamp: string
  length_mm: number
  area_mm2: number
  confidence: number
}

// Analytics types
export interface AnalyticsRun {
  run_id: string
  created_at: string
  tenant_id?: string
}

// Search types
export interface SearchResult {
  session_id: string
  title: string
  message_id: string
  excerpt: string
  created_at: string
}

// Share types
export interface ShareData {
  session_id: string
  title: string
  messages: ChatMessage[]
  share_url: string
}

// Analysis settings
export interface AnalysisSettings {
  crop: string
  conf: number
  iou: number
  max_det: number
  use_ensemble: boolean
  source_type: string
  camera_id: string
}

export interface ChatAnalyzeResponse {
  assistant_reply: string
  result: PredictResponse
  session_id?: number | null
}

export const DEFAULT_ANALYSIS_SETTINGS: AnalysisSettings = {
  crop: 'Unknown',
  conf: 0.08,
  iou: 0.45,
  max_det: 60,
  use_ensemble: false,
  source_type: 'lab_camera',
  camera_id: 'default',
}

export const CROP_OPTIONS = [
  { value: 'Unknown', label: 'Авто' },
  { value: 'Wheat', label: 'Пшеница' },
  { value: 'Arugula', label: 'Руккола' },
]

// Local storage types for guests
export interface GuestSession {
  id: string
  title: string
  messages: ChatMessage[]
  created_at: string
  updated_at: string
}

// Legend colors for segmentation
export const CLASS_COLORS: Record<string, string> = {
  root: '#ef4444',
  stem: '#10b981',
  leaves: '#3b82f6',
}

export const CLASS_LABELS: Record<string, string> = {
  root: 'Корень',
  stem: 'Стебель',
  leaves: 'Листья',
}
