from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, EmailStr, Field


class PlantMeasurement(BaseModel):
    instance_id: int
    crop: str = 'Unknown'
    class_name: str
    confidence: float
    area_px: int
    area_mm2: float | None = None
    length_px: float
    length_mm: float | None = None
    reliable: bool = True


class Recommendation(BaseModel):
    severity: Literal['ok', 'warning', 'critical']
    message: str
    action: str


class PHIResult(BaseModel):
    score: float = 0.0
    status: Literal['Healthy', 'Risk', 'Critical'] = 'Risk'
    components: dict[str, float] = Field(default_factory=dict)
    reasons: list[str] = Field(default_factory=list)


class ExplainabilityArtifacts(BaseModel):
    confidence_map: str | None = None
    attention_heatmap: str | None = None
    gradcam: str | None = None
    uncertainty_map: str | None = None
    notes: list[str] = Field(default_factory=list)


class ActiveLearningSummary(BaseModel):
    collected: int = 0
    threshold: float = 0.0
    queue_items: list[str] = Field(default_factory=list)


class PredictResponse(BaseModel):
    run_id: str
    scale_mm_per_px: float
    scale_source: str
    measurements: list[PlantMeasurement]
    summary: dict
    recommendations: list[Recommendation] = Field(default_factory=list)
    disease_analysis: dict = Field(default_factory=dict)
    phi: PHIResult = Field(default_factory=PHIResult)
    explainability: ExplainabilityArtifacts = Field(default_factory=ExplainabilityArtifacts)
    active_learning: ActiveLearningSummary = Field(default_factory=ActiveLearningSummary)
    files: dict


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str
    storage_free_gb: float | None = None
    storage_total_gb: float | None = None
    storage_low_space: bool = False
    storage_min_free_gb: float | None = None
    run_dirs_count: int | None = None


class FineTuneRequest(BaseModel):
    data_yaml: str
    epochs: int = 30
    batch: int = 8
    imgsz: int = 640
    name: str = 'api_finetune'


class FineTuneResponse(BaseModel):
    status: str
    command: str


class ChatAnalyzeResponse(BaseModel):
    assistant_reply: str
    result: PredictResponse
    session_id: int | None = None


class ChatTextRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    session_id: int | None = None


class ChatTextResponse(BaseModel):
    assistant_reply: str
    session_id: int | None = None


class ChatSearchHitResponse(BaseModel):
    session_id: int
    title: str
    message_id: int
    excerpt: str
    created_at: str


class UserRegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)


class UserLoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)


class ChangePasswordRequest(BaseModel):
    old_password: str = Field(min_length=8, max_length=128)
    new_password: str = Field(min_length=8, max_length=128)


class UserResponse(BaseModel):
    id: int
    email: EmailStr


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = 'bearer'
    user: UserResponse


class ChatSessionCreateRequest(BaseModel):
    title: str = Field(default='Новый чат', min_length=1, max_length=255)


class ChatSessionResponse(BaseModel):
    id: int
    title: str
    created_at: str
    updated_at: str


class ChatMessageResponse(BaseModel):
    id: int
    role: str
    content: str
    run_id: str | None = None
    created_at: str


class ShareChatResponse(BaseModel):
    session_id: int
    title: str
    messages: list[ChatMessageResponse]


class RunRecord(BaseModel):
    run_id: str
    created_at: str
    tenant_id: str = 'default'
    source_type: str = 'unknown'
    camera_id: str = 'default'
    scale_mm_per_px: float
    scale_source: str
    count: int
    crops: list[str] = Field(default_factory=list)
    traits: dict = Field(default_factory=dict)
    summary: dict = Field(default_factory=dict)
    files: dict = Field(default_factory=dict)


class TrendPoint(BaseModel):
    run_id: str
    created_at: str
    value: float


class TrendResponse(BaseModel):
    crop: str
    class_name: str
    metric: str
    points: list[TrendPoint]
    slope_per_step: float
    latest_value: float | None = None


class CompareRunsResponse(BaseModel):
    run_a: str
    run_b: str
    crop: str
    class_name: str
    metric: str
    value_a: float
    value_b: float
    delta: float
    delta_pct: float


class RegisterModelRequest(BaseModel):
    path: str
    metrics: dict[str, float] = Field(default_factory=dict)
    dataset_version: str | None = None
    tags: list[str] = Field(default_factory=list)
    source: str = 'manual'


class ModelVersionEntry(BaseModel):
    version_id: str
    created_at: str
    path: str
    metrics: dict[str, float] = Field(default_factory=dict)
    dataset_version: str | None = None
    tags: list[str] = Field(default_factory=list)
    source: str = 'manual'


class RegisterDatasetRequest(BaseModel):
    dataset_version: str
    source: str
    task_type: str = 'instance_segmentation'
    classes: list[str] = Field(default_factory=lambda: ['root', 'stem', 'leaves'])
    augmentation: dict = Field(default_factory=dict)
    notes: str = ''


class DatasetVersionEntry(BaseModel):
    dataset_version: str
    created_at: str
    source: str
    task_type: str
    classes: list[str] = Field(default_factory=list)
    augmentation: dict = Field(default_factory=dict)
    notes: str = ''


class BlindEvalRequest(BaseModel):
    data_yaml: str
    split: Literal['train', 'val', 'test'] = 'val'
    max_images: int = 200
    iou_sla: float = 0.5


class JobEnqueueResponse(BaseModel):
    job_id: str
    status: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    result: dict | None = None
    error: str | None = None


class RobustnessPoint(BaseModel):
    level: int
    miou: float
    score: float


class RobustnessResponse(BaseModel):
    run_id: str
    baseline_instances: int
    curves: dict[str, list[RobustnessPoint]] = Field(default_factory=dict)
    robustness_score: float = 0.0
    files: dict = Field(default_factory=dict)


class GrowthTrackPoint(BaseModel):
    frame_index: int
    timestamp: str
    length_mm: float
    area_mm2: float
    confidence: float


class GrowthTrack(BaseModel):
    track_id: int
    class_name: str
    points: list[GrowthTrackPoint] = Field(default_factory=list)
    growth_rate_mm_per_day: float = 0.0
    smoothed_length_mm: list[float] = Field(default_factory=list)


class GrowthFrameSummary(BaseModel):
    frame_index: int
    timestamp: str
    count: int
    phi: PHIResult = Field(default_factory=PHIResult)


class GrowthTrackingResponse(BaseModel):
    series_id: str
    crop: str
    tracks: list[GrowthTrack] = Field(default_factory=list)
    frames: list[GrowthFrameSummary] = Field(default_factory=list)
    files: dict = Field(default_factory=dict)


class ActiveLearningItem(BaseModel):
    item_id: str
    status: str
    run_id: str
    class_name: str
    confidence: float
    metadata_path: str
    created_at: str


class ActiveLearningQueueResponse(BaseModel):
    status: str
    count: int
    items: list[ActiveLearningItem] = Field(default_factory=list)
