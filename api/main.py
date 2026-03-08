from __future__ import annotations

import asyncio
import errno
import logging
import logging.handlers
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from prometheus_fastapi_instrumentator import Instrumentator
from sqlalchemy.orm import Session

from calibration.chessboard import ScaleCalibrator
from db.database import get_db, init_db
from db.models import User
from services.active_learning_service import ActiveLearningService
from services.auth_service import AuthService
from services.chat_service import ChatService
from services.dataset_registry_service import DatasetRegistryService
from services.evaluation_service import BlindEvaluationService
from services.growth_tracking_service import GrowthTrackingService
from services.history_service import RunHistoryService
from services.inference_service import InferenceService
from services.insight_service import InsightService
from services.job_service import JobService
from services.model_registry_service import ModelRegistryService
from services.model_service import ModelService
from services.plantcv_service import PlantCVService
from services.consultation_service import ConsultationService
from services.phi_service import PHIService
from services.recommendation_service import RecommendationService
from services.report_service import ReportService
from services.robustness_service import RobustnessService
from services.storage_service import StorageService
from services.xai_service import XAIService
from utils.config import load_app_config
from utils.errors import AgroAIError
from utils.image_io import decode_image_bytes
from utils.logging import setup_logging
from utils.schemas import (
    ActiveLearningQueueResponse,
    BlindEvalRequest,
    CalibrationProfileListResponse,
    CalibrationProfileResponse,
    ChatAnalyzeResponse,
    ChatTextRequest,
    ChatTextResponse,
    ChatSearchHitResponse,
    ChatMessageResponse,
    ChatSessionCreateRequest,
    ChatSessionResponse,
    CompareRunsResponse,
    ChangePasswordRequest,
    DatasetVersionEntry,
    FineTuneRequest,
    FineTuneResponse,
    HealthResponse,
    GrowthTrackingResponse,
    JobEnqueueResponse,
    JobStatusResponse,
    ModelVersionEntry,
    PredictResponse,
    RegisterDatasetRequest,
    RegisterModelRequest,
    RunRecord,
    RobustnessResponse,
    ShareChatResponse,
    TokenResponse,
    TrendResponse,
    UserLoginRequest,
    UserRegisterRequest,
    UserResponse,
)
from utils.seed import set_global_seed


logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parents[1]
IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}


def _resolve_local_path(raw_path: str) -> Path:
    p = Path(str(raw_path).strip())
    if p.is_absolute():
        return p
    return (BASE_DIR / p).resolve()


def _expand_image_candidates(items: list[str], max_images: int) -> list[Path]:
    found: list[Path] = []
    seen: set[str] = set()
    for raw in items:
        raw = str(raw).strip()
        if not raw:
            continue
        p = _resolve_local_path(raw)
        if ('*' in raw or '?' in raw or '[' in raw) and (not p.exists()):
            for gp in sorted(BASE_DIR.glob(raw)):
                if gp.is_file() and gp.suffix.lower() in IMAGE_EXTS:
                    key = str(gp.resolve())
                    if key not in seen:
                        seen.add(key)
                        found.append(gp.resolve())
                if len(found) >= max_images:
                    return found
            continue
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            key = str(p.resolve())
            if key not in seen:
                seen.add(key)
                found.append(p.resolve())
        elif p.is_dir():
            for img in sorted(p.rglob('*')):
                if not img.is_file() or img.suffix.lower() not in IMAGE_EXTS:
                    continue
                key = str(img.resolve())
                if key in seen:
                    continue
                seen.add(key)
                found.append(img.resolve())
                if len(found) >= max_images:
                    return found
    return found


def _resolve_bootstrap_camera_id(config: dict, camera_id: str, source_type: str) -> str:
    cid = str(camera_id or 'default').strip() or 'default'
    if cid != 'default':
        return cid
    profile_map = config.get('calibration', {}).get('default_camera_profiles', {}) or {}
    src = str(source_type or 'unknown').strip().lower()
    mapped = profile_map.get(src) or profile_map.get('default')
    if isinstance(mapped, str) and mapped.strip():
        return mapped.strip()
    return cid


def _bootstrap_calibration_profiles(config: dict, calibrator: ScaleCalibrator) -> None:
    boot_cfg = config.get('calibration', {}).get('startup_bootstrap', {}) or {}
    enabled = bool(boot_cfg.get('enabled', False))
    if not enabled:
        return
    only_if_missing = bool(boot_cfg.get('only_if_missing', True))
    max_images = max(1, int(boot_cfg.get('max_images_per_profile', 15)))
    profiles = boot_cfg.get('profiles', []) or []
    if not isinstance(profiles, list):
        return

    for row in profiles:
        if not isinstance(row, dict):
            continue
        raw_camera_id = str(row.get('camera_id', 'default'))
        source_type = str(row.get('source_type', 'unknown'))
        camera_id = _resolve_bootstrap_camera_id(config, raw_camera_id, source_type)

        if only_if_missing and calibrator.is_cache_scale_validated(camera_id):
            logger.info('Calibration bootstrap skip: profile %s already validated.', camera_id)
            continue

        image_paths = row.get('image_paths', []) or []
        if isinstance(image_paths, str):
            image_paths = [image_paths]
        candidates = _expand_image_candidates([str(x) for x in image_paths], max_images=max_images)

        applied = False
        for path in candidates:
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                continue
            scale, source = calibrator.calibrate_and_store(image=img, camera_id=camera_id)
            if scale is None or source is None:
                continue
            logger.info(
                'Calibration bootstrap applied: camera_id=%s mm_per_px=%.6f source=%s image=%s',
                camera_id,
                float(scale),
                source,
                path,
            )
            applied = True
            break

        if applied:
            continue

        manual_mm_per_px = row.get('manual_mm_per_px', None)
        if manual_mm_per_px is not None:
            try:
                mm = float(manual_mm_per_px)
                if 0.0 < mm <= 2.0:
                    calibrator.upsert_scale(camera_id=camera_id, mm_per_px=mm, fingerprint='manual_bootstrap')
                    logger.warning(
                        'Calibration bootstrap fallback manual profile: camera_id=%s mm_per_px=%.6f',
                        camera_id,
                        mm,
                    )
                    applied = True
            except Exception:
                applied = False

        if not applied:
            logger.warning(
                'Calibration bootstrap failed: camera_id=%s source_type=%s (no valid calibration image found).',
                camera_id,
                source_type,
            )


app = FastAPI(
    title='Agro AI System',
    version='2.0.0',
    description='SaaS-ready AI platform for plant organ segmentation, morphometry and diagnostics.',
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


@app.middleware('http')
async def force_utf8_for_frontend_assets(request: Request, call_next):
    response = await call_next(request)
    path = request.url.path
    content_type = response.headers.get('content-type', '')
    if path.startswith('/frontend/'):
        lower_ct = content_type.lower()
        is_text_like = lower_ct.startswith('text/') or 'javascript' in lower_ct or lower_ct == ''
        if is_text_like and 'charset=' not in lower_ct:
            base_ct = content_type or (
                'application/javascript' if path.endswith('.js') else 'text/css' if path.endswith('.css') else 'text/html'
            )
            response.headers['content-type'] = f'{base_ct}; charset=utf-8'
    return response


frontend_dir = BASE_DIR / 'frontend'
if frontend_dir.exists():
    app.mount('/frontend', StaticFiles(directory=str(frontend_dir), html=True), name='frontend')
outputs_dir = BASE_DIR / 'outputs'
if outputs_dir.exists():
    app.mount('/outputs', StaticFiles(directory=str(outputs_dir)), name='outputs')

instrumentator = Instrumentator().instrument(app)
instrumentator.expose(app, include_in_schema=False, endpoint='/metrics')


def _setup_user_audit_logger(logs_dir: str = 'outputs/logs') -> logging.Logger:
    path = (BASE_DIR / logs_dir).resolve()
    path.mkdir(parents=True, exist_ok=True)
    log_file = path / 'user_audit.log'

    audit_logger = logging.getLogger('user_audit')
    audit_logger.setLevel(logging.INFO)
    audit_logger.propagate = False

    has_handler = False
    for handler in audit_logger.handlers:
        base_file = getattr(handler, 'baseFilename', '')
        if base_file and Path(base_file).resolve() == log_file:
            has_handler = True
            break

    if not has_handler:
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
        )
        rotating = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=5 * 1024 * 1024,
            backupCount=5,
            encoding='utf-8',
        )
        rotating.setFormatter(formatter)
        audit_logger.addHandler(rotating)

    return audit_logger


@app.on_event('startup')
async def startup_event() -> None:
    os.chdir(BASE_DIR)

    config = load_app_config('configs/app.yaml')
    setup_logging(config['app'].get('log_level', 'INFO'))
    app.state.user_audit_logger = _setup_user_audit_logger('outputs/logs')
    set_global_seed(int(config.get('seed', {}).get('value', 42)))
    os.environ.setdefault('DATABASE_URL', str(config.get('database', {}).get('url', 'sqlite:///./data/agro_ai.db')))
    os.environ.setdefault('REDIS_URL', str(config.get('queue', {}).get('redis_url', 'redis://127.0.0.1:6379/0')))
    init_db()

    model_service = ModelService(config)
    if os.getenv('DISABLE_MODEL_LOAD', '0') != '1':
        await model_service.load()

    calibrator = ScaleCalibrator(
        cache_path=config['calibration']['cache_path'],
        default_mm_per_px=float(config['morphometry']['default_mm_per_px']),
        board_size=tuple(config['calibration']['board_size']),
        board_size_candidates=[
            tuple(item)
            for item in config.get('calibration', {}).get('board_size_candidates', [])
            if isinstance(item, (list, tuple)) and len(item) == 2
        ],
        square_size_mm=float(config['calibration']['square_size_mm']),
        charuco_enabled=bool(config.get('calibration', {}).get('charuco', {}).get('enabled', False)),
        charuco_squares_x=int(config.get('calibration', {}).get('charuco', {}).get('squares_x', 5)),
        charuco_squares_y=int(config.get('calibration', {}).get('charuco', {}).get('squares_y', 7)),
        charuco_square_size_mm=float(config.get('calibration', {}).get('charuco', {}).get('square_size_mm', 8.0)),
        charuco_marker_size_mm=float(config.get('calibration', {}).get('charuco', {}).get('marker_size_mm', 6.0)),
        charuco_dictionary=str(config.get('calibration', {}).get('charuco', {}).get('dictionary', 'DICT_4X4_50')),
        auto_profile_enabled=bool(config.get('calibration', {}).get('auto_profile', {}).get('enabled', True)),
        auto_min_samples=int(config.get('calibration', {}).get('auto_profile', {}).get('min_samples', 3)),
        auto_stable_samples=int(config.get('calibration', {}).get('auto_profile', {}).get('stable_samples', 8)),
        auto_max_cv=float(config.get('calibration', {}).get('auto_profile', {}).get('max_cv', 0.35)),
        scene_aware_cache_enabled=bool(config.get('calibration', {}).get('scene_aware_cache', {}).get('enabled', True)),
        scene_hash_size=int(config.get('calibration', {}).get('scene_aware_cache', {}).get('hash_size', 8)),
        scene_max_hamming_distance=int(
            config.get('calibration', {}).get('scene_aware_cache', {}).get('max_hamming_distance', 4)
        ),
        allow_legacy_cache_without_scene=bool(
            config.get('calibration', {}).get('scene_aware_cache', {}).get('allow_legacy_cache_without_scene', False)
        ),
    )
    _bootstrap_calibration_profiles(config=config, calibrator=calibrator)
    history_service = RunHistoryService(
        config.get('analytics', {}).get('run_history_path', 'outputs/run_history.json')
    )
    model_registry = ModelRegistryService(
        config.get('registry', {}).get('model_registry_path', 'models/registry.json')
    )
    dataset_registry = DatasetRegistryService(
        config.get('registry', {}).get('dataset_registry_path', 'data/datasets_registry.json')
    )
    model_registry.ensure_registered(
        str(config.get('model', {}).get('weights', 'models/best.pt')),
        source='startup-bootstrap',
    )

    class_colors = {k: tuple(v) for k, v in config['inference'].get('class_colors', {}).items()}
    phi_service = PHIService(config['morphometry'].get('recommendation_thresholds', {}))
    xai_service = XAIService()
    plantcv_service = PlantCVService()
    active_learning_service = ActiveLearningService(
        root_dir=str(config.get('active_learning', {}).get('root_dir', 'data/active_learning')),
        low_conf_threshold=float(config.get('active_learning', {}).get('low_conf_threshold', 0.12)),
    )

    app.state.config = config
    app.state.model_service = model_service
    app.state.insight_service = InsightService(config.get('chat_reply', {}))
    app.state.history_service = history_service
    app.state.model_registry = model_registry
    app.state.dataset_registry = dataset_registry
    app.state.auth_service = AuthService()
    app.state.chat_service = ChatService()
    app.state.consultation_service = ConsultationService()
    app.state.plantcv_service = plantcv_service
    app.state.blind_eval_service = BlindEvaluationService(model_service=model_service, class_colors=class_colors)
    app.state.robustness_service = RobustnessService(
        model_service=model_service,
        class_colors=class_colors,
        overlay_alpha=float(config['inference'].get('overlay_alpha', 0.45)),
    )
    storage_service = StorageService(
        config['inference']['output_root'],
        retention_cfg=config.get('inference', {}).get('storage_retention', {}),
        logger=logging.getLogger('services.storage_service'),
    )
    app.state.storage_service = storage_service
    app.state.active_learning_service = active_learning_service
    app.state.growth_tracking_service = GrowthTrackingService(
        model_service=model_service,
        calibrator=calibrator,
        storage=storage_service,
        phi_service=phi_service,
        config=config,
    )
    try:
        app.state.job_service = JobService(queue_name='default')
        app.state.queue_available = True
    except Exception as exc:
        logger.warning('RQ queue is not available at startup: %s', exc)
        app.state.job_service = None
        app.state.queue_available = False

    app.state.inference_service = InferenceService(
        model_service=model_service,
        calibrator=calibrator,
        storage=storage_service,
        reporter=ReportService(),
        recommender=RecommendationService(config['morphometry'].get('recommendation_thresholds', {})),
        config=config,
        history_service=history_service,
        phi_service=phi_service,
        plantcv_service=plantcv_service,
        xai_service=xai_service,
        active_learning_service=active_learning_service,
    )
    app.state.calibrator = calibrator
    logger.info('Application startup complete.')


@app.exception_handler(AgroAIError)
async def agro_error_handler(_, exc: AgroAIError):
    return JSONResponse(status_code=400, content={'detail': str(exc)})


@app.exception_handler(Exception)
async def unhandled_error_handler(request, exc: Exception):
    logger.exception('Unhandled exception at %s %s: %s', request.method, request.url.path, exc)
    return JSONResponse(status_code=500, content={'detail': 'Internal server error'})


def _resolve_tenant(tenant_id: str | None) -> str:
    default_tenant = str(app.state.config.get('saas', {}).get('default_tenant', 'default'))
    if tenant_id and tenant_id.strip():
        return tenant_id.strip()
    return default_tenant


def _extract_token(authorization: str | None) -> str | None:
    if not authorization:
        return None
    parts = authorization.strip().split(' ', 1)
    if len(parts) == 2 and parts[0].lower() == 'bearer':
        return parts[1].strip()
    return None


def _resolve_user_from_auth_header(
    db: Session,
    authorization: str | None,
    required: bool = False,
) -> User | None:
    token = _extract_token(authorization)
    if not token:
        if required:
            raise HTTPException(status_code=401, detail='Authorization required.')
        return None

    auth_service: AuthService = app.state.auth_service
    try:
        payload = auth_service.decode_token(token)
        user_id = int(payload.get('sub'))
    except Exception as exc:
        if required:
            raise HTTPException(status_code=401, detail='Invalid token.') from exc
        return None

    user = auth_service.get_user_by_id(db, user_id)
    if user is None and required:
        raise HTTPException(status_code=401, detail='User not found.')
    return user


@app.get('/health', response_model=HealthResponse)
async def health() -> HealthResponse:
    model_service: ModelService = app.state.model_service
    config = app.state.config
    storage_service: StorageService | None = getattr(app.state, 'storage_service', None)
    storage = storage_service.health_status() if storage_service is not None else {}
    health_status = 'degraded' if bool(storage.get('low_space', False)) else 'ok'
    return HealthResponse(
        status=health_status,
        model_loaded=model_service.is_loaded(),
        model_path=str(config['model']['weights']),
        storage_free_gb=storage.get('free_gb'),
        storage_total_gb=storage.get('total_gb'),
        storage_low_space=bool(storage.get('low_space', False)),
        storage_min_free_gb=storage.get('min_free_gb'),
        run_dirs_count=storage.get('run_dirs_count'),
    )


@app.get('/calibration/profiles', response_model=CalibrationProfileListResponse)
async def calibration_profiles(
    validated_only: bool = Query(True),
) -> CalibrationProfileListResponse:
    calibrator: ScaleCalibrator = app.state.calibrator
    items = [CalibrationProfileResponse(**row) for row in calibrator.list_profiles(validated_only=validated_only)]
    return CalibrationProfileListResponse(count=len(items), items=items)


@app.get('/calibration/profile', response_model=CalibrationProfileResponse)
async def calibration_profile(
    camera_id: str = Query('default'),
    source_type: str = Query('lab_camera'),
) -> CalibrationProfileResponse:
    inference_service: InferenceService = app.state.inference_service
    calibrator: ScaleCalibrator = app.state.calibrator
    resolved_camera_id = inference_service._resolve_camera_id(camera_id=camera_id, source_type=source_type)
    profile = calibrator.get_profile(resolved_camera_id)
    if profile is None:
        raise HTTPException(status_code=404, detail=f'Calibration profile not found for camera_id={resolved_camera_id}.')
    return CalibrationProfileResponse(**profile)


@app.post('/calibration/profile/auto', response_model=CalibrationProfileResponse)
async def calibration_profile_auto(
    calibration_image: UploadFile = File(...),
    camera_id: str = Form('default'),
    source_type: str = Form('lab_camera'),
) -> CalibrationProfileResponse:
    raw = await calibration_image.read()
    image = decode_image_bytes(raw)
    if image is None:
        raise HTTPException(status_code=400, detail='Cannot decode calibration image.')

    inference_service: InferenceService = app.state.inference_service
    calibrator: ScaleCalibrator = app.state.calibrator
    resolved_camera_id = inference_service._resolve_camera_id(camera_id=camera_id, source_type=source_type)
    scale, source = calibrator.calibrate_and_store(image=image, camera_id=resolved_camera_id)
    if scale is None or source is None:
        raise HTTPException(
            status_code=422,
            detail='Failed to detect valid checkerboard/Charuco on calibration image.',
        )
    profile = calibrator.get_profile(resolved_camera_id)
    if profile is None:
        profile = {
            'camera_id': resolved_camera_id,
            'mm_per_px': float(scale),
            'validated': True,
            'fingerprint': '',
            'calibration_source': str(source),
            'updated_at': '',
        }
    return CalibrationProfileResponse(**profile)


@app.post('/calibration/profile/manual', response_model=CalibrationProfileResponse)
async def calibration_profile_manual(
    mm_per_px: float = Form(..., gt=0.0, le=2.0),
    camera_id: str = Form('default'),
    source_type: str = Form('lab_camera'),
) -> CalibrationProfileResponse:
    inference_service: InferenceService = app.state.inference_service
    calibrator: ScaleCalibrator = app.state.calibrator
    resolved_camera_id = inference_service._resolve_camera_id(camera_id=camera_id, source_type=source_type)
    calibrator.upsert_scale(resolved_camera_id, mm_per_px, fingerprint='manual_api')
    profile = calibrator.get_profile(resolved_camera_id)
    if profile is None:
        raise HTTPException(status_code=500, detail='Failed to save calibration profile.')
    return CalibrationProfileResponse(**profile)


@app.post('/auth/register', response_model=TokenResponse)
async def auth_register(request: UserRegisterRequest, db: Session = Depends(get_db)) -> TokenResponse:
    auth_service: AuthService = app.state.auth_service
    try:
        user = auth_service.register(db, request.email, request.password)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    token = auth_service.create_access_token(user.id, user.email)
    return TokenResponse(
        access_token=token,
        user=UserResponse(id=user.id, email=user.email),
    )


@app.post('/auth/login', response_model=TokenResponse)
async def auth_login(request: UserLoginRequest, db: Session = Depends(get_db)) -> TokenResponse:
    auth_service: AuthService = app.state.auth_service
    try:
        user = auth_service.login(db, request.email, request.password)
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc

    token = auth_service.create_access_token(user.id, user.email)
    return TokenResponse(
        access_token=token,
        user=UserResponse(id=user.id, email=user.email),
    )


@app.get('/auth/me', response_model=UserResponse)
async def auth_me(
    authorization: str | None = Header(default=None),
    db: Session = Depends(get_db),
) -> UserResponse:
    user = _resolve_user_from_auth_header(db, authorization, required=True)
    return UserResponse(id=user.id, email=user.email)


@app.post('/auth/change-password')
async def auth_change_password(
    request: ChangePasswordRequest,
    authorization: str | None = Header(default=None),
    db: Session = Depends(get_db),
):
    user = _resolve_user_from_auth_header(db, authorization, required=True)
    auth_service: AuthService = app.state.auth_service
    try:
        auth_service.change_password(db, user_id=user.id, old_password=request.old_password, new_password=request.new_password)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {'status': 'ok'}


@app.post('/chat/sessions', response_model=ChatSessionResponse)
async def create_chat_session(
    request: ChatSessionCreateRequest,
    authorization: str | None = Header(default=None),
    db: Session = Depends(get_db),
) -> ChatSessionResponse:
    user = _resolve_user_from_auth_header(db, authorization, required=True)
    chat_service: ChatService = app.state.chat_service
    session = chat_service.create_session(db, user_id=user.id, title=request.title)
    return ChatSessionResponse(
        id=session.id,
        title=session.title,
        created_at=session.created_at.isoformat() if session.created_at else '',
        updated_at=session.updated_at.isoformat() if session.updated_at else '',
    )


@app.get('/chat/sessions', response_model=list[ChatSessionResponse])
async def list_chat_sessions(
    limit: int = Query(100, ge=1, le=500),
    authorization: str | None = Header(default=None),
    db: Session = Depends(get_db),
) -> list[ChatSessionResponse]:
    user = _resolve_user_from_auth_header(db, authorization, required=True)
    chat_service: ChatService = app.state.chat_service
    return chat_service.list_sessions(db, user_id=user.id, limit=limit)


@app.delete('/chat/sessions/{session_id}')
async def delete_chat_session(
    session_id: int,
    authorization: str | None = Header(default=None),
    db: Session = Depends(get_db),
):
    user = _resolve_user_from_auth_header(db, authorization, required=True)
    chat_service: ChatService = app.state.chat_service
    removed = chat_service.delete_session(db, user_id=user.id, session_id=session_id)
    if not removed:
        raise HTTPException(status_code=404, detail='Chat session not found.')
    return {'status': 'deleted', 'session_id': session_id}


@app.get('/chat/sessions/{session_id}/messages', response_model=list[ChatMessageResponse])
async def get_chat_messages(
    session_id: int,
    limit: int = Query(200, ge=1, le=1000),
    authorization: str | None = Header(default=None),
    db: Session = Depends(get_db),
) -> list[ChatMessageResponse]:
    user = _resolve_user_from_auth_header(db, authorization, required=True)
    chat_service: ChatService = app.state.chat_service
    try:
        return chat_service.get_messages(db, user_id=user.id, session_id=session_id, limit=limit)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get('/chat/search', response_model=list[ChatSearchHitResponse])
async def search_chat_messages(
    query: str = Query(..., min_length=1, max_length=120),
    limit: int = Query(50, ge=1, le=200),
    authorization: str | None = Header(default=None),
    db: Session = Depends(get_db),
) -> list[ChatSearchHitResponse]:
    user = _resolve_user_from_auth_header(db, authorization, required=True)
    chat_service: ChatService = app.state.chat_service
    return chat_service.search_messages(
        db=db,
        user_id=user.id,
        query=query,
        limit=limit,
    )


@app.get('/chat/share/{session_id}', response_model=ShareChatResponse)
async def share_chat(session_id: int, db: Session = Depends(get_db)) -> ShareChatResponse:
    chat_service: ChatService = app.state.chat_service
    try:
        messages = chat_service.get_messages(db, user_id=None, session_id=session_id, limit=500, allow_any=True)
        session = chat_service.get_session(db, user_id=None, session_id=session_id, allow_any=True)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return ShareChatResponse(
        session_id=session_id,
        title=session.title if session else 'Shared chat',
        messages=messages,
    )


@app.post('/predict', response_model=PredictResponse)
async def predict(
    image: UploadFile = File(...),
    crop: str = Form('Unknown'),
    source_type: str = Form('lab_camera'),
    camera_id: str = Form('default'),
    conf: float | None = Form(default=None),
    iou: float | None = Form(default=None),
    max_det: int | None = Form(default=None),
    use_ensemble: bool = Form(default=False),
    calibration_image: UploadFile | None = File(default=None),
    tenant_id: str | None = Header(default=None, alias='X-Tenant-ID'),
    authorization: str | None = Header(default=None),
    db: Session = Depends(get_db),
) -> PredictResponse:
    audit_logger: logging.Logger = app.state.user_audit_logger
    user = _resolve_user_from_auth_header(db, authorization, required=False)
    effective_tenant = _resolve_tenant(tenant_id)
    if user is not None:
        effective_tenant = f'user_{user.id}'

    image_bytes = await image.read()
    calibration_bytes = await calibration_image.read() if calibration_image else None

    inference_service: InferenceService = app.state.inference_service
    result = await inference_service.run_single(
        image_bytes=image_bytes,
        image_name=image.filename or 'input.png',
        crop=crop,
        calibration_bytes=calibration_bytes,
        camera_id=camera_id,
        tenant_id=effective_tenant,
        source_type=source_type,
        conf=conf,
        iou=iou,
        max_det=max_det,
        use_ensemble=use_ensemble,
    )
    audit_logger.info(
        'predict | user=%s tenant=%s image=%s crop=%s run_id=%s count=%s mode=%s',
        user.id if user else 'guest',
        effective_tenant,
        image.filename or 'input.png',
        crop,
        result.run_id,
        len(result.measurements),
        (result.summary or {}).get('inference_mode', 'unknown'),
    )
    return result


@app.post('/chat/analyze', response_model=ChatAnalyzeResponse)
async def chat_analyze(
    image: UploadFile = File(...),
    crop: str = Form('Unknown'),
    message: str = Form('Проведи диагностику растения по фото.'),
    session_id: int | None = Form(default=None),
    source_type: str = Form('lab_camera'),
    camera_id: str = Form('default'),
    conf: float | None = Form(default=None),
    iou: float | None = Form(default=None),
    max_det: int | None = Form(default=None),
    use_ensemble: bool = Form(default=False),
    calibration_image: UploadFile | None = File(default=None),
    tenant_id: str | None = Header(default=None, alias='X-Tenant-ID'),
    authorization: str | None = Header(default=None),
    db: Session = Depends(get_db),
) -> ChatAnalyzeResponse:
    audit_logger: logging.Logger = app.state.user_audit_logger
    user = _resolve_user_from_auth_header(db, authorization, required=False)
    effective_tenant = _resolve_tenant(tenant_id)

    chat_service: ChatService = app.state.chat_service
    resolved_session_id = session_id
    prior_messages: list[str] = []

    if user is not None:
        effective_tenant = f'user_{user.id}'
        if resolved_session_id is None:
            created = chat_service.create_session(db, user_id=user.id, title='Диагностика растения')
            resolved_session_id = created.id
        else:
            if chat_service.get_session(db, user_id=user.id, session_id=resolved_session_id) is None:
                raise HTTPException(status_code=404, detail='Chat session not found.')
            history = chat_service.get_messages(db, user_id=user.id, session_id=resolved_session_id, limit=16)
            prior_messages = [m.content for m in history[-10:]]
        chat_service.append_message(
            db,
            user_id=user.id,
            session_id=resolved_session_id,
            role='user',
            content=message,
        )

    image_bytes = await image.read()
    calibration_bytes = await calibration_image.read() if calibration_image else None
    safe_message = ' '.join((message or '').strip().split())
    safe_message = safe_message[:1000]
    audit_logger.info(
        'chat_input | user=%s tenant=%s session=%s crop=%s image=%s text="%s"',
        user.id if user else 'guest',
        effective_tenant,
        resolved_session_id,
        crop,
        image.filename or 'input.png',
        safe_message,
    )

    inference_service: InferenceService = app.state.inference_service
    try:
        result = await inference_service.run_single(
            image_bytes=image_bytes,
            image_name=image.filename or 'input.png',
            crop=crop,
            calibration_bytes=calibration_bytes,
            camera_id=camera_id,
            tenant_id=effective_tenant,
            source_type=source_type,
            conf=conf,
            iou=iou,
            max_det=max_det,
            use_ensemble=use_ensemble,
        )
    except Exception as exc:
        audit_logger.exception(
            'chat_inference_error | user=%s tenant=%s session=%s image=%s error=%s',
            user.id if user else 'guest',
            effective_tenant,
            resolved_session_id,
            image.filename or 'input.png',
            exc,
        )
        if isinstance(exc, OSError) and exc.errno in {28, errno.ENOSPC}:
            raise HTTPException(
                status_code=507,
                detail='Недостаточно места на диске сервера. Повторите позже или очистите outputs.',
            ) from exc
        raise

    insight_service: InsightService = app.state.insight_service
    reply = insight_service.compose_reply(
        result,
        user_message=message,
        prior_context=prior_messages,
    )

    if user is not None and resolved_session_id is not None:
        chat_service.append_message(
            db,
            user_id=user.id,
            session_id=resolved_session_id,
            role='assistant',
            content=reply,
            run_id=result.run_id,
        )

    audit_logger.info(
        'chat_output | user=%s tenant=%s session=%s run_id=%s count=%s mode=%s',
        user.id if user else 'guest',
        effective_tenant,
        resolved_session_id,
        result.run_id,
        len(result.measurements),
        (result.summary or {}).get('inference_mode', 'unknown'),
    )

    return ChatAnalyzeResponse(assistant_reply=reply, result=result, session_id=resolved_session_id)


@app.post('/chat/message', response_model=ChatTextResponse)
async def chat_message(
    request: ChatTextRequest,
    tenant_id: str | None = Header(default=None, alias='X-Tenant-ID'),
    authorization: str | None = Header(default=None),
    db: Session = Depends(get_db),
) -> ChatTextResponse:
    audit_logger: logging.Logger = app.state.user_audit_logger
    user = _resolve_user_from_auth_header(db, authorization, required=False)
    effective_tenant = _resolve_tenant(tenant_id)

    chat_service: ChatService = app.state.chat_service
    consultant: ConsultationService = app.state.consultation_service
    resolved_session_id = request.session_id
    user_message = request.message.strip()
    if not user_message:
        raise HTTPException(status_code=400, detail='Message must not be empty.')

    prior_messages: list[str] = []
    if user is not None:
        effective_tenant = f'user_{user.id}'
        if resolved_session_id is None:
            created = chat_service.create_session(db, user_id=user.id, title='Консультация по растению')
            resolved_session_id = created.id
        else:
            if chat_service.get_session(db, user_id=user.id, session_id=resolved_session_id) is None:
                raise HTTPException(status_code=404, detail='Chat session not found.')
            history = chat_service.get_messages(db, user_id=user.id, session_id=resolved_session_id, limit=20)
            prior_messages = [m.content for m in history[-12:]]

        chat_service.append_message(
            db,
            user_id=user.id,
            session_id=resolved_session_id,
            role='user',
            content=user_message,
        )

    safe_message = ' '.join(user_message.split())[:1000]
    audit_logger.info(
        'chat_text_input | user=%s tenant=%s session=%s text="%s"',
        user.id if user else 'guest',
        effective_tenant,
        resolved_session_id,
        safe_message,
    )

    reply = consultant.compose(user_message, prior_context=prior_messages)

    if user is not None and resolved_session_id is not None:
        chat_service.append_message(
            db,
            user_id=user.id,
            session_id=resolved_session_id,
            role='assistant',
            content=reply,
        )

    audit_logger.info(
        'chat_text_output | user=%s tenant=%s session=%s chars=%s',
        user.id if user else 'guest',
        effective_tenant,
        resolved_session_id,
        len(reply),
    )
    return ChatTextResponse(assistant_reply=reply, session_id=resolved_session_id)


@app.post('/predict/batch')
async def predict_batch(
    images: list[UploadFile] = File(...),
    crops: str = Form(''),
    source_type: str = Form('lab_camera'),
    camera_id: str = Form('default'),
    conf: float | None = Form(default=None),
    iou: float | None = Form(default=None),
    max_det: int | None = Form(default=None),
    use_ensemble: bool = Form(default=False),
    calibration_image: UploadFile | None = File(default=None),
    tenant_id: str | None = Header(default=None, alias='X-Tenant-ID'),
):
    crop_values = [c.strip() for c in crops.split(',')] if crops else []
    payload = []
    for idx, upload in enumerate(images):
        crop = crop_values[idx] if idx < len(crop_values) and crop_values[idx] else 'Unknown'
        payload.append((await upload.read(), upload.filename or f'img_{idx}.png', crop))

    calibration_bytes = await calibration_image.read() if calibration_image else None

    inference_service: InferenceService = app.state.inference_service
    results = await inference_service.run_batch(
        payload,
        calibration_bytes=calibration_bytes,
        camera_id=camera_id,
        tenant_id=_resolve_tenant(tenant_id),
        source_type=source_type,
        conf=conf,
        iou=iou,
        max_det=max_det,
        use_ensemble=use_ensemble,
    )
    return {'count': len(results), 'items': [r.model_dump() for r in results]}


@app.post('/robustness/stress-test', response_model=RobustnessResponse)
async def robustness_stress_test(
    image: UploadFile = File(...),
) -> RobustnessResponse:
    model_service: ModelService = app.state.model_service
    if not model_service.is_loaded():
        raise HTTPException(status_code=503, detail='Model is not loaded.')

    image_bytes = await image.read()
    img = decode_image_bytes(image_bytes)
    if img is None:
        raise HTTPException(status_code=400, detail='Cannot decode image.')

    run_id = f'stress_{datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")}_{uuid4().hex[:6]}'
    robustness_service: RobustnessService = app.state.robustness_service
    return await robustness_service.evaluate(img, run_id=run_id)


@app.post('/growth/track-series', response_model=GrowthTrackingResponse)
async def growth_track_series(
    images: list[UploadFile] = File(...),
    crop: str = Form('Unknown'),
    frame_interval_hours: float = Form(24.0),
    camera_id: str = Form('default'),
    calibration_image: UploadFile | None = File(default=None),
    tenant_id: str | None = Header(default=None, alias='X-Tenant-ID'),
    authorization: str | None = Header(default=None),
    db: Session = Depends(get_db),
) -> GrowthTrackingResponse:
    model_service: ModelService = app.state.model_service
    if not model_service.is_loaded():
        raise HTTPException(status_code=503, detail='Model is not loaded.')

    files = [(await upload.read(), upload.filename or f'frame_{idx}.png') for idx, upload in enumerate(images)]
    if not files:
        raise HTTPException(status_code=400, detail='No images were provided.')
    calibration_bytes = await calibration_image.read() if calibration_image else None

    user = _resolve_user_from_auth_header(db, authorization, required=False)
    effective_tenant = _resolve_tenant(tenant_id)
    if user is not None:
        effective_tenant = f'user_{user.id}'

    growth_service: GrowthTrackingService = app.state.growth_tracking_service
    return await growth_service.run_series(
        files=files,
        crop=crop,
        camera_id=camera_id,
        frame_interval_hours=frame_interval_hours,
        calibration_bytes=calibration_bytes,
        tenant_id=effective_tenant,
        user_id=user.id if user is not None else None,
        db=db,
    )


@app.get('/active-learning/queue', response_model=ActiveLearningQueueResponse)
async def active_learning_queue(
    status: str = Query('pending'),
    limit: int = Query(200, ge=1, le=2000),
) -> ActiveLearningQueueResponse:
    service: ActiveLearningService = app.state.active_learning_service
    return service.list_items(status=status, limit=limit)


@app.post('/active-learning/export')
async def active_learning_export(
    status: str = Form('pending'),
    out_path: str = Form('data/active_learning/manifests/pending.jsonl'),
):
    service: ActiveLearningService = app.state.active_learning_service
    return service.export_manifest(out_path=out_path, status=status)


@app.post('/active-learning/{item_id}/status')
async def active_learning_set_status(
    item_id: str,
    new_status: str = Form(...),
):
    service: ActiveLearningService = app.state.active_learning_service
    try:
        return service.set_status(item_id=item_id, new_status=new_status)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get('/analytics/runs', response_model=list[RunRecord])
async def analytics_runs(
    limit: int = Query(20, ge=1, le=500),
    crop: str | None = Query(default=None),
    tenant_id: str | None = Header(default=None, alias='X-Tenant-ID'),
) -> list[RunRecord]:
    history: RunHistoryService = app.state.history_service
    return history.list_runs(tenant_id=_resolve_tenant(tenant_id), crop=crop, limit=limit)


@app.get('/analytics/trends', response_model=TrendResponse)
async def analytics_trends(
    crop: str = Query('Wheat'),
    class_name: str = Query('root'),
    metric: str = Query('avg_length_mm'),
    limit: int = Query(30, ge=1, le=500),
    tenant_id: str | None = Header(default=None, alias='X-Tenant-ID'),
) -> TrendResponse:
    history: RunHistoryService = app.state.history_service
    return history.trend(
        crop=crop,
        class_name=class_name,
        metric=metric,
        tenant_id=_resolve_tenant(tenant_id),
        limit=limit,
    )


@app.get('/analytics/compare', response_model=CompareRunsResponse)
async def analytics_compare(
    run_a: str = Query(...),
    run_b: str = Query(...),
    crop: str = Query('Wheat'),
    class_name: str = Query('root'),
    metric: str = Query('avg_length_mm'),
    tenant_id: str | None = Header(default=None, alias='X-Tenant-ID'),
) -> CompareRunsResponse:
    history: RunHistoryService = app.state.history_service
    try:
        return history.compare_runs(
            run_a=run_a,
            run_b=run_b,
            crop=crop,
            class_name=class_name,
            metric=metric,
            tenant_id=_resolve_tenant(tenant_id),
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post('/evaluation/blind')
async def evaluate_blind(request: BlindEvalRequest):
    model_service: ModelService = app.state.model_service
    if not model_service.is_loaded():
        raise HTTPException(status_code=503, detail='Model is not loaded.')

    evaluator: BlindEvaluationService = app.state.blind_eval_service
    try:
        return await evaluator.evaluate(
            data_yaml=request.data_yaml,
            split=request.split,
            max_images=request.max_images,
            iou_sla=request.iou_sla,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get('/models/versions', response_model=list[ModelVersionEntry])
async def model_versions(limit: int = Query(50, ge=1, le=500)) -> list[ModelVersionEntry]:
    registry: ModelRegistryService = app.state.model_registry
    return registry.list_versions(limit=limit)


@app.post('/models/register', response_model=ModelVersionEntry)
async def model_register(request: RegisterModelRequest) -> ModelVersionEntry:
    registry: ModelRegistryService = app.state.model_registry
    return registry.register(request)


@app.get('/models/best', response_model=ModelVersionEntry | None)
async def model_best(metric: str = Query('map50')) -> ModelVersionEntry | None:
    registry: ModelRegistryService = app.state.model_registry
    return registry.best_by_metric(metric=metric)


@app.get('/datasets/versions', response_model=list[DatasetVersionEntry])
async def dataset_versions(limit: int = Query(50, ge=1, le=500)) -> list[DatasetVersionEntry]:
    registry: DatasetRegistryService = app.state.dataset_registry
    return registry.list_versions(limit=limit)


@app.post('/datasets/register', response_model=DatasetVersionEntry)
async def dataset_register(request: RegisterDatasetRequest) -> DatasetVersionEntry:
    registry: DatasetRegistryService = app.state.dataset_registry
    return registry.register(request)


@app.post('/jobs/fine-tune', response_model=JobEnqueueResponse)
async def enqueue_fine_tune(request: FineTuneRequest) -> JobEnqueueResponse:
    job_service: JobService | None = app.state.job_service
    if job_service is None:
        raise HTTPException(status_code=503, detail='Queue service is not available. Start Redis + RQ worker.')

    try:
        job_id = job_service.enqueue_finetune(
            config_path='configs/train.yaml',
            data_yaml=request.data_yaml,
            epochs=request.epochs,
            batch=request.batch,
            imgsz=request.imgsz,
            name=request.name,
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f'Queue enqueue failed: {exc}') from exc
    return JobEnqueueResponse(job_id=job_id, status='queued')


@app.post('/jobs/blind-eval', response_model=JobEnqueueResponse)
async def enqueue_blind_eval(request: BlindEvalRequest) -> JobEnqueueResponse:
    job_service: JobService | None = app.state.job_service
    if job_service is None:
        raise HTTPException(status_code=503, detail='Queue service is not available. Start Redis + RQ worker.')

    try:
        job_id = job_service.enqueue_blind_eval(
            data_yaml=request.data_yaml,
            split=request.split,
            max_images=request.max_images,
            iou_sla=request.iou_sla,
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f'Queue enqueue failed: {exc}') from exc
    return JobEnqueueResponse(job_id=job_id, status='queued')


@app.get('/jobs/{job_id}', response_model=JobStatusResponse)
async def job_status(job_id: str) -> JobStatusResponse:
    job_service: JobService | None = app.state.job_service
    if job_service is None:
        raise HTTPException(status_code=503, detail='Queue service is not available. Start Redis + RQ worker.')

    try:
        status = job_service.status(job_id)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f'Job not found or unavailable: {job_id}') from exc
    return JobStatusResponse(**status)


@app.post('/model/export/onnx')
async def export_onnx():
    model_service: ModelService = app.state.model_service
    if not model_service.is_loaded():
        raise HTTPException(status_code=503, detail='Model is not loaded.')
    path = await model_service.export_onnx('models')
    return {'status': 'ok', 'onnx_path': path}


@app.post('/model/export/int8')
async def export_int8():
    model_service: ModelService = app.state.model_service
    if not model_service.is_loaded():
        raise HTTPException(status_code=503, detail='Model is not loaded.')
    try:
        path = await model_service.export_int8('models')
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f'INT8 export failed: {exc}') from exc
    return {'status': 'ok', 'int8_onnx_path': path}


@app.post('/train/fine-tune', response_model=FineTuneResponse)
async def fine_tune(request: FineTuneRequest) -> FineTuneResponse:
    cmd = [
        sys.executable,
        '-m',
        'training.train_yolo_seg',
        '--config',
        'configs/train.yaml',
        '--data',
        request.data_yaml,
        '--epochs',
        str(request.epochs),
        '--batch',
        str(request.batch),
        '--imgsz',
        str(request.imgsz),
        '--name',
        request.name,
    ]

    asyncio.create_task(
        asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(BASE_DIR),
        )
    )

    return FineTuneResponse(status='started', command=' '.join(cmd))
