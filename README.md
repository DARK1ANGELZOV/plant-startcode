# Agro AI System

Готовая к внедрению платформа сегментации и морфометрии растений (Arugula/Wheat) с классами `root`, `stem`, `leaves`.

## Сильные стороны решения

- Единый продуктовый контур: от загрузки фото и сегментации до морфометрии, рекомендаций и отчетов.
- Строгая метрическая политика: без валидной калибровки сервис не подменяет реальность «угаданными» миллиметрами.
- Практичный backend для SaaS: FastAPI + async API + история чатов + auth + batch + очереди задач.
- Инженерная устойчивость: fallback-сценарии, health-check, логирование, мониторинг (Prometheus/Grafana).
- Готовность к дообучению: набор скриптов для hard-negative mining, golden-наборов, повторяемых train/eval циклов.
- Продуктовый UX: chat-like интерфейс, работа как с фото, так и с текстовыми вопросами пользователя.

## Текущий этап продукта

Стадия: **расширенный pre-MVP (демо, близкое к MVP)**.

Что уже готово:
- полный демо-сценарий (загрузка -> анализ -> ответ -> история);
- сегментация и морфометрия с контролем достоверности;
- API, фронтенд, авторизация, хранение истории, экспорт артефактов;
- инфраструктурная база для пилотов (Docker, мониторинг, фоновые задачи).

Что нужно добить до промышленного SLA:
- финальная стабилизация метрик качества на целевом домене данных;
- расширение ручного golden-набора и независимая валидация метрик на полевых кейсах;
- регламент релизного контроля (авто-гейты + приемочные тесты на реальных сценариях заказчика).

## Что внутри

- Сегментация: Ultralytics YOLO-seg, fine-tuning, экспорт в ONNX.
- Backend: FastAPI (async), health-check, batch/chat inference, обработка ошибок.
- Морфометрия:
  - площадь (`px`, `mm2` при валидной калибровке);
  - длина через skeleton + longest path;
  - строгая метрическая политика (без «фейковых» мм при ненадежном масштабе).
- Калибровка:
  - chessboard (`findChessboardCorners` / SB), ChArUco;
  - cache + auto-profile;
  - отдельный benchmark по `mm_per_px`.
- Продуктовые функции:
  - chat-like frontend;
  - auth (SQLite + JWT), история чатов;
  - фоновые задачи (RQ + Redis);
  - мониторинг Prometheus/Grafana.

## Технологии

- Python 3.11+
- FastAPI + Uvicorn
- Ultralytics YOLO
- OpenCV, scikit-image, networkx
- SQLite (SQLAlchemy)
- Redis + RQ
- Prometheus + Grafana

## Структура репозитория

```text
agro_ai_system/
  api/
  calibration/
  configs/
  db/
  frontend/
  inference/
  models/
  monitoring/
  morphometry/
  services/
  task_queue/
  tests/
  tools/
  training/
  utils/
  Dockerfile
  docker-compose.yml
  requirements.txt
  README.md
```

## Быстрый старт (локально)

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

Запуск API:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Запуск worker:

```bash
python -m task_queue.worker
```

Доступ:

- Swagger: `http://localhost:8000/docs`
- Frontend: `http://localhost:8000/frontend`
- Метрики: `http://localhost:8000/metrics`

## Быстрый старт (Docker)

```bash
docker compose up --build
```

Сервисы:

- API: `http://localhost:8000`
- Redis: `localhost:6379`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000`

## Конфигурация

Основной файл: `configs/app.yaml`.

Ключевые параметры:

- `model.weights` - путь к активной модели.
- `model.conf`, `model.iou`, `model.max_det` - пороги детекции.
- `morphometry.metric_policy.strict_scale_required` - строгий метрический режим.
- `calibration.board_size`, `calibration.square_size_mm` - геометрия шахматки.
- `calibration.charuco.*` - параметры ChArUco.
- `calibration.startup_bootstrap.*` - автокалибровка профилей камеры при запуске API.

## API (минимум)

- `GET /health` - статус сервиса и модели.
- `POST /auth/register`
- `POST /auth/login`
- `POST /chat/analyze` - текст + изображение + optional calibration image.
- `POST /predict/batch` - batch inference.
- `POST /calibration/profile/auto` - одноразовая калибровка камеры по шахматке/ChArUco.
- `POST /calibration/profile/manual` - ручная установка проверенного `mm_per_px` для камеры.
- `GET /calibration/profile` и `GET /calibration/profiles` - просмотр активных профилей калибровки.
- `GET /models/*` - реестр моделей.

Пример `chat/analyze`:

```bash
curl -X POST "http://localhost:8000/chat/analyze" \
  -F "message=Проверь растение" \
  -F "crop=Wheat" \
  -F "image=@data/demo/dataset/images/val/wheat_001.png" \
  -F "calibration_image=@data/demo/calibration_board.png"
```

Одноразовая калибровка профиля камеры (после этого можно отправлять фото без шахматки):

```bash
curl -X POST "http://localhost:8000/calibration/profile/auto" \
  -F "camera_id=lab_camera" \
  -F "source_type=lab_camera" \
  -F "calibration_image=@data/demo/calibration_board.png"
```

Автокалибровка при старте:

- По умолчанию API пытается заполнить `lab_camera` из:
  - `data/demo/calibration_board.png`
  - `data/calibration/user_chessboard.png`
  - `data/calibration/predictor_cloud_calib/raw/source/calib`
- Настройка: `calibration.startup_bootstrap` в `configs/app.yaml`.

## Политика измерений (важно)

- Конвертация `px -> mm` разрешена только при валидном источнике масштаба (`chessboard` / `charuco` / доверенный cache).
- Если валидного масштаба нет, длины остаются в `px`, а мм помечаются как недоступные/приблизительные по политике.
- В strict режиме адаптивный «угаданный» масштаб не используется.

## Обучение

Базовый fine-tuning:

```bash
python -m training.train_yolo_seg --config configs/train.yaml
```

Максимальный pipeline:

```bash
python -m training.train_maximum --config configs/train_max.yaml
```

Auto HF pipeline (prepare -> balance -> train -> ranking):

```bash
python -m training.auto_hf_train_pipeline \
  --prepare-out data/hf_multisource_raw_auto \
  --balanced-out data/hf_multisource_balanced_auto \
  --train-config configs/train_final_cpu.yaml \
  --run-name auto_hf_train_v1 \
  --include-weak-100crops \
  --include-plantseg-lesions
```

Hard-negative mining + Golden (цель 500) + ручная валидация:

```bash
# Важно: использовать Python из .venv
.\.venv\Scripts\python.exe -m training.run_hardneg_golden_pipeline \
  --python-exe .\.venv\Scripts\python.exe \
  --model models/best_max.pt \
  --data-root data/hf_multisource_mega10 \
  --hard-out data/hard_mined_rootstem_rs_v2 \
  --train-limit 1000 \
  --val-limit 300 \
  --golden-out data/golden_rootstem_500 \
  --golden-target 500 \
  --golden-min 300 \
  --golden-max 500 \
  --review-out data/golden_rootstem_500_review
```

2-stage class-aware train (root/stem-first -> mixed):

```bash
python -m training.train_maximum \
  --config configs/train_rootstem_2stage_classaware.yaml \
  --stage1-data data/hf_multisource_rootstem_boost/dataset.yaml \
  --stage2-data data/hf_multisource_hardmix/dataset.yaml \
  --name rootstem_classaware_v1 \
  --no-plots
```

Единый раннер по пунктам `1,2,3,6,5`:

```bash
python -m training.run_points_1_2_3_5_6 \
  --golden-target 500 \
  --train-name points123_stage \
  --strict-n 30 \
  --strict-threshold 0.90 \
  --lightweight-gate
```

Release blocker (strict gate + floors по recall/mAP50):

```bash
python -m training.release_guard_strict \
  --candidate-model models/best_candidate_points123_latest.pt \
  --benchmark-data data/hf_multisource_mega10_fast/dataset.yaml \
  --strict-n 30 \
  --strict-threshold 0.90 \
  --lightweight-gate \
  --auto-deploy
```

Ручная проверка (CLI):

```bash
.\.venv\Scripts\python.exe -m training.manual_review_cli \
  --review-csv data/golden_rootstem_400_review/manual_review.csv \
  --reviewer expert_1 \
  --only-pending \
  --open-files
```

Сбор финального validated Golden после ручной проверки:

```bash
.\.venv\Scripts\python.exe -m training.apply_golden_manual_decisions \
  --review-csv data/golden_rootstem_400_review/manual_review.csv \
  --source-root data/golden_rootstem_400 \
  --out data/golden_rootstem_400_validated \
  --min-approved 300
```

Оценка:

```bash
python -m training.evaluate --model models/best.pt --data data/demo/dataset.yaml
```

## Калибровка и benchmark по мм

Fit scale cache:

```bash
python -m tools.calibration_fit_cache \
  --app-config configs/app.yaml \
  --sources-config configs/calibration_datasets.yaml \
  --camera-id lab_camera \
  --min-detections 20 \
  --commit \
  --report reports/calibration_fit_report.json
```

Строгий benchmark:

```bash
python -m tools.calibrated_mm_benchmark \
  --app-config configs/app.yaml \
  --n 40 \
  --seed 42 \
  --out reports/calibrated_mm_benchmark_40.json
```

Примечание: для доски `calib.io 5x8` (squares) OpenCV обычно использует `board_size: [7, 4]` (inner corners).

## Тестирование

```bash
python -m pytest -q
```

Строгий quality gate:

```bash
python -m tools.strict_photo_quality_gate \
  --n 30 \
  --seed 42 \
  --out reports/strict_photo_quality_report_very_strict_30.json \
  --strict-pass-threshold 0.93
```

## Мониторинг

- Prometheus: `monitoring/prometheus/prometheus.yml`
- Grafana: `monitoring/grafana/`

## PlantCV (опционально)

В платформу добавлен опциональный post-analysis слой `services/plantcv_service.py`.

- Если `plantcv` установлен, сервис рассчитывает дополнительные признаки маски.
- Если `plantcv` не установлен, автоматически используется fallback без падения API.

Установка (опционально):

```bash
pip install plantcv
```

## Для экспертов

- Крупные артефакты (датасеты, веса, outputs, reports) намеренно исключены из git.
- Обучение и оценка воспроизводятся через скрипты и конфиги из репозитория.
- Для итоговой оценки проекта приоритетны:
  - качество сегментации на вашем домене;
  - достоверность метрических измерений;
  - корректное поведение при низкой уверенности модели.

## Разметка в Roboflow

- Проект (Instance Segmentation): `https://app.roboflow.com/s-workspace-wlhsh/plant-2f4ay`
- Актуальная версия датасета (строгая фильтрация): `https://app.roboflow.com/s-workspace-wlhsh/plant-2f4ay/2`
- Автоподготовка upload pack: `python -m tools.prepare_roboflow_pack ...`
- Строгая подготовка gold-pack: `python -m tools.prepare_roboflow_gold_pack ...`
- Автозагрузка в Roboflow: `python -m tools.upload_roboflow_pack ...`
- Полная очистка изображений проекта перед перезаливкой: `python -m tools.reset_roboflow_project_images ...`
