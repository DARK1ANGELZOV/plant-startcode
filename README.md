# Agro AI System

Production-ready платформа сегментации и морфометрии растений (Arugula/Wheat) с классами `root`, `stem`, `leaves`.

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

## API (минимум)

- `GET /health` - статус сервиса и модели.
- `POST /auth/register`
- `POST /auth/login`
- `POST /chat/analyze` - текст + изображение + optional calibration image.
- `POST /predict/batch` - batch inference.
- `GET /models/*` - реестр моделей.

Пример `chat/analyze`:

```bash
curl -X POST "http://localhost:8000/chat/analyze" \
  -F "message=Проверь растение" \
  -F "crop=Wheat" \
  -F "image=@data/demo/dataset/images/val/wheat_001.png" \
  -F "calibration_image=@data/demo/calibration_board.png"
```

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

## Для экспертов

- Крупные артефакты (датасеты, веса, outputs, reports) намеренно исключены из git.
- Обучение и оценка воспроизводятся через скрипты и конфиги из репозитория.
- Для итоговой оценки проекта приоритетны:
  - качество сегментации на вашем домене;
  - достоверность метрических измерений;
  - корректное поведение при низкой уверенности модели.
