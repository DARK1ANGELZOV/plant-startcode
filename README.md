# Agro AI System

Production-ready AI система сегментации и морфометрического анализа растений для культур **Arugula** и **Wheat** с классами **root / stem / leaves**.

## Что реализовано

- YOLO-seg pipeline на базе Ultralytics с fine-tuning предобученной модели.
- FastAPI backend в MVC-стиле (View: API, Controller: services, Model: pydantic schemas/domain logic).
- Async endpoints, startup model loading, health-check, batch inference, error handling, logging.
- Морфометрия после сегментации:
  - площадь в пикселях и мм2;
  - длина через скелетизацию + граф + longest path;
  - калибровка масштаба через `cv2.findChessboardCorners` + fallback + cache.
- Визуализация и артефакты:
  - overlay масок с легендой классов;
  - CSV экспорт измерений;
  - PDF отчет;
  - распределения размеров (plot);
  - сравнение Wheat vs Arugula.
- Product/SaaS расширения:
  - drag-and-drop chat-like UI;
  - auth + регистрация пользователей (SQLite, email/password, JWT);
  - сохранение истории чатов и сессий;
  - трекинг запусков и growth-trend analytics;
  - сравнение run-vs-run по метрикам органов;
  - registry версий моделей и датасетов через API;
  - фоновые очереди задач через RQ + Redis;
  - monitoring `/metrics` + Prometheus + Grafana;
  - blind evaluation с SLA (IoU/Precision/Recall/F1 per class);
  - disease diagnosis (эвристический модуль рисков заболеваний);
  - multi-tenant header (`X-Tenant-ID`) для SaaS сценариев.
- Конфигурация через YAML и фиксирование seed для воспроизводимости.
- Экспорт лучшей модели в ONNX.
- Основа под API-дообучение, облачный деплой, CI/CD и SaaS.

## Актуальный статус обучения (2026-02-28)

- Выполнен полный финальный цикл CPU-дообучения: `configs/train_final_cpu.yaml` (20+8 эпох) с сохранением:
  - `models/best_max.pt`
  - `models/best_max.onnx`
- По текущему локальному датасету `data/hf_multisource_medium` качество остается низким
  (`mAP50/mIoU` близки к нулю), поэтому в API включен **strict model mode**:
  - эвристический fallback отключен;
  - численные морфометрические значения выводятся только при уверенных масках;
  - при низкой уверенности сервис возвращает понятное предупреждение вместо псевдо-цифр.
- Это убирает «рандомные» ответы и делает выдачу честной до расширения/очистки train-data.

---

## Структура проекта

```text
agro_ai_system/
│
├── configs/
├── data/
├── models/
├── training/
├── inference/
├── morphometry/
├── calibration/
├── api/
├── services/
├── frontend/
├── utils/
├── reports/
├── tests/
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

---

## Product Architecture (SaaS-ready)

```text
Web UI (chat + drag&drop)
        |
        v
FastAPI Gateway (async, tenant-aware)
        |
        +--> Inference Service (YOLO-seg + overlay)
        |         |
        |         +--> Morphometry (area + skeleton longest path)
        |         +--> Calibration (chessboard + fallback + cache)
        |
        +--> Insight Service (human-readable recommendations)
        +--> Report Service (PDF/CSV/plots)
        +--> Run History Service (growth tracking, compare runs)
        +--> Model Registry Service (versioning + best model selection)
        +--> Dataset Registry Service (dataset lineage + augmentation meta)
```

Ключевая дифференциация против аналогов:
- детальная organ segmentation (`root/stem/leaves`) вместо только disease label;
- количественная морфометрия в мм/мм2;
- growth analytics во времени и сравнение между прогонами;
- API-first интеграция в агро-ERP/LIMS.

---

## Быстрый старт

### 1) Установка

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

Если на Windows появляется ошибка `ModuleNotFoundError: torch._C` или `No such file or directory` при установке `torch`, используйте short-path workaround:

```bash
mkdir C:\pt
python -m pip install --target C:\pt torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install --target C:\pt --no-deps ultralytics ultralytics-thop onnx
```

И запускайте команды с переменной окружения:

```powershell
$env:PYTHONPATH='C:\pt'
python -m training.train_yolo_robust ...
```

### 2) Генерация demo-датасета (YOLO-seg)

```bash
python data/demo/generate_demo_data.py --train 16 --val 4 --size 640 --seed 42
```

### 3) Запуск API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

API docs: `http://localhost:8000/docs`  
Frontend demo (chat-style): `http://localhost:8000/frontend`
Prometheus metrics: `http://localhost:8000/metrics`

### 3.1) Запуск background worker (RQ)

```bash
python -m task_queue.worker
```

Требуется Redis (`REDIS_URL`, по умолчанию `redis://127.0.0.1:6379/0`).

### 4) Demo инференс из CLI

```bash
python -m inference.demo_run \
  --image data/demo/dataset/images/val/wheat_001.png \
  --crop Wheat \
  --calibration data/demo/calibration_board.png \
  --conf 0.05 \
  --iou 0.5 \
  --max-det 40
```

---

## Команды обучения (YOLO-seg)

### Базовый fine-tuning

```bash
python -m training.train_yolo_seg --config configs/train.yaml
```

### Финальный CPU-пайплайн (рекомендуется для текущего окружения)

```bash
python -m training.train_maximum \
  --config configs/train_final_cpu.yaml \
  --stage1-data data/hf_multisource_medium/dataset.yaml \
  --stage2-data data/hf_multisource_medium/dataset.yaml \
  --name plantvision_final_cpu_v1
```

### Auto-pipeline: HF -> фильтрация/баланс -> train -> ranking

```bash
python -m training.auto_hf_train_pipeline \
  --prepare-out data/hf_multisource_raw_auto \
  --balanced-out data/hf_multisource_balanced_auto \
  --train-config configs/train_final_cpu.yaml \
  --run-name auto_hf_train_v1 \
  --include-weak-100crops \
  --include-plantseg-lesions \
  --update-app-config
```

Скрипт сохраняет:
- `reports/auto_hf_pipeline_report.json`
- `reports/auto_hf_model_ranking.json`
- `data/hf_multisource_balanced_auto/filter_balance_report.json`

### Кастомные параметры

```bash
python -m training.train_yolo_seg \
  --config configs/train.yaml \
  --data data/demo/dataset.yaml \
  --epochs 20 \
  --batch 8 \
  --imgsz 640 \
  --name wheat_arugula_ft
```

### Оценка обученной модели

```bash
python -m training.evaluate --model models/best.pt --data data/demo/dataset.yaml
```

### TensorBoard

```bash
tensorboard --logdir runs/segment
```

---

## Обучение на новых датасетах (HF)

### Что реально используется для `root/stem/leaves`

- `ngaggion/ChronoRoot2` (multi-class root/shoot структуры, `.nii.gz` маски)
- `farmaieu/plantorgans` (pixel masks: stem/leaf/flower/fruit)

### Источники, которые подключены как опциональные

- `devshaheen/100_crops_plants_object_detection_25k_image_dataset`:
  weak-supervision режим, bbox -> полигон класса `leaves`.
- `Voxel51/PlantSeg-Test`:
  содержит polylines болезней листьев, полезен как disease/lesion data, но не прямой organ-seg для `root/stem/leaves`.

### Источники, не совместимые напрямую с organ-seg

- `mohanty/PlantVillage`, `leoho36/plant_village_dataset`:
  классификация, без organ-mask.
- `harpreetsahota/crops3d`:
  3D point clouds.
- `Saon110/bd-crop-vegetable-plant-disease-dataset`:
  gated dataset, нужен HF token и это не organ-seg.

### Сборка объединенного YOLO-seg датасета

```bash
python -m training.prepare_multisource_dataset \
  --out data/hf_multisource_yoloseg \
  --chronoroot-max 1200 \
  --plantorgans-max 4000 \
  --img-max-side 1280 \
  --val-ratio 0.15 \
  --min-area 64 \
  --seed 42
```

Опционально добавить weak-supervision из `100_crops`:

```bash
python -m training.prepare_multisource_dataset \
  --out data/hf_multisource_yoloseg \
  --chronoroot-max 1200 \
  --plantorgans-max 4000 \
  --include-weak-100crops \
  --weak-100crops-max 2500
```

Опционально добавить `PlantSeg-Test` (lesion polylines как weak leaf proxy):

```bash
python -m training.prepare_multisource_dataset \
  --out data/hf_multisource_yoloseg \
  --chronoroot-max 1200 \
  --plantorgans-max 4000 \
  --include-plantseg-lesions \
  --plantseg-max 1000
```

Итоги сборки пишутся в:
- `data/hf_multisource_yoloseg/dataset.yaml`
- `data/hf_multisource_yoloseg/build_stats.json`

### Multi-stage обучение по максимуму

```bash
python -m training.train_maximum \
  --config configs/train_max.yaml \
  --stage1-data data/hf_multisource_yoloseg/dataset.yaml \
  --stage2-data data/demo/dataset.yaml \
  --name hf_max_run
```

Артефакты:
- `models/best_max.pt`
- `models/best_max.onnx`

### Robust training на сложных условиях (SA-1B/COCO/OpenImages/ACDC/WeatherProof/RaidaR/MSeg)

Pipeline реализует:
- progressive training (pretrain -> plant fine-tune -> adverse fine-tune),
- heavy augmentations (blur/noise/fog/rain/jpeg/shadow/crop + MixUp/CutMix),
- multi-domain mixing в одном train loop,
- curriculum learning (рост adverse ratio и силы искажений),
- сравнение минимум 2 архитектур (`yolo_seg`, `deeplabv3`),
- robust validation (clean vs corrupted) и выбор лучшей архитектуры.

Поддерживаемые пути датасетов задаются в `configs/robust_train.yaml`.
Если конкретный датасет отсутствует локально, домен автоматически пропускается без падения пайплайна.

1) Собрать train/val manifest из мультидоменных источников:

```bash
python -m training.build_multidomain_manifest \
  --config configs/robust_train.yaml \
  --output data/robust/train_manifest.jsonl \
  --split train

python -m training.build_multidomain_manifest \
  --config configs/robust_train.yaml \
  --output data/robust/val_manifest.jsonl \
  --split val
```

2) Обучить DeepLabV3 robust:

```bash
python -m training.train_deeplab_robust \
  --config configs/robust_train.yaml \
  --train-manifest data/robust/train_manifest.jsonl \
  --val-manifest data/robust/val_manifest.jsonl \
  --output models/robust
```

3) Обучить YOLO-seg robust:

```bash
python -m training.train_yolo_robust \
  --config configs/robust_train.yaml \
  --pretrain-data data/hf_multisource_medium/dataset.yaml \
  --plant-data data/hf_multisource_medium/dataset.yaml \
  --name robust_yolo \
  --output models/robust
```

4) Сравнить архитектуры и выбрать лучшую:

```bash
python -m training.compare_architectures \
  --yolo-report models/robust/yolo_metrics.json \
  --deeplab-report models/robust/deeplab_metrics.json \
  --output models/robust/best_architecture.json
```

5) Бенчмарк latency (CPU/GPU + ONNX):

```bash
python -m training.benchmark_robust \
  --config configs/robust_train.yaml \
  --architecture auto \
  --output models/robust/benchmark.json
```

6) One-shot оркестратор полного цикла:

```bash
python -m training.run_robust_pipeline \
  --config configs/robust_train.yaml \
  --train-manifest data/robust/train_manifest.jsonl \
  --val-manifest data/robust/val_manifest.jsonl \
  --output-dir models/robust
```

Быстрый smoke-прогон (короткий цикл для проверки окружения):

```powershell
$env:PYTHONPATH='C:\pt'
python -m training.run_robust_pipeline \
  --config configs/robust_train_smoke.yaml \
  --train-manifest outputs/robust_smoke/train_manifest.jsonl \
  --val-manifest outputs/robust_smoke/val_manifest.jsonl \
  --output-dir outputs/robust_smoke/models \
  --max-images-per-domain 32 \
  --skip-deeplab
```

Ключевые артефакты robust-пайплайна:
- `models/robust/deeplab_metrics.json`
- `models/robust/yolo_metrics.json`
- `models/robust/best_architecture.json`
- `models/robust/benchmark.json`
- `models/robust/pipeline_report.json`

### Workflow дообучения и MLOps

1. Зарегистрировать новую версию датасета:
```bash
curl -X POST "http://localhost:8000/datasets/register" \
  -H "Content-Type: application/json" \
  -d "{\"dataset_version\":\"roboflow-v25\",\"source\":\"roboflow\",\"classes\":[\"root\",\"stem\",\"leaves\"],\"augmentation\":{\"mosaic\":0.2,\"hsv_s\":0.5}}"
```

2. Запустить `POST /train/fine-tune` с `data_yaml`.
3. После обучения зарегистрировать новую модель:
```bash
curl -X POST "http://localhost:8000/models/register" \
  -H "Content-Type: application/json" \
  -d "{\"path\":\"models/best_max.pt\",\"metrics\":{\"map50\":0.68,\"precision\":0.71},\"dataset_version\":\"roboflow-v25\",\"source\":\"train_pipeline\"}"
```

4. Получить лучшую модель для rollout:
```bash
curl "http://localhost:8000/models/best?metric=map50"
```

### Калибровочные датасеты для точности длины (Chessboard/ChArUco)

Для физически корректных мм/см2 нужна надежная калибровка масштаба.  
В проект добавлен конфиг источников:
- `configs/calibration_datasets.yaml`

Поддержаны ваши источники:
- Kaggle chessboard (`danielwe14/stereocamera-chessboard-pictures`)
- Mendeley calibration plate
- METRIC (Zenodo)
- MCalib
- KIT fisheye calibration
- Roboflow chess
- GitHub примеры ChArUco/chessboard

1) Проверить локальную готовность источников:

```bash
python -m tools.calibration_sources_status \
  --config configs/calibration_datasets.yaml \
  --out reports/calibration_sources_status.json
```

2) Автозагрузка поддерживаемых источников (`git`, `kaggle`):

```bash
python -m tools.fetch_calibration_sources \
  --config configs/calibration_datasets.yaml \
  --source-ids chessboard_kaggle charuco_examples_whatpity calibcam_examples \
  --skip-existing
```

3) Пакетный fit scale-cache по калибровочным папкам:

```bash
python -m tools.calibration_fit_cache \
  --app-config configs/app.yaml \
  --sources-config configs/calibration_datasets.yaml \
  --source-ids chessboard_kaggle \
  --camera-id lab_camera \
  --min-detections 20 \
  --commit \
  --report reports/calibration_fit_report.json
```

`--commit` нужен для записи в `calibration/scale_cache.json`.  
Без `--commit` скрипт работает как безопасный dry-run и только строит отчет.

4) Строгий mm-per-pixel benchmark (синтетика + реальный calib root):

```bash
python -m tools.calibrated_mm_benchmark \
  --app-config configs/app.yaml \
  --n 40 \
  --seed 42 \
  --out reports/calibrated_mm_benchmark_40.json
```

Скрипт сохраняет:
- synthetic detection rate + MAE/RMSE/MAPE по `mm_per_px`,
- real-world detection rate и dispersion по `data/calibration/predictor_cloud_calib/raw/source/calib`.

Важно: для доски `calib.io 5x8` (squares) в OpenCV обычно используется `board_size: [7, 4]` (inner corners).  
Это уже выставлено в `configs/app.yaml`.

### Важная логика точности длины

- Если шахматка/ChArUco не найдены, сервис не использует stale-cache для `camera_id=default` (по умолчанию).
- Ответ помечает такие длины как приблизительные (`~... мм`, с указанием `px`).
- PHI и рекомендации переходят в осторожный режим без жестких биологических выводов по ненадежной калибровке.
- Поддержан `charuco` в `configs/app.yaml` + детекция `source=charuco`.

---

## Docker

```bash
docker compose up --build
```

Сервисы:
- API: `http://localhost:8000`
- Redis: `localhost:6379`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000` (admin/admin)

### Cloud Deployment Blueprint

- `api` контейнер (stateless FastAPI).
- `worker-train` контейнер (отдельный пул для fine-tuning).
- Object Storage (S3/MinIO) для `outputs/` и model artifacts.
- Postgres (опционально) как замена JSON registry/history для multi-tenant SaaS.
- Nginx/API Gateway:
  - rate limit,
  - auth,
  - tenant routing через `X-Tenant-ID`.

Минимальный production pipeline:
1. Build and scan Docker image.
2. Run `pytest`.
3. Push image to registry.
4. Blue/green deploy inference API.
5. Health-check + smoke `/chat/analyze`.

---

## API Endpoints

- `GET /health` - health-check + статус модели.
- `POST /auth/register`, `POST /auth/login`, `GET /auth/me` - авторизация.
- `POST /chat/sessions`, `GET /chat/sessions`, `GET /chat/sessions/{id}/messages` - история чатов.
- `POST /chat/analyze` - чатовый анализ изображения с понятным текстовым объяснением.
- `POST /predict` - одиночный инференс + морфометрия + авто-сохранение результатов.
- `POST /predict/batch` - batch инференс (поддержка нескольких изображений и культур).
- `POST /evaluation/blind` - blind validation + SLA per class.
- `POST /model/export/onnx` - экспорт текущей модели в ONNX.
- `POST /train/fine-tune` - запуск дообучения из API (background process).
- `POST /jobs/fine-tune`, `POST /jobs/blind-eval`, `GET /jobs/{job_id}` - очередь задач RQ.
- `GET /analytics/runs` - история запусков (tenant-aware).
- `GET /analytics/trends` - динамика по `crop/class/metric`.
- `GET /analytics/compare` - сравнение двух запусков.
- `GET /models/versions` / `POST /models/register` / `GET /models/best` - versioning моделей.
- `GET /datasets/versions` / `POST /datasets/register` - versioning датасетов.

### SaaS-параметры

- Tenant isolation: HTTP header `X-Tenant-ID: <tenant_name>`.
- Source metadata: `source_type` (`smartphone|lab_camera|drone|scanner`) + `camera_id`.
- Auth: HTTP header `Authorization: Bearer <JWT>`.

### Пример `curl` для `/chat/analyze`

```bash
curl -X POST "http://localhost:8000/chat/analyze" \
  -F "image=@data/demo/dataset/images/val/arugula_001.png" \
  -F "message=Что не так с растением и что сделать в первую очередь?" \
  -F "crop=Arugula" \
  -F "camera_id=cam_lab_1" \
  -F "conf=0.05" \
  -F "iou=0.5" \
  -F "calibration_image=@data/demo/calibration_board.png"
```

### Пример auth flow

```bash
# register
curl -X POST "http://localhost:8000/auth/register" \
  -H "Content-Type: application/json" \
  -d "{\"email\":\"user@example.com\",\"password\":\"StrongPass123\"}"

# login
curl -X POST "http://localhost:8000/auth/login" \
  -H "Content-Type: application/json" \
  -d "{\"email\":\"user@example.com\",\"password\":\"StrongPass123\"}"
```

Используйте `access_token` в заголовке:
`Authorization: Bearer <token>`.

---

## Робофлоу: разметка и экспорт

### Инструкция разметки в RoboFlow

1. Создать проект Instance Segmentation.
2. Классы: `root`, `stem`, `leaves`.
3. Загружать изображения Arugula/Wheat.
4. Размечать каждый объект отдельной instance-маской (полигоном).
5. Проверить консистентность классов в QA.

### Формат экспорта

- Экспортировать в формате **YOLOv8/YOLO-seg**.
- Структура датасета должна содержать `images/` и `labels/` для train/val.

### Стратегия аугментации

- Геометрия: `degrees`, `translate`, `scale`, `shear`, `fliplr`.
- Цвет: `hsv_h`, `hsv_s`, `hsv_v`.
- Mix-based: `mosaic`, `mixup`.
- Подбирается для устойчивости к вариациям света, угла съемки, шума.

### Версионирование датасета

- Каждую итерацию разметки публиковать как новую версию в RoboFlow.
- В experiments фиксировать:
  - `dataset_version`
  - `train config hash`
  - `best metrics`
- В production использовать pin на конкретную версию датасета.

---

## Метрики и артефакты

После валидации сохраняются:

- IoU per class
- mAP
- Precision
- Recall
- confusion matrix (путь к изображению)
- `runs/segment/.../metrics.json`

Blind evaluation (SLA gate):

```bash
python -m training.evaluate_blind \
  --data data/demo/dataset.yaml \
  --split val \
  --max-images 50 \
  --iou-sla 0.50
```

Отчет сохраняется в `reports/blind_eval_*.json` c:
- IoU/Precision/Recall/F1 per class (`root/stem/leaves`)
- macro metrics
- `sla_pass` по каждому классу.

После инференса сохраняются:

- `overlay.png`
- `measurements.csv`
- `distribution.png`
- `report.pdf`
- `result.json`

---

## Оптимизация под 8 GB GPU

Уже применено в `configs/train.yaml`:

- `batch: 8`
- `imgsz: 640`
- `amp: true`
- `cache: ram`
- `model: yolo11n-seg.pt` (легкий baseline)

Рекомендации при OOM:

1. Снизить `batch` до 4/2.
2. Снизить `imgsz` до 512.
3. Отключить тяжелые аугментации.
4. Использовать градиентное накопление через scheduler workflow.

---

## Масштабирование в будущее

- Добавление новых культур: без изменения API-контрактов (через dataset + config).
- Добавление новых классов: расширение `names` в dataset и color map в `configs/app.yaml`.
- Дообучение через API: endpoint `/train/fine-tune` уже предусмотрен.
- Облачный деплой: Docker-ready, stateless API + volume/object storage.
- CI/CD: тесты + линтер + build/push образа + rollout.
- SaaS-модель: multi-tenant routing, billing, quotas, модельные версии per tenant.

---

# Для жюри

## 1. Problem Statement

Агропредприятиям нужен быстрый и объективный инструмент для оценки состояния растений по изображению, без ручных измерений и субъективной экспертной интерпретации.

## 2. Why This Matters for Precision Agriculture

Точная и регулярная морфометрия (корень/стебель/листья) позволяет:

- оперативно выявлять ранние стрессовые состояния,
- корректировать полив и питание до потерь урожайности,
- стандартизировать контроль качества в теплицах и лабораториях.

## 3. Technical Architecture

- **Segmentation engine:** Ultralytics YOLO-seg (instance segmentation).
- **Backend:** FastAPI async (MVC-style layering).
- **Morphometry:** mask area + skeleton longest-path.
- **Calibration:** chessboard scaling with cache/fallback.
- **Reporting:** overlay + CSV + PDF + distributions.
- **Reproducibility:** YAML configs + fixed seed.

## 4. Why YOLO

Выбор YOLO обоснован:

- высокая скорость инференса на edge/GPU;
- простое масштабирование и deployment pipeline;
- нативная поддержка instance segmentation;
- зрелая экосистема (документация, tooling, export to ONNX, community).

## 5. Scalability Strategy

- Горизонтальное масштабирование API-контейнеров.
- Разделение inference и training в отдельные worker pools.
- Хранение артефактов в object storage.
- Версионирование моделей и staged rollout.
- MLOps-интеграция: метрики, мониторинг drift, retraining loops.

## 6. Results

На реальном датасете (из RoboFlow) система возвращает:

- сегментацию с масками root/stem/leaves;
- количественную морфометрию в мм/мм2;
- class-level quality metrics (IoU, mAP, Precision, Recall, confusion matrix);
- action-oriented рекомендации по состоянию растения.

В репозитории добавлен synthetic demo dataset для технической валидации пайплайна end-to-end.

---

## Демо-сценарий для презентации

1. Сгенерировать demo dataset.
2. Запустить API.
3. Отправить `POST /chat/analyze` с изображением + calibration board.
4. Показать overlay, CSV, PDF и рекомендации.
5. Запустить batch inference для Wheat и Arugula и показать сравнение.

---

## Важные примечания

- Для production-качества требуется обучение на реальном размеченном датасете из RoboFlow.
- Synthetic demo нужен для проверяемого запуска пайплайна без внешних данных.
- При отсутствии шахматной калибровки используется fallback коэффициент и/или cache.
