# Agro AI System

Production-ready платформа сегментации и морфометрии растений (Arugula/Wheat) с классами `root`, `stem`, `leaves`.

## Scope

- Segmentation: Ultralytics YOLO-seg, fine-tuning, ONNX export.
- Backend: FastAPI (async), health-check, batch/chat inference, structured error handling.
- Morphometry:
  - area (px, mm2 при валидной калибровке);
  - length via skeleton + longest path;
  - strict metric policy (без фейковых мм при ненадежном масштабе).
- Calibration:
  - chessboard (`findChessboardCorners`/SB), ChArUco;
  - cache + auto-profile;
  - отдельный mm benchmark.
- Product features:
  - chat-like frontend;
  - auth (SQLite, JWT), chat history;
  - background jobs (RQ + Redis);
  - Prometheus/Grafana.

## Tech Stack

- Python 3.11+
- FastAPI + Uvicorn
- Ultralytics YOLO
- OpenCV, scikit-image, networkx
- SQLite (SQLAlchemy)
- Redis + RQ
- Prometheus + Grafana

## Repository Layout

```text
agro_ai_system/
├── api/
├── calibration/
├── configs/
├── db/
├── frontend/
├── inference/
├── models/
├── monitoring/
├── morphometry/
├── services/
├── task_queue/
├── tests/
├── tools/
├── training/
├── utils/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Quick Start (Local)

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

Run API:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Run worker:

```bash
python -m task_queue.worker
```

Open:

- API docs: `http://localhost:8000/docs`
- Frontend: `http://localhost:8000/frontend`
- Metrics: `http://localhost:8000/metrics`

## Quick Start (Docker)

```bash
docker compose up --build
```

Services:

- API: `http://localhost:8000`
- Redis: `localhost:6379`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000`

## Configuration

Main config: `configs/app.yaml`

Critical params:

- `model.weights` - active model path.
- `model.conf`, `model.iou`, `model.max_det` - detection thresholds.
- `morphometry.metric_policy.strict_scale_required` - strict metric mode.
- `calibration.board_size`, `calibration.square_size_mm` - chessboard geometry.
- `calibration.charuco.*` - ChArUco settings.

## API (Minimal)

- `GET /health` - service/model status.
- `POST /auth/register`
- `POST /auth/login`
- `POST /chat/analyze` - text + image + optional calibration image.
- `POST /predict/batch` - batch inference.
- `GET /models/*` - model registry endpoints.

Example `chat/analyze`:

```bash
curl -X POST "http://localhost:8000/chat/analyze" \
  -F "message=Проверь растение" \
  -F "crop=Wheat" \
  -F "image=@data/demo/dataset/images/val/wheat_001.png" \
  -F "calibration_image=@data/demo/calibration_board.png"
```

## Measurement Policy (Important)

- Conversion `px -> mm` is allowed only with valid scale source (`chessboard` / `charuco` / trusted cache).
- If valid scale is absent, response must keep lengths in `px` and mark mm values as unavailable/approximate per policy.
- No adaptive guessed scale in strict mode.

## Training

Base fine-tuning:

```bash
python -m training.train_yolo_seg --config configs/train.yaml
```

Maximum pipeline:

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

Evaluation:

```bash
python -m training.evaluate --model models/best.pt --data data/demo/dataset.yaml
```

## Calibration and mm Benchmark

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

Strict mm benchmark:

```bash
python -m tools.calibrated_mm_benchmark \
  --app-config configs/app.yaml \
  --n 40 \
  --seed 42 \
  --out reports/calibrated_mm_benchmark_40.json
```

Note: for `calib.io 5x8` board (squares), OpenCV usually uses `board_size: [7, 4]` (inner corners).

## Testing

```bash
python -m pytest -q
```

Targeted strict quality gate:

```bash
python -m tools.strict_photo_quality_gate \
  --n 30 \
  --seed 42 \
  --out reports/strict_photo_quality_report_very_strict_30.json \
  --strict-pass-threshold 0.93
```

## Monitoring

- Prometheus config: `monitoring/prometheus/prometheus.yml`
- Grafana provisioning and dashboard: `monitoring/grafana/`

## Notes for Reviewers

- Large artifacts (datasets, model weights, outputs, reports) are intentionally excluded from git.
- Reproduce training with provided scripts/configs and your local data storage.
- For production scoring, prioritize:
  - segmentation quality on your domain;
  - calibrated metric trustworthiness;
  - strict handling of low-confidence cases.
