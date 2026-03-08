# Quality Baseline (2026-03-06)

## Scope
- Endpoint: `/chat/analyze`
- Gate: `tools.strict_photo_quality_gate`
- Dataset root: `data/hf_multisource_mega10/images`
- Mode: `--lightweight`

## Baseline Run
- Command:
  - `python -m tools.strict_photo_quality_gate --n 50 --seed 42 --data-roots data/hf_multisource_mega10/images --out reports/strict_after_line_caps_50_very_strict.json --strict-pass-threshold 0.95 --lightweight`

## Results
- `quality_gate_passed`: `true`
- `global_checks_pass_rate`: `100.0`
- `strict_sample_pass_rate`: `100.0`
- `critical_checks_pass_rate`: `100.0`
- `area_realism_vs_labels`: `100.0`
- `failures`: `0`

## Notes
- Report file in runtime: `reports/strict_after_line_caps_50_very_strict.json`
- `reports/` is git-ignored, metrics are frozen in this markdown for release tracking.
