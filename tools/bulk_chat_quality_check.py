from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import httpx


def guess_crop(path: Path) -> str:
    name = path.name.lower()
    if 'wheat' in name:
        return 'Wheat'
    if 'arugula' in name:
        return 'Arugula'
    return 'Unknown'


def collect_images(paths: list[Path]) -> list[Path]:
    exts = {'.png', '.jpg', '.jpeg', '.bmp'}
    out: list[Path] = []
    for base in paths:
        if not base.exists():
            continue
        if base.is_file() and base.suffix.lower() in exts:
            out.append(base)
            continue
        for p in base.rglob('*'):
            if p.is_file() and p.suffix.lower() in exts:
                out.append(p)
    return sorted(out)


def main() -> None:
    parser = argparse.ArgumentParser(description='Run bulk chat-analysis quality checks on many photos.')
    parser.add_argument('--api', default='http://127.0.0.1:8000')
    parser.add_argument('--n', type=int, default=130)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--timeout', type=float, default=120.0)
    parser.add_argument('--out', default='reports/bulk_chat_quality_report.json')
    parser.add_argument(
        '--data-roots',
        nargs='*',
        default=[
            'data/demo/dataset/images',
            'data/hf_multisource_medium/images',
            'data/hf_multisource_quick/images',
        ],
    )
    args = parser.parse_args()

    random.seed(args.seed)
    roots = [Path(p) for p in args.data_roots]
    images = collect_images(roots)
    if not images:
        raise RuntimeError('No input images found for quality run.')

    if len(images) >= args.n:
        chosen = random.sample(images, args.n)
    else:
        chosen = [random.choice(images) for _ in range(args.n)]

    required_phrases = [
        'Результаты анализа изображения:',
        '1. Распознаны структуры растения:',
        '2. Измерения (примерные):',
        '3. Качество изображения:',
        '4. Уверенность модели:',
        '5. Риски и рекомендации:',
    ]

    ok_count = 0
    status_200 = 0
    phrase_all_ok = 0
    measurement_nonempty = 0
    failures: list[dict] = []
    samples: list[dict] = []

    with httpx.Client(timeout=args.timeout) as client:
        for idx, path in enumerate(chosen, start=1):
            crop = guess_crop(path)
            with path.open('rb') as fh:
                files = {'image': (path.name, fh, 'image/png')}
                data = {
                    'message': 'Проведи анализ растения и дай рекомендации по рискам при ухудшении параметров.',
                    'crop': crop,
                }
                try:
                    resp = client.post(f'{args.api}/chat/analyze', files=files, data=data)
                except Exception as exc:
                    failures.append(
                        {
                            'index': idx,
                            'image': str(path),
                            'error': str(exc),
                            'status_code': None,
                        }
                    )
                    continue

            if resp.status_code == 200:
                status_200 += 1
            else:
                failures.append(
                    {
                        'index': idx,
                        'image': str(path),
                        'status_code': resp.status_code,
                        'response': resp.text[:500],
                    }
                )
                continue

            payload = resp.json()
            reply = payload.get('assistant_reply', '')
            result = payload.get('result', {})
            measurements = result.get('measurements', []) or []
            if measurements:
                measurement_nonempty += 1

            missing = [p for p in required_phrases if p not in reply]
            if not missing:
                phrase_all_ok += 1
                ok_count += 1
            else:
                failures.append(
                    {
                        'index': idx,
                        'image': str(path),
                        'status_code': 200,
                        'missing_phrases': missing,
                        'reply_excerpt': reply[:600],
                    }
                )

            if len(samples) < 12:
                samples.append(
                    {
                        'index': idx,
                        'image': str(path),
                        'crop': crop,
                        'status_code': resp.status_code,
                        'measurement_count': len(measurements),
                        'reply_excerpt': reply[:700],
                    }
                )

    report = {
        'requested': args.n,
        'executed': len(chosen),
        'status_200': status_200,
        'structured_phrase_pass': phrase_all_ok,
        'non_empty_measurements': measurement_nonempty,
        'pass_rate': round((ok_count / max(1, len(chosen))) * 100.0, 2),
        'seed': args.seed,
        'api': args.api,
        'samples': samples,
        'failures': failures[:100],
    }

    out_path = Path(args.out)
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    except OSError as exc:
        print(f'Warning: failed to save report to {out_path}: {exc}')

    pretty = json.dumps(report, ensure_ascii=False, indent=2)
    try:
        print(pretty)
    except UnicodeEncodeError:
        # Windows terminals may use cp1251/cp866 and fail on emoji symbols.
        print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()
