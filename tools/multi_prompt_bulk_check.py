from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path
import re

import httpx


def guess_crop(path: Path) -> str:
    name = path.name.lower()
    if 'wheat' in name:
        return 'Wheat'
    if 'arugula' in name:
        return 'Arugula'
    return 'Unknown'


def collect_images(paths: list[Path]) -> list[Path]:
    exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}
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


def build_prompt_pool() -> list[str]:
    return [
        'Проведи диагностику растения по фото и объясни понятным языком, что не так и что делать.',
        'Проанализируй растение. Дай короткий вывод и 3 действия на сегодня.',
        'Что видно по корням, стеблю и листьям? Укажи риски простыми словами.',
        'Оцени состояние растения и скажи, что исправить в поливе и свете.',
        'Сделай разбор фото как агроном: проблемы, причины, шаги.',
        'Мне нужен практичный отчет: что критично, что под контролем, что проверить завтра.',
        'Где слабое место растения? Дай рекомендации без сложных терминов.',
        'Проверь, есть ли признаки стресса, и как быстро это исправить.',
        'Опиши состояние растения в 5 пунктах и добавь предупреждения.',
        'Сделай диагностику и укажи, когда ситуация станет опасной.',
        'Сравни развитие корней и надземной части, если есть дисбаланс.',
        'Проверь качество снимка и скажи, влияет ли это на достоверность вывода.',
        'Нужен понятный отчет для новичка: что плохо и что делать по шагам.',
        'Если есть риск осложнений, напиши через сколько и из-за чего.',
        'Сделай нейтральный профессиональный вывод по растению.',
        'Оцени текущее состояние и предложи план на 24 часа.',
        'Дай рекомендации, чтобы улучшить корневую систему.',
        'Проведи диагностику и добавь рекомендации по повторной съемке.',
        'Насколько растение здорово? Объясни, почему.',
        'Проверь, нет ли признаков недоразвития корней.',
        'Оцени растение максимально честно, без украшений.',
        'Разбери фото и скажи, что нужно контролировать в динамике.',
        'Сделай диагностику в формате: проблема -> причина -> действие.',
        'Проведи анализ и скажи, какие параметры ухудшаются в первую очередь.',
        'Мне нужен ответ для фермера: коротко и по делу.',
        'Проведи диагностику и укажи, что делать в первую очередь сегодня.',
        'Сделай подробный разбор состояния растения и рисков.',
        'Проверь растение и дай рекомендации по свету, поливу, питанию.',
        'Определи, что не так, и предложи план корректировки условий.',
        'Оцени фото как эксперт по фенотипированию растений.',
    ]


def _contains_all_sections(reply: str) -> bool:
    text = str(reply or '')
    if 'Результаты анализа изображения:' not in text:
        return False

    # Accept both legacy and current response templates.
    variants = [
        [
            r'1\.\s*Распознаны структуры растения:',
            r'2\.\s*Измерения\s*\(примерные\):',
            r'3\.\s*Качество изображения:',
            r'4\.\s*Уверенность модели:',
            r'5\.\s*Риски и рекомендации:',
        ],
        [
            r'1\.\s*Сегментация:',
            r'2\.\s*Измерения',
            r'3\.\s*Перевод в мм:',
            r'4\.\s*Вывод:',
            r'5\.\s*Рекомендации:',
        ],
    ]

    for patterns in variants:
        if all(re.search(p, text, flags=re.IGNORECASE) for p in patterns):
            return True
    return False


def _is_finite_non_negative(v: object) -> bool:
    try:
        x = float(v)
    except Exception:
        return False
    return math.isfinite(x) and x >= 0.0


def evaluate_case(payload: dict, reply: str) -> list[str]:
    issues: list[str] = []
    result = payload.get('result') or {}
    summary = result.get('summary') or {}
    measurements = result.get('measurements') or []

    if not _contains_all_sections(reply):
        issues.append('missing_sections')

    if 'Запрос:' not in reply:
        issues.append('missing_request_echo')

    if not isinstance(measurements, list):
        issues.append('invalid_measurements_type')
        return issues

    allowed_classes = {'root', 'stem', 'leaves'}
    for m in measurements:
        cls = str(m.get('class_name', ''))
        if cls not in allowed_classes:
            issues.append('invalid_class_name')
            break
        for k in ('confidence', 'area_px', 'area_mm2', 'length_px', 'length_mm'):
            if not _is_finite_non_negative(m.get(k)):
                issues.append('invalid_numeric_value')
                break

    scale_source = str(result.get('scale_source', ''))
    calibration_reliable = bool(summary.get('calibration_reliable', scale_source in {'chessboard', 'charuco'}))
    if not calibration_reliable:
        if ('без точной шахматки' not in reply) and ('точность не подтверждена' not in reply):
            issues.append('missing_calibration_disclaimer')

    if len(measurements) == 0:
        if ('Не вижу уверенно выделенных частей' not in reply) and ('Структуры не распознаны' not in reply):
            issues.append('empty_without_notice')

    # Request echo is optional and should not fail quality by itself.
    return [x for x in issues if x != 'missing_request_echo']


def main() -> None:
    parser = argparse.ArgumentParser(description='Run multi-prompt chat-analysis quality checks on many photos.')
    parser.add_argument('--api', default='http://127.0.0.1:8000')
    parser.add_argument('--n', type=int, default=180)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--timeout', type=float, default=120.0)
    parser.add_argument('--out', default='reports/multi_prompt_bulk_quality_report.json')
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
        raise RuntimeError('No input images found for multi-prompt run.')

    prompts = build_prompt_pool()
    if len(images) >= args.n:
        chosen_images = random.sample(images, args.n)
    else:
        chosen_images = [random.choice(images) for _ in range(args.n)]

    cases = [(img, random.choice(prompts)) for img in chosen_images]

    status_200 = 0
    pass_count = 0
    failures: list[dict] = []
    samples: list[dict] = []
    issue_counter: Counter[str] = Counter()

    with httpx.Client(timeout=args.timeout) as client:
        for idx, (path, prompt) in enumerate(cases, start=1):
            crop = guess_crop(path)
            with path.open('rb') as fh:
                files = {'image': (path.name, fh, 'image/png')}
                data = {'message': prompt, 'crop': crop}
                try:
                    resp = client.post(f'{args.api}/chat/analyze', files=files, data=data)
                except Exception as exc:
                    failures.append(
                        {
                            'index': idx,
                            'image': str(path),
                            'prompt': prompt,
                            'status_code': None,
                            'error': str(exc),
                        }
                    )
                    issue_counter['request_exception'] += 1
                    continue

            if resp.status_code != 200:
                failures.append(
                    {
                        'index': idx,
                        'image': str(path),
                        'prompt': prompt,
                        'status_code': resp.status_code,
                        'response': resp.text[:600],
                    }
                )
                issue_counter['non_200'] += 1
                continue

            status_200 += 1
            payload = resp.json()
            reply = payload.get('assistant_reply', '')
            issues = evaluate_case(payload, reply)
            if not issues:
                pass_count += 1
            else:
                issue_counter.update(issues)
                failures.append(
                    {
                        'index': idx,
                        'image': str(path),
                        'prompt': prompt,
                        'status_code': 200,
                        'issues': issues,
                        'reply_excerpt': reply[:800],
                    }
                )

            if len(samples) < 20:
                result = payload.get('result') or {}
                summary = result.get('summary') or {}
                samples.append(
                    {
                        'index': idx,
                        'image': str(path),
                        'prompt': prompt,
                        'crop': crop,
                        'status_code': resp.status_code,
                        'measurement_count': len(result.get('measurements') or []),
                        'scale_source': result.get('scale_source'),
                        'calibration_reliable': summary.get('calibration_reliable'),
                        'issues': issues,
                        'reply_excerpt': reply[:700],
                    }
                )

    report = {
        'requested': args.n,
        'executed': len(cases),
        'status_200': status_200,
        'passed': pass_count,
        'pass_rate': round((pass_count / max(1, len(cases))) * 100.0, 2),
        'prompts_pool_size': len(prompts),
        'seed': args.seed,
        'api': args.api,
        'issue_counts': dict(issue_counter),
        'samples': samples,
        'failures': failures[:200],
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')

    pretty = json.dumps(report, ensure_ascii=False, indent=2)
    try:
        print(pretty)
    except UnicodeEncodeError:
        print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()
