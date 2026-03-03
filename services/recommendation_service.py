from __future__ import annotations

from collections import defaultdict
from statistics import mean, pstdev

from utils.schemas import Recommendation


class RecommendationService:
    def __init__(self, thresholds: dict) -> None:
        self.thresholds = thresholds or {}

    def generate(
        self,
        measurements: list[dict],
        absolute_scale_reliable: bool = True,
        scale_source: str = 'unknown',
    ) -> list[Recommendation]:
        if not measurements:
            return [
                Recommendation(
                    severity='warning',
                    message='На изображении не найдено сегментированных частей растения.',
                    action='Проверьте фокус, освещение, ракурс, порог confidence и качество калибровки.',
                )
            ]

        grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        counts: dict[str, int] = defaultdict(int)

        for item in measurements:
            crop = item.get('crop', 'Unknown')
            cls = item.get('class_name', 'unknown')
            counts[crop] += 1
            length_mm = item.get('length_mm')
            area_mm2 = item.get('area_mm2')
            if length_mm is not None:
                grouped[crop][cls].append(float(length_mm))
            if area_mm2 is not None:
                grouped[crop][f'{cls}_area'].append(float(area_mm2))

        recommendations: list[Recommendation] = []

        for crop, values in grouped.items():
            crop_thresholds = self.thresholds.get(crop, {})
            root_lengths = values.get('root', [])
            stem_lengths = values.get('stem', [])
            leaf_areas = values.get('leaves_area', [])

            avg_root = mean(root_lengths) if root_lengths else 0.0
            avg_stem = mean(stem_lengths) if stem_lengths else 0.0
            avg_leaf_area = mean(leaf_areas) if leaf_areas else 0.0

            min_root = float(crop_thresholds.get('min_root_length_mm', 15.0))
            min_stem = float(crop_thresholds.get('min_stem_length_mm', 12.0))
            min_leaf_area = float(crop_thresholds.get('min_leaf_area_mm2', 85.0))
            min_leaf_root_ratio = float(crop_thresholds.get('min_leaf_root_ratio', 1.0))
            max_leaf_cv = float(crop_thresholds.get('max_leaf_cv', 0.9))

            leaf_root_ratio = (avg_leaf_area / avg_root) if avg_root > 0 else 0.0
            leaf_cv = (pstdev(leaf_areas) / avg_leaf_area) if len(leaf_areas) > 1 and avg_leaf_area > 0 else 0.0
            noisy_instances = counts[crop] > 35

            if noisy_instances:
                recommendations.append(
                    Recommendation(
                        severity='warning',
                        message=f'{crop}: обнаружено {counts[crop]} инстансов, вероятен шум сегментации.',
                        action='Повысить confidence до 0.01-0.05, снизить max_det и проверить фон/контраст съемки.',
                    )
                )

            if (not absolute_scale_reliable) and crop_thresholds:
                recommendations.append(
                    Recommendation(
                        severity='warning',
                        message=(
                            f'{crop}: перевод в мм невозможен без валидной геометрической калибровки '
                            '(шахматка/линейка/эталон или заранее вычисленный mm_per_pixel для этой камеры).'
                        ),
                        action=(
                            'Используйте текущие px-метрики для сравнений между кадрами. '
                            'Чтобы получить мм, добавьте калибровочный эталон и повторите анализ.'
                        ),
                    )
                )
                continue

            if avg_root < min_root:
                recommendations.append(
                    Recommendation(
                        severity='critical',
                        message=f'{crop}: корневая система недоразвита (avg {avg_root:.2f} мм < {min_root:.2f} мм).',
                        action='Увеличить глубину полива, проверить pH и EC субстрата, добавить стимуляцию корнеобразования.',
                    )
                )

            if avg_stem < min_stem:
                recommendations.append(
                    Recommendation(
                        severity='warning',
                        message=f'{crop}: слабое развитие стебля (avg {avg_stem:.2f} мм < {min_stem:.2f} мм).',
                        action='Проверить баланс азота/калия, плотность посадки и конкуренцию за свет.',
                    )
                )

            if avg_leaf_area < min_leaf_area:
                recommendations.append(
                    Recommendation(
                        severity='warning',
                        message=f'{crop}: недостаточная площадь листьев (avg {avg_leaf_area:.2f} мм2 < {min_leaf_area:.2f} мм2).',
                        action='Проверить азотное питание, PPFD/фотопериод и наличие температурного стресса.',
                    )
                )

            if leaf_root_ratio < min_leaf_root_ratio:
                recommendations.append(
                    Recommendation(
                        severity='warning',
                        message=f'{crop}: дисбаланс лист/корень (ratio {leaf_root_ratio:.2f} < {min_leaf_root_ratio:.2f}).',
                        action='Снизить вегетативную нагрузку и стабилизировать водный режим для усиления корней.',
                    )
                )

            if leaf_cv > max_leaf_cv:
                recommendations.append(
                    Recommendation(
                        severity='warning',
                        message=f'{crop}: высокая неоднородность листовой площади (CV={leaf_cv:.2f}).',
                        action='Проверить равномерность полива и освещения, выровнять микроклимат по секциям.',
                    )
                )

            if (
                avg_root >= min_root
                and avg_stem >= min_stem
                and avg_leaf_area >= min_leaf_area
                and leaf_root_ratio >= min_leaf_root_ratio
                and leaf_cv <= max_leaf_cv
                and not noisy_instances
            ):
                recommendations.append(
                    Recommendation(
                        severity='ok',
                        message=f'{crop}: морфометрические показатели в целевом диапазоне.',
                        action='Сохранять текущий режим полива и питания, мониторить динамику раз в 24 часа.',
                    )
                )

        return recommendations
